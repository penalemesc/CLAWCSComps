import os
import json
from pathlib import Path
from functools import partial
import numpy as np
from PIL import Image
from tqdm import tqdm
from pycocotools import mask as mask_utils

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW

from sam3.model_builder import build_sam3_image_model
from sam3.model.utils.misc import copy_data_to_device
from sam3.train.data.sam3_image_dataset import Sam3ImageDataset
from sam3.train.data.coco_json_loaders import COCO_FROM_JSON
from sam3.train.data.collator import collate_fn_api
from sam3.train.transforms.basic_for_api import (
    ComposeAPI,
    RandomResizeAPI,
    PadToSizeAPI,
    ToTensorAPI,
    NormalizeAPI,
)
from sam3.train.transforms.segmentation import DecodeRle
from sam3.train.transforms.point_sampling import RandomizeInputBbox
from sam3.train.transforms.filter_query_transforms import (
    FlexibleFilterFindGetQueries,
    FilterCrowds,
    FilterEmptyTargets,
)
from sam3.train.matcher import BinaryHungarianMatcherV2, BinaryOneToManyMatcher
from sam3.train.loss.loss_fns import IABCEMdetr, Boxes, Masks, CORE_LOSS_KEY
from sam3.train.loss.sam3_loss import Sam3LossWrapper

# ---------------------------------------------------------------------------
# Fixed Configuration
# ---------------------------------------------------------------------------
PROMPT = "Cars"
DATASET_DIR = Path("/dataset")
CHECKPOINT_PATH = "sam3.pt"
BPE_PATH = "/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz"
OUTPUT_DIR = Path("./checkpoints")

RESOLUTION = 1008
EPOCHS = 20
BATCH_SIZE = 1
GRAD_ACCUM = 4

def build_coco_json(split: str) -> tuple[str, str]:
    images_dir = DATASET_DIR / "images" / split
    masks_dir = DATASET_DIR / "bw_masks" / split
    output_json = DATASET_DIR / f"_sam3_{split}_annotations.coco.json"

    if output_json.exists():
        print(f"[{split.upper()}] Using existing COCO JSON: {output_json}")
        return str(images_dir), str(output_json)

    print(f"[{split.upper()}] Generating COCO annotations from {masks_dir}...")
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": PROMPT}],
    }

    ann_id = 1
    valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
    img_files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith(valid_exts)])

    for img_id, fname in enumerate(tqdm(img_files, desc=f"Indexing {split}"), start=1):
        img_file = images_dir / fname
        mask_file = masks_dir / fname
        if not mask_file.exists():
            mask_file = masks_dir / f"{Path(fname).stem}.png"
        if not mask_file.exists():
            continue

        with Image.open(img_file) as img:
            width, height = img.size

        binary_mask = (np.array(Image.open(mask_file).convert("L")) > 0).astype(np.uint8)
        if binary_mask.sum() == 0:
            continue

        coco_data["images"].append({
            "id": img_id,
            "file_name": fname,
            "width": width,
            "height": height,
        })

        rle = mask_utils.encode(np.asfortranarray(binary_mask))
        rle["counts"] = rle["counts"].decode("utf-8")

        coco_data["annotations"].append({
            "id": ann_id,
            "image_id": img_id,
            "category_id": 1,
            "bbox": mask_utils.toBbox(rle).tolist(),
            "area": float(mask_utils.area(rle)),
            "segmentation": rle,
            "iscrowd": 0,
            "noun_phrase": PROMPT,
        })
        ann_id += 1

    with open(output_json, "w") as f:
        json.dump(coco_data, f, indent=2)

    return str(images_dir), str(output_json)

def build_transforms(is_train: bool = True):
    steps = [
        FlexibleFilterFindGetQueries(query_filter=FilterCrowds()),
        DecodeRle(),
    ]
    if is_train:
        steps.append(RandomizeInputBbox(box_noise_scale=0.1))

    steps.extend([
        RandomResizeAPI(sizes=[RESOLUTION], max_size=RESOLUTION, square=True, consistent_transform=False),
        PadToSizeAPI(size=RESOLUTION, consistent_transform=False),
        ToTensorAPI(),
        FlexibleFilterFindGetQueries(query_filter=FilterEmptyTargets()),
        NormalizeAPI(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return ComposeAPI(transforms=steps)


def build_loss():
    return Sam3LossWrapper(
        matcher=BinaryHungarianMatcherV2(focal=True, cost_class=2.0, cost_bbox=5.0, cost_giou=2.0, alpha=0.25, gamma=2.0),
        o2m_matcher=BinaryOneToManyMatcher(alpha=0.3, threshold=0.4, topk=4),
        o2m_weight=2.0,
        use_o2m_matcher_on_o2m_aux=False,
        loss_fns_find=[
            Boxes(weight_dict={"loss_bbox": 5.0, "loss_giou": 2.0}),
            IABCEMdetr(
                weak_loss=False, pos_weight=10.0, alpha=0.25, gamma=2.0,
                use_presence=True, pos_focal=False,
                weight_dict={"loss_ce": 20.0, "presence_loss": 20.0},
            ),
            Masks(
                focal_alpha=0.25, focal_gamma=2.0, num_sample_points=4096,
                oversample_ratio=3.0, importance_sample_ratio=0.75,
                weight_dict={"loss_mask": 80.0, "loss_dice": 10.0},
            ),
        ],
        normalization="local",
    )

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Build COCO JSONs from ./dataset/images/{train,val} and ./dataset/bw_masks/{train,val}
    train_img_dir, train_ann = build_coco_json("train")
    val_img_dir, val_ann = build_coco_json("val")

    # 2. Load SAM 3 Model
    print("Loading SAM 3 model...")
    model = build_sam3_image_model(
        bpe_path=BPE_PATH,
        device="cpu",
        eval_mode=False,
        checkpoint_path=CHECKPOINT_PATH,
        load_from_HF=False,
        enable_segmentation=True,
    ).to(device)

    # 3. Setup DataLoaders
    coco_loader = partial(COCO_FROM_JSON, include_negatives=True, category_chunk_size=18)
    collate_fn = partial(collate_fn_api, repeats=1, dict_key="all", with_seg_masks=True)

    train_dataset = Sam3ImageDataset(
        img_folder=train_img_dir, ann_file=train_ann,
        transforms=[build_transforms(is_train=True)],
        load_segmentation=True, max_ann_per_img=500000, coco_json_loader=coco_loader,
    )
    val_dataset = Sam3ImageDataset(
        img_folder=val_img_dir, ann_file=val_ann,
        transforms=[build_transforms(is_train=False)],
        load_segmentation=True, max_ann_per_img=500000, coco_json_loader=coco_loader,
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2, collate_fn=collate_fn, pin_memory=True)

    # 4. Setup Loss & Optimizer
    criterion = build_loss().to(device)
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-5, weight_decay=0.05)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")

    # 5. Train & Validate
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [Train]")

        for step, batch_dict in enumerate(pbar, start=1):
            batch = copy_data_to_device(batch_dict["all"], device, non_blocking=True)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(batch)
                loss = criterion(outputs, batch)[CORE_LOSS_KEY] / GRAD_ACCUM

            loss.backward()

            if step % GRAD_ACCUM == 0 or step == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            step_loss = loss.item() * GRAD_ACCUM
            train_loss += step_loss
            pbar.set_postfix(loss=f"{step_loss:.4f}")

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_dict in tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} [Val]"):
                batch = copy_data_to_device(batch_dict["all"], device, non_blocking=True)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    outputs = model(batch)
                    val_loss += criterion(outputs, batch)[CORE_LOSS_KEY].item()

        avg_train = train_loss / max(1, len(train_loader))
        avg_val = val_loss / max(1, len(val_loader))
        print(f"Epoch {epoch} | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_ckpt = OUTPUT_DIR / "sam3_cars_best.pt"
            torch.save({"model": model.state_dict()}, best_ckpt)
            print(f"--> Saved new best model to {best_ckpt}")


if __name__ == "__main__":
    main()