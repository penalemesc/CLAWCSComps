import os
import json
import argparse
from pathlib import Path
from functools import partial
import numpy as np
from PIL import Image
from tqdm import tqdm
from pycocotools import mask as mask_utils

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW

import sam3
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
# 1. Automatic Dataset Converter (images/{split} + bw_masks/{split} -> COCO JSON)
# ---------------------------------------------------------------------------
def build_coco_json_for_split(
    dataset_root: Path,
    split: str,
    mask_folder_name: str,
    noun_phrase: str,
    force_rebuild: bool = False,
) -> tuple[str, str]:
    """
    Scans dataset/{images,bw_masks}/{train,val} and builds the COCO JSON
    with `noun_phrase` annotations required by Sam3ImageDataset.
    """
    images_dir = dataset_root / "images" / split
    masks_dir = dataset_root / mask_folder_name / split
    output_json = dataset_root / f"_sam3_{split}_annotations.coco.json"

    if not images_dir.exists():
        raise FileNotFoundError(f"Missing images directory: {images_dir}")
    if not masks_dir.exists():
        raise FileNotFoundError(f"Missing masks directory: {masks_dir}")

    if output_json.exists() and not force_rebuild:
        print(f"[{split.upper()}] Using existing COCO JSON: {output_json}")
        return str(images_dir), str(output_json)

    print(f"[{split.upper()}] Generating SAM 3 COCO annotations from {masks_dir}...")
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": noun_phrase}],
    }

    ann_id = 1
    valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
    img_files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith(valid_exts)])

    for img_id, fname in enumerate(tqdm(img_files, desc=f"Indexing {split}"), start=1):
        img_file = images_dir / fname

        # Match mask by exact filename or stem + .png / .jpg
        mask_file = masks_dir / fname
        if not mask_file.exists():
            for ext in [".png", ".jpg", ".jpeg", ".tif", ".bmp"]:
                candidate = masks_dir / f"{Path(fname).stem}{ext}"
                if candidate.exists():
                    mask_file = candidate
                    break

        if not mask_file.exists():
            print(f"Warning: No matching mask found for {fname}, skipping.")
            continue

        with Image.open(img_file) as img:
            width, height = img.size

        # Load grayscale mask; treat any pixel > 0 as foreground
        mask_np = np.array(Image.open(mask_file).convert("L"))
        binary_mask = (mask_np > 0).astype(np.uint8)

        # Skip completely empty masks so FilterEmptyTargets doesn't drop unannotated frames
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
        bbox = mask_utils.toBbox(rle).tolist()  # [x, y, w, h]
        area = float(mask_utils.area(rle))

        coco_data["annotations"].append({
            "id": ann_id,
            "image_id": img_id,
            "category_id": 1,
            "bbox": bbox,
            "area": area,
            "segmentation": rle,
            "iscrowd": 0,
            "noun_phrase": noun_phrase,
        })
        ann_id += 1

    with open(output_json, "w") as f:
        json.dump(coco_data, f, indent=2)

    print(
        f"[{split.upper()}] Saved {len(coco_data['images'])} valid images and "
        f"{len(coco_data['annotations'])} annotations to {output_json}"
    )
    return str(images_dir), str(output_json)


# ---------------------------------------------------------------------------
# 2. Native SAM 3 Transforms, Loss & Optimizer
# ---------------------------------------------------------------------------
def build_transforms(resolution: int = 1008, is_train: bool = True):
    transforms_list = [
        FlexibleFilterFindGetQueries(query_filter=FilterCrowds()),
        DecodeRle(),
    ]
    if is_train:
        transforms_list.append(RandomizeInputBbox(box_noise_scale=0.1))

    transforms_list.extend([
        RandomResizeAPI(
            sizes=[resolution],
            max_size=resolution,
            square=True,
            consistent_transform=False,
        ),
        PadToSizeAPI(size=resolution, consistent_transform=False),
        ToTensorAPI(),
        FlexibleFilterFindGetQueries(query_filter=FilterEmptyTargets()),
        NormalizeAPI(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return ComposeAPI(transforms=transforms_list)


def build_native_loss():
    matcher = BinaryHungarianMatcherV2(
        focal=True,
        cost_class=2.0,
        cost_bbox=5.0,
        cost_giou=2.0,
        alpha=0.25,
        gamma=2.0,
    )
    o2m_matcher = BinaryOneToManyMatcher(alpha=0.3, threshold=0.4, topk=4)
    loss_fns = [
        Boxes(weight_dict={"loss_bbox": 5.0, "loss_giou": 2.0}),
        IABCEMdetr(
            weak_loss=False,
            pos_weight=10.0,
            alpha=0.25,
            gamma=2.0,
            use_presence=True,
            pos_focal=False,
            weight_dict={"loss_ce": 20.0, "presence_loss": 20.0},
        ),
        Masks(
            focal_alpha=0.25,
            focal_gamma=2.0,
            num_sample_points=4096,
            oversample_ratio=3.0,
            importance_sample_ratio=0.75,
            weight_dict={"loss_mask": 80.0, "loss_dice": 10.0},
        ),
    ]
    return Sam3LossWrapper(
        matcher=matcher,
        o2m_matcher=o2m_matcher,
        o2m_weight=2.0,
        use_o2m_matcher_on_o2m_aux=False,
        loss_fns_find=loss_fns,
        normalization="local",
    )


def build_optimizer(model: torch.nn.Module, args):
    vision_params, language_params, mask_params, other_params = [], [], [], []

    for name, param in model.named_parameters():
        if args.freeze_backbone and ("backbone" in name or "vision" in name or "language" in name):
            param.requires_grad = False
            continue
        if not param.requires_grad:
            continue

        if "vision" in name or "image_encoder" in name:
            vision_params.append(param)
        elif "language" in name or "text" in name:
            language_params.append(param)
        elif "segmentation" in name or "mask" in name or "pixel_decoder" in name:
            mask_params.append(param)
        else:
            other_params.append(param)

    param_groups = [
        {"params": vision_params, "lr": args.lr_vision_backbone},
        {"params": language_params, "lr": args.lr_language_backbone},
        {"params": mask_params, "lr": args.lr_segmentation_head},
        {"params": other_params, "lr": args.lr_transformer},
    ]
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    return AdamW([g for g in param_groups if len(g["params"]) > 0], weight_decay=args.weight_decay)


# ---------------------------------------------------------------------------
# 3. Main Training & Validation Loop
# ---------------------------------------------------------------------------
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_root = Path(args.dataset_dir)

    # 1. Prepare COCO JSON files for train and val splits automatically
    train_img_dir, train_ann_file = build_coco_json_for_split(
        dataset_root, "train", args.mask_folder, args.prompt, args.rebuild_json
    )
    val_img_dir, val_ann_file = build_coco_json_for_split(
        dataset_root, "val", args.mask_folder, args.prompt, args.rebuild_json
    )

    # 2. Build Native SAM 3 Image Model
    print("Building native SAM 3 model...")
    model = build_sam3_image_model(
        bpe_path=args.bpe_path,
        device="cpu",
        eval_mode=False,
        checkpoint_path=args.checkpoint_path,
        load_from_HF=(args.checkpoint_path is None),
        enable_segmentation=True,
    )
    model.to(device)

    # 3. Build Train & Val Datasets and DataLoaders
    coco_loader = partial(COCO_FROM_JSON, include_negatives=True, category_chunk_size=18)
    collate_fn = partial(collate_fn_api, repeats=1, dict_key="all", with_seg_masks=True)

    train_dataset = Sam3ImageDataset(
        img_folder=train_img_dir,
        ann_file=train_ann_file,
        transforms=[build_transforms(resolution=args.resolution, is_train=True)],
        load_segmentation=True,
        max_ann_per_img=500000,
        coco_json_loader=coco_loader,
    )
    val_dataset = Sam3ImageDataset(
        img_folder=val_img_dir,
        ann_file=val_ann_file,
        transforms=[build_transforms(resolution=args.resolution, is_train=False)],
        load_segmentation=True,
        max_ann_per_img=500000,
        coco_json_loader=coco_loader,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # 4. Initialize Loss & Optimizer
    criterion = build_native_loss().to(device)
    optimizer = build_optimizer(model, args)

    os.makedirs(args.output_dir, exist_ok=True)
    best_val_loss = float("inf")

    # 5. Run Epochs
    for epoch in range(1, args.epochs + 1):
        # --- TRAIN ---
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Train]")

        for step, batch_dict in enumerate(pbar, start=1):
            batch = copy_data_to_device(batch_dict["all"], device, non_blocking=True)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(batch)
                loss_dict = criterion(outputs, batch)
                loss = loss_dict[CORE_LOSS_KEY] / args.grad_accum

            loss.backward()

            if step % args.grad_accum == 0 or step == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            step_loss = loss.item() * args.grad_accum
            train_loss += step_loss
            pbar.set_postfix(loss=f"{step_loss:.4f}")

        avg_train_loss = train_loss / max(1, len(train_loader))

        # --- VALIDATE ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_dict in tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} [Val]"):
                batch = copy_data_to_device(batch_dict["all"], device, non_blocking=True)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    outputs = model(batch)
                    loss_dict = criterion(outputs, batch)
                    val_loss += loss_dict[CORE_LOSS_KEY].item()

        avg_val_loss = val_loss / max(1, len(val_loader))
        print(f"Epoch {epoch} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # Save latest checkpoint
        latest_ckpt = os.path.join(args.output_dir, "sam3_finetuned_latest.pt")
        torch.save({"model": model.state_dict()}, latest_ckpt)

        # Save best checkpoint based on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_ckpt = os.path.join(args.output_dir, "sam3_finetuned_best.pt")
            torch.save({"model": model.state_dict()}, best_ckpt)
            print(f"--> New best validation loss ({best_val_loss:.4f})! Saved to {best_ckpt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-Tune Native SAM 3 on Custom Dataset Layout")
    parser.add_argument("--dataset_dir", type=str, default="./dataset", help="Root dataset folder")
    parser.add_argument("--mask_folder", type=str, default="bw_masks", choices=["bw_masks", "masks"],
                        help="Subdirectory containing ground-truth masks")
    parser.add_argument("--prompt", type=str, required=True,
                        help="Text prompt (noun phrase) describing the target object")
    parser.add_argument("--checkpoint_path", type=str, default="sam3.pt",
                        help="Path to base sam3.pt checkpoint")
    parser.add_argument("--bpe_path", type=str, default="/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz",
                        help="Path to BPE vocab inside the container")
    parser.add_argument("--output_dir", type=str, default="./checkpoints",
                        help="Folder where fine-tuned .pt files will be saved")
    parser.add_argument("--rebuild_json", action="store_true",
                        help="Force regenerate COCO JSON files even if they already exist")
    parser.add_argument("--resolution", type=int, default=1008)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--freeze_backbone", action="store_true",
                        help="Freeze vision & text backbones to reduce VRAM usage (~10GB instead of ~20GB)")
    parser.add_argument("--lr_vision_backbone", type=float, default=5e-6)
    parser.add_argument("--lr_language_backbone", type=float, default=5e-6)
    parser.add_argument("--lr_transformer", type=float, default=2e-5)
    parser.add_argument("--lr_segmentation_head", type=float, default=6e-5)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()

    main(args)