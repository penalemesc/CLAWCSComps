# Zero-shot SAM3 inference

import argparse
import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_utils
from tqdm import tqdm

import common

from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16}


def load_finetuned(model, path):
    ckpt = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    state = ckpt.get("model", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [k for k in missing if "sam2_convs" not in k]
    if unexpected or missing:
        raise SystemExit(f"fine-tuned weights don't fit the model: {len(unexpected)} unexpected "
                         f"{unexpected[:3]}, {len(missing)} missing {missing[:3]}")
    print(f"loaded {len(state)} fine-tuned tensors from {os.path.basename(path)} (epoch {ckpt.get('epoch')})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default=os.path.join(common.WORKSPACE_DIR, "sam3.pt"))
    parser.add_argument("--finetuned", default=None, help="fine-tuned state dict, loaded over --weights")
    parser.add_argument("--images", default=os.path.join(common.DATASET_DIR, "images", "val"))
    parser.add_argument("--out", default=os.path.join(common.OUTPUT_DIR, "pred_sam3_val.json"))
    parser.add_argument("--prompt", default="car")
    parser.add_argument("--conf", type=float, default=0.01)
    parser.add_argument("--max-det", type=int, default=100)
    parser.add_argument("--precision", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--no-masks", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--file-list", default=None, help="only these file names (e.g. gt/dev_files.txt)")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    device = common.pick_device(args.device)
    precision = args.precision

    model = build_sam3_image_model(checkpoint_path=args.weights, load_from_HF=False, eval_mode=True,
                                   device="cuda" if device.startswith("cuda") else "cpu")
    if args.finetuned:
        load_finetuned(model, args.finetuned)
    model = model.to(device)
    processor = Sam3Processor(model, device=device, confidence_threshold=args.conf)

    def autocast():
        return torch.autocast(device_type=device.split(":")[0], dtype=DTYPES[precision])

    def run(image):
        with autocast():
            state = processor.set_image(image)
            return processor.set_text_prompt(prompt=args.prompt, state=state)

    files = common.list_images(args.images, args.limit, args.file_list)
    for name in files[: args.warmup]:
        run(Image.open(os.path.join(args.images, name)).convert("RGB"))

    images, detections = [], []
    common.reset_peak_memory(device)
    peak_mb = 0.0
    for name in tqdm(files, desc="sam3"):
        image = Image.open(os.path.join(args.images, name)).convert("RGB")
        width, height = image.size
        with common.Timer(device) as timer:
            out = run(image)
        peak_mb = max(peak_mb, common.memory_mb(device))
        images.append({"file_name": name, "width": width, "height": height, "latency_ms": timer.ms})

        scores = out["scores"].float().cpu()
        order = torch.argsort(scores, descending=True)[: args.max_det]
        boxes = out["boxes"].float().cpu()[order].tolist()
        masks = None if args.no_masks else out["masks"][order].squeeze(1).cpu().numpy()
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            det = {"file_name": name, "label": args.prompt, "score": float(scores[order[i]]),
                   "bbox": [x1, y1, x2 - x1, y2 - y1]}
            if masks is not None:
                rle = mask_utils.encode(np.asfortranarray(masks[i].astype(np.uint8)))
                rle["counts"] = rle["counts"].decode("ascii")
                det["segmentation"] = rle
            detections.append(det)

    common.write_json(args.out, {
        "model": os.path.basename(args.finetuned or args.weights),
        "config": {"finetuned": bool(args.finetuned), "prompt": args.prompt, "resolution": processor.resolution, "conf": args.conf,
                   "max_det": args.max_det, "precision": precision, "warmup": args.warmup},
        "env": common.environment_info(device),
        "peak_memory_mb": peak_mb,
        "images": images,
        "detections": detections,
    })
    print(f"wrote {len(detections)} detections for {len(images)} images -> {args.out}")


if __name__ == "__main__":
    main()
