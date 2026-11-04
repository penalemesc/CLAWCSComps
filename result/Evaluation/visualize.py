# Side-by-side comparison

import argparse
import json
import os
import random

import numpy as np
from PIL import Image, ImageDraw
from pycocotools import mask as mask_utils

import common

CAR_TRAIN_ID = 13


def draw_panel(image, title, boxes=(), masks=(), tint=None, color=(255, 64, 64)):
    panel = image.copy()
    if tint is not None:
        overlay = np.array(panel, dtype=np.float32)
        overlay[tint] = 0.5 * overlay[tint] + 0.5 * np.array([64, 160, 255])
        panel = Image.fromarray(overlay.astype(np.uint8))
    if len(masks):
        overlay = np.array(panel, dtype=np.float32)
        for m in masks:
            overlay[m] = 0.45 * overlay[m] + 0.55 * np.array(color)
        panel = Image.fromarray(overlay.astype(np.uint8))
    draw = ImageDraw.Draw(panel)
    for x, y, w, h in boxes:
        draw.rectangle([x, y, x + w, y + h], outline=color, width=3)
    draw.rectangle([0, 0, 12 + 7 * len(title), 22], fill=(0, 0, 0))
    draw.text((6, 5), title, fill=(255, 255, 255))
    return panel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", default=os.path.join(common.OUTPUT_DIR, "metrics_val.json"))
    parser.add_argument("--yolo", default="yolo26m")
    parser.add_argument("--sam3", default="sam3")
    parser.add_argument("--images", default=os.path.join(common.DATASET_DIR, "images", "val"))
    parser.add_argument("--sem-masks", default=os.path.join(common.DATASET_DIR, "bw_masks", "val"))
    parser.add_argument("--files", nargs="*", help="specific file names (default: random sample)")
    parser.add_argument("--n", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--out-dir", default=os.path.join(common.OUTPUT_DIR, "figures"))
    args = parser.parse_args()

    metrics = json.load(open(args.metrics))
    gt = json.load(open(metrics["gt"]))
    gt_boxes, gt_segs = {}, {}
    id_to_name = {img["id"]: img["file_name"] for img in gt["images"]}
    for ann in gt["annotations"]:
        gt_boxes.setdefault(id_to_name[ann["image_id"]], []).append(ann["bbox"])
        if "segmentation" in ann:
            gt_segs.setdefault(id_to_name[ann["image_id"]], []).append(ann["segmentation"])

    models = {}
    for key in (args.yolo, args.sam3):
        entry = metrics["models"][key]
        pred = json.load(open(entry["pred"]))
        labels = set(entry["labels"].split("+"))
        dets = {}
        for d in pred["detections"]:
            if d["label"] in labels and d["score"] >= entry["threshold"]:
                dets.setdefault(d["file_name"], []).append(d)
        models[key] = (entry["threshold"], dets)

    files = args.files or sorted(random.Random(args.seed).sample(sorted(id_to_name.values()), args.n))
    os.makedirs(args.out_dir, exist_ok=True)
    for name in files:
        image = Image.open(os.path.join(args.images, name)).convert("RGB")
        if gt_segs or not gt["annotations"]:
            sem = np.zeros((image.height, image.width), bool)
            for seg in gt_segs.get(name, []):
                sem |= mask_utils.decode({"size": seg["size"], "counts": seg["counts"].encode("ascii")}).astype(bool)
        else:
            sem = np.array(Image.open(os.path.join(args.sem_masks, os.path.splitext(name)[0] + "_train_id.png"))) == CAR_TRAIN_ID
        panels = [draw_panel(image, f"GT: {len(gt_boxes.get(name, []))} boxes + car pixels",
                             boxes=gt_boxes.get(name, []), tint=sem, color=(255, 230, 0))]
        thr, dets = models[args.yolo]
        panels.append(draw_panel(image, f"{args.yolo} >= {thr:.2f}: {len(dets.get(name, []))}",
                                 boxes=[d["bbox"] for d in dets.get(name, [])], color=(255, 64, 64)))
        thr, dets = models[args.sam3]
        masks = [mask_utils.decode({"size": d["segmentation"]["size"],
                                    "counts": d["segmentation"]["counts"].encode("ascii")}).astype(bool)
                 for d in dets.get(name, []) if "segmentation" in d]
        panels.append(draw_panel(image, f"{args.sam3} >= {thr:.2f}: {len(dets.get(name, []))}",
                                 boxes=[d["bbox"] for d in dets.get(name, [])], masks=masks, color=(64, 255, 120)))
        scale = args.width / image.width
        panels = [p.resize((args.width, round(image.height * scale))) for p in panels]
        sheet = Image.new("RGB", (args.width * len(panels), panels[0].height))
        for i, p in enumerate(panels):
            sheet.paste(p, (i * args.width, 0))
        sheet.save(os.path.join(args.out_dir, os.path.splitext(name)[0] + ".jpg"), quality=88)
    print(f"wrote {len(files)} figures to {args.out_dir}")


if __name__ == "__main__":
    main()
