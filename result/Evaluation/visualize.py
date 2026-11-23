# Draws side-by-side comparison figures
# Each figure has three panels: ground truth, YOLO26 detections and SAM 3 detections

import argparse
import json
import os
import random

import numpy as np
from PIL import Image, ImageDraw
from pycocotools import mask as mask_utils

import common

# Class id of car in the semantic mask files
CAR_TRAIN_ID = 13


# Input: image, title, boxes, masks, tint, color
# Output: a new PIL image
# Does: draws the boxes, the masks, the blue tint and a title in the top-left corner on a copy of the image
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


# Input: command-line arguments (metrics file, model names, image folder, which images, panel width, output folder)
# Does: saves one JPEG per image with the ground truth, the YOLO26 detections and the SAM 3 detections side by side
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

    # Load the ground truth and group its boxes and masks by image file name
    metrics = json.load(open(args.metrics))
    gt = json.load(open(metrics["gt"]))
    gt_boxes, gt_segs = {}, {}
    id_to_name = {img["id"]: img["file_name"] for img in gt["images"]}
    for ann in gt["annotations"]:
        gt_boxes.setdefault(id_to_name[ann["image_id"]], []).append(ann["bbox"])
        if "segmentation" in ann:
            gt_segs.setdefault(id_to_name[ann["image_id"]], []).append(ann["segmentation"])

    # For each model, keep the detections with a scored label and a score at or above the model's threshold
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

    # Use the given files, or a random sample that is the same for the same seed
    files = args.files or sorted(random.Random(args.seed).sample(sorted(id_to_name.values()), args.n))
    os.makedirs(args.out_dir, exist_ok=True)
    for name in files:
        image = Image.open(os.path.join(args.images, name)).convert("RGB")
        # Car pixels come from the ground-truth masks if there are any, otherwise from the semantic mask file
        if gt_segs or not gt["annotations"]:
            sem = np.zeros((image.height, image.width), bool)
            for seg in gt_segs.get(name, []):
                sem |= mask_utils.decode({"size": seg["size"], "counts": seg["counts"].encode("ascii")}).astype(bool)
        else:
            sem = np.array(Image.open(os.path.join(args.sem_masks, os.path.splitext(name)[0] + "_train_id.png"))) == CAR_TRAIN_ID
        # Panel 1: ground-truth boxes in yellow, car pixels in blue
        panels = [draw_panel(image, f"GT: {len(gt_boxes.get(name, []))} boxes + car pixels",
                             boxes=gt_boxes.get(name, []), tint=sem, color=(255, 230, 0))]
        # Panel 2: YOLO26 boxes in red
        thr, dets = models[args.yolo]
        panels.append(draw_panel(image, f"{args.yolo} >= {thr:.2f}: {len(dets.get(name, []))}",
                                 boxes=[d["bbox"] for d in dets.get(name, [])], color=(255, 64, 64)))
        # Panel 3: SAM 3 boxes and masks in green
        thr, dets = models[args.sam3]
        masks = [mask_utils.decode({"size": d["segmentation"]["size"],
                                    "counts": d["segmentation"]["counts"].encode("ascii")}).astype(bool)
                 for d in dets.get(name, []) if "segmentation" in d]
        panels.append(draw_panel(image, f"{args.sam3} >= {thr:.2f}: {len(dets.get(name, []))}",
                                 boxes=[d["bbox"] for d in dets.get(name, [])], masks=masks, color=(64, 255, 120)))
        # Resize the panels to the same width, put them side by side and save
        scale = args.width / image.width
        panels = [p.resize((args.width, round(image.height * scale))) for p in panels]
        sheet = Image.new("RGB", (args.width * len(panels), panels[0].height))
        for i, p in enumerate(panels):
            sheet.paste(p, (i * args.width, 0))
        sheet.save(os.path.join(args.out_dir, os.path.splitext(name)[0] + ".jpg"), quality=88)
    print(f"wrote {len(files)} figures to {args.out_dir}")


if __name__ == "__main__":
    main()
