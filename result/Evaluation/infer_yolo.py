# Zero-shot YOLO26 inference

import argparse
import os

import cv2
from tqdm import tqdm
from ultralytics import YOLO

import common

# COCO classes decides which ones count as car
KEEP_LABELS = {"car", "person", "truck", "bus"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default=os.path.join(common.WORKSPACE_DIR, "yolo26m.pt"))
    parser.add_argument("--images", default=os.path.join(common.DATASET_DIR, "images", "val"))
    parser.add_argument("--out", default=os.path.join(common.OUTPUT_DIR, "pred_yolo26m_val.json"))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU (ignored with --nms-free)")
    parser.add_argument("--nms-free", action="store_true",
                        help="use YOLO26's one-to-one end-to-end head instead of one-to-many + NMS")
    parser.add_argument("--half", action="store_true", help="fp16 inference (CUDA only)")
    parser.add_argument("--device", default=None)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--file-list", default=None, help="only these file names (e.g. gt/dev_files.txt)")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    device = common.pick_device(args.device)
    model = YOLO(args.weights)
    names = model.names
    # default one-to-many head + NMS
    # nms=False selects the NMS-free head
    predict_args = dict(imgsz=args.imgsz, conf=args.conf, max_det=args.max_det, iou=args.iou,
                        device=device, verbose=False)
    if args.nms_free:
        predict_args["nms"] = False
    if args.half:
        predict_args["half"] = True

    files = common.list_images(args.images, args.limit, args.file_list)
    for name in files[: args.warmup]:
        model(cv2.imread(os.path.join(args.images, name)), **predict_args)

    images, detections = [], []
    common.reset_peak_memory(device)
    peak_mb = 0.0
    for name in tqdm(files, desc="yolo"):
        # timer
        frame = cv2.imread(os.path.join(args.images, name))
        height, width = frame.shape[:2]
        with common.Timer(device) as timer:
            result = model(frame, **predict_args)[0]
        peak_mb = max(peak_mb, common.memory_mb(device))
        images.append({"file_name": name, "width": width, "height": height, "latency_ms": timer.ms})

        boxes = result.boxes
        for (x1, y1, x2, y2), score, cls in zip(boxes.xyxy.tolist(), boxes.conf.tolist(), boxes.cls.tolist()):
            label = names[int(cls)]
            if label in KEEP_LABELS:
                detections.append({"file_name": name, "label": label, "score": score,
                                   "bbox": [x1, y1, x2 - x1, y2 - y1]})

    common.write_json(args.out, {
        "model": os.path.basename(args.weights),
        "config": {"imgsz": args.imgsz, "conf": args.conf, "max_det": args.max_det,
                   "precision": "fp16" if args.half else "fp32", "warmup": args.warmup,
                   "head": "one2one (NMS-free)" if args.nms_free else f"one2many + NMS (iou={args.iou})",
                   "class_names": "COCO" if len(names) == 80 else list(names.values())},
        "env": common.environment_info(device),
        "peak_memory_mb": peak_mb,
        "images": images,
        "detections": detections,
    })
    print(f"wrote {len(detections)} detections for {len(images)} images -> {args.out}")


if __name__ == "__main__":
    main()
