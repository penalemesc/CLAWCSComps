# Builds the shared labels and splits. Same output every run.
#
#   python SharedGT/build_shared_gt.py --pan-seg <...>/bdd100k/labels/pan_seg
#
# Labels: BDD100K panoptic segmentation (bdd100k_pan_seg_labels_trainval.zip),
# one box and one mask per car. Images without a panoptic label are dropped.
# test = val, dev = 500 random train images (seed 0), train = the rest.

import argparse
import csv
import hashlib
import json
import os
import random
import zipfile

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils

HERE = os.path.dirname(os.path.abspath(__file__))
DATASET = os.path.join(os.path.dirname(HERE), "dataset")
PAN_CAR = 35  # BDD100K panoptic category id for car
DEV_SIZE, DEV_SEED = 500, 0


def list_images(split):
    return sorted(f for f in os.listdir(os.path.join(DATASET, "images", split)) if f.endswith(".jpg"))


def car_instances(bitmask_path):
    # RGBA: R = category, G = attribute bits (crowd = 2, ignore = 1), B * 256 + A = instance id
    bm = np.array(Image.open(bitmask_path))
    car = bm[..., 0] == PAN_CAR
    inst = bm[..., 2].astype(np.int32) * 256 + bm[..., 3]
    out = []
    for inst_id in np.unique(inst[car]):
        m = car & (inst == inst_id)
        attrs = int(np.bincount(bm[..., 1][m]).argmax())
        out.append((m, bool(attrs & 3)))
    return out, bm.shape[1], bm.shape[0]


def build_split(pan_seg, folder, files):
    images, annotations = [], []
    for file_name in files:
        width, height = Image.open(os.path.join(DATASET, "images", folder, file_name)).size
        instances, bw, bh = car_instances(os.path.join(pan_seg, "bitmasks", folder, file_name[:-4] + ".png"))
        assert (bw, bh) == (width, height), file_name
        image_id = len(images) + 1
        images.append({"id": image_id, "file_name": file_name, "width": width, "height": height})
        for m, crowd in instances:
            rle = mask_utils.encode(np.asfortranarray(m.astype(np.uint8)))
            x, y, w, h = (float(v) for v in mask_utils.toBbox(rle))
            annotations.append({"id": len(annotations) + 1, "image_id": image_id, "category_id": 1,
                                "bbox": [x, y, w, h], "area": int(m.sum()), "iscrowd": int(crowd),
                                "segmentation": {"size": [height, width], "counts": rle["counts"].decode("ascii")}})
    return images, annotations


def yolo_lines(images, annotations):
    # One txt per image, class 0 = car, normalized center/size; crowd regions left out
    boxes = {}
    for a in annotations:
        if not a["iscrowd"]:
            boxes.setdefault(a["image_id"], []).append(a["bbox"])
    out = {}
    for img in images:
        w, h = img["width"], img["height"]
        out[img["file_name"]] = "".join(f"0 {(x + bw / 2) / w:.6f} {(y + bh / 2) / h:.6f} {bw / w:.6f} {bh / h:.6f}\n"
                                        for x, y, bw, bh in boxes.get(img["id"], []))
    return out


def write_zip(path, files):
    # Fixed order and timestamps so the zip is byte-identical on every rebuild
    with zipfile.ZipFile(path, "w") as z:
        for name in sorted(files):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            z.writestr(info, files[name])


def write_lines(path, lines):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pan-seg", required=True, help="folder with bitmasks/ from the panoptic zip")
    args = parser.parse_args()

    labelled = {s: {f[:-4] + ".jpg" for f in os.listdir(os.path.join(args.pan_seg, "bitmasks", s)) if f.endswith(".png")}
                for s in ("train", "val")}
    usable = {s: set(list_images(s)) & labelled[s] for s in ("train", "val")}
    excluded = sorted(f"{s}/{f}" for s in ("train", "val") for f in set(list_images(s)) - labelled[s])

    # Sample from all of train so dropped images don't change the sample
    dev_sample = set(random.Random(DEV_SEED).sample(list_images("train"), DEV_SIZE))
    splits = {
        "train": ("train", sorted(usable["train"] - dev_sample)),
        "dev": ("train", sorted(usable["train"] & dev_sample)),
        "test": ("val", sorted(usable["val"])),
    }

    write_lines(os.path.join(HERE, "excluded.txt"), [f"{e}\t# no BDD panoptic label" for e in excluded])
    summary, manifest, labels = {}, [], {}
    for name, (folder, files) in splits.items():
        images, annotations = build_split(args.pan_seg, folder, files)
        for file_name, text in yolo_lines(images, annotations).items():
            labels[f"labels/{folder}/{file_name[:-4]}.txt"] = text
        cars = {img["file_name"]: 0 for img in images}
        id_to_name = {img["id"]: img["file_name"] for img in images}
        for a in annotations:
            cars[id_to_name[a["image_id"]]] += 1
        for f in files:
            path = f"dataset/images/{folder}/{f}"
            md5 = hashlib.md5(open(os.path.join(DATASET, "images", folder, f), "rb").read()).hexdigest()
            manifest.append((name, path, cars[f], md5))
        write_lines(os.path.join(HERE, "splits", f"{name}.txt"), files)
        # Ultralytics reads "./" as relative to this file; yolo/make_yolo_labels.py unzips labels/ and links images/
        write_lines(os.path.join(HERE, "yolo", f"{name}.txt"), [f"./images/{folder}/{f}" for f in files])
        os.makedirs(os.path.join(HERE, "coco"), exist_ok=True)
        with open(os.path.join(HERE, "coco", f"{name}_car.json"), "w") as f:
            json.dump({"info": {"description": f"CLAW shared GT, {name} split (images in dataset/images/{folder})",
                                "source": "BDD100K panoptic segmentation, car instances", "excluded": excluded},
                       "images": images, "annotations": annotations,
                       "categories": [{"id": 1, "name": "car"}]}, f)
        summary[name] = {"images": len(images), "cars": len(annotations),
                         "images_without_car": len(images) - len({a["image_id"] for a in annotations})}
    write_zip(os.path.join(HERE, "yolo", "labels.zip"), labels)
    with open(os.path.join(HERE, "manifest.csv"), "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["split", "image", "cars", "md5"])
        w.writerows(manifest)
    print(json.dumps({"excluded": excluded, "splits": summary}, indent=1))


if __name__ == "__main__":
    main()
