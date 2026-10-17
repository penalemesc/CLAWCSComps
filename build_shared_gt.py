# Builds the shared labels and splits from dataset/. Same output every run.
#
#   python SharedGT/build_shared_gt.py
#
# Boxes: dataset/labels, car only.
# Drops images whose mask size doesn't match the photo.
# test = val, dev = 500 random train images (seed 0), train = the rest.

import json
import os
import random

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
DATASET = os.path.join(os.path.dirname(HERE), "dataset")
CAR_CLASS_ID = 1  # dataset/data.yaml: 0 person, 1 car
DEV_SIZE, DEV_SEED = 500, 0


def list_images(split):
    return sorted(f for f in os.listdir(os.path.join(DATASET, "images", split)) if f.endswith(".jpg"))


def mask_mismatch(split, file_name):
    image_size = Image.open(os.path.join(DATASET, "images", split, file_name)).size
    mask_size = Image.open(os.path.join(DATASET, "bw_masks", split, file_name[:-4] + "_train_id.png")).size
    return image_size != mask_size, image_size


def coco_for(split, files):
    images, annotations = [], []
    for file_name in files:
        width, height = Image.open(os.path.join(DATASET, "images", split, file_name)).size
        image_id = len(images) + 1
        images.append({"id": image_id, "file_name": file_name, "width": width, "height": height})
        for line in open(os.path.join(DATASET, "labels", split, file_name[:-4] + ".txt")):
            cls, xc, yc, w, h = line.split()
            if int(cls) != CAR_CLASS_ID:
                continue
            # Labels came from integer pixel boxes, so round
            x1 = round((float(xc) - float(w) / 2) * width)
            y1 = round((float(yc) - float(h) / 2) * height)
            bw, bh = round(float(w) * width), round(float(h) * height)
            annotations.append({"id": len(annotations) + 1, "image_id": image_id, "category_id": 1,
                                "bbox": [x1, y1, bw, bh], "area": bw * bh, "iscrowd": 0})
    return images, annotations


def write_lines(path, lines):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    excluded = {}
    usable = {}
    for split in ("train", "val"):
        files = list_images(split)
        usable[split] = set()
        for file_name in files:
            bad, size = mask_mismatch(split, file_name)
            if bad:
                excluded[f"{split}/{file_name}"] = size
            else:
                usable[split].add(file_name)

    # Sample from all of train so dropped images don't change the sample
    dev_sample = random.Random(DEV_SEED).sample(list_images("train"), DEV_SIZE)
    splits = {
        "train": ("train", sorted(usable["train"] - set(dev_sample))),
        "dev": ("train", sorted(usable["train"] & set(dev_sample))),
        "test": ("val", sorted(usable["val"])),
    }

    write_lines(os.path.join(HERE, "excluded.txt"),
                [f"{k}\t# photo is {w}x{h} but its mask is not; boxes/masks don't line up" for k, (w, h) in
                 sorted(excluded.items())])
    summary = {}
    for name, (folder, files) in splits.items():
        write_lines(os.path.join(HERE, "splits", f"{name}.txt"), files)
        # Ultralytics reads "./" as relative to this file
        write_lines(os.path.join(HERE, "yolo", f"{name}.txt"),
                    [f"./../../dataset/images/{folder}/{f}" for f in files])
        images, annotations = coco_for(folder, files)
        os.makedirs(os.path.join(HERE, "coco"), exist_ok=True)
        with open(os.path.join(HERE, "coco", f"{name}_car.json"), "w") as f:
            json.dump({"info": {"description": f"CLAW shared GT, {name} split (images in dataset/images/{folder})",
                                "rules": "repo dataset/labels car boxes; images with mask/photo size mismatch dropped",
                                "excluded": sorted(excluded)},
                       "images": images, "annotations": annotations,
                       "categories": [{"id": 1, "name": "car"}]}, f)
        summary[name] = {"images": len(images), "car_boxes": len(annotations),
                         "images_without_car": len(images) - len({a["image_id"] for a in annotations})}
    print(json.dumps({"excluded": sorted(excluded), "splits": summary}, indent=1))


if __name__ == "__main__":
    main()
