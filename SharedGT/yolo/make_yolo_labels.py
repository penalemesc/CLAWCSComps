# Unzips the YOLO labels (class 0 = car) and links yolo/images to dataset/images.
# Run once before training YOLO:
#
#   python SharedGT/yolo/make_yolo_labels.py

import os
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    link = os.path.join(HERE, "images")
    if not os.path.lexists(link):
        os.symlink(os.path.join("..", "..", "dataset", "images"), link, target_is_directory=True)

    with zipfile.ZipFile(os.path.join(HERE, "labels.zip")) as z:
        z.extractall(HERE)
        names = z.namelist()
    for folder in ("train", "val"):
        print(f"labels/{folder}: {sum(n.startswith(f'labels/{folder}/') for n in names)} files")


if __name__ == "__main__":
    main()
