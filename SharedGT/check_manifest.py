# Checks dataset/images match manifest.csv,
# so everyone trains, tunes and tests on the same split.

import csv
import hashlib
import os
import sys
from collections import Counter

# Folder of this file and folder of the repository
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)


# Input: manifest.csv in this folder, and the image files that it lists
# Does: checks that every listed image exists and has the same md5 as in the manifest
def main():
    rows = list(csv.DictReader(open(os.path.join(HERE, "manifest.csv"))))
    missing, changed = [], []
    for row in rows:
        path = os.path.join(REPO, row["image"])
        if not os.path.exists(path):
            missing.append(row["image"])
        # Same md5 means the file has exactly the same bytes
        elif hashlib.md5(open(path, "rb").read()).hexdigest() != row["md5"]:
            changed.append(row["image"])
    counts = Counter(row["split"] for row in rows)
    print(", ".join(f"{s}: {counts[s]}" for s in ("train", "dev", "test")))
    if missing or changed:
        print(f"{len(missing)} missing, {len(changed)} different, e.g. {(missing + changed)[:3]}")
        sys.exit(1)
    print("all images match")


if __name__ == "__main__":
    main()
