# Checks that your dataset/images match manifest.csv (same files, same bytes),
# so everyone trains, tunes and tests on the same split.
#
#   python SharedGT/check_manifest.py

import csv
import hashlib
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)


def main():
    rows = list(csv.DictReader(open(os.path.join(HERE, "manifest.csv"))))
    missing, changed = [], []
    for row in rows:
        path = os.path.join(REPO, row["image"])
        if not os.path.exists(path):
            missing.append(row["image"])
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
