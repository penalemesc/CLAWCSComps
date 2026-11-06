# Do two models succeed and fail on the same cars and the same images?
# Uses each model's dev-chosen threshold from a metrics file written by evaluate.py,
# and COCO matching at IoU 0.5

import argparse
import contextlib
import io
import json
import os

import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from scipy import stats

import common

SIZES = {"small": (0, 32 ** 2), "medium": (32 ** 2, 96 ** 2), "large": (96 ** 2, np.inf)}


def quiet(fn, *args):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args)


def per_image_and_car(coco_gt, pred, threshold):
    """Per image: TP/FP/FN at the threshold; per GT car: found or not."""
    by_name = {img["file_name"]: img["id"] for img in coco_gt.dataset["images"]}
    results = [{"image_id": by_name[d["file_name"]], "category_id": 1, "bbox": d["bbox"], "score": d["score"]}
               for d in pred["detections"] if d["label"] == "car" and d["file_name"] in by_name]
    ev = COCOeval(coco_gt, quiet(coco_gt.loadRes, results), "bbox")
    quiet(ev.evaluate)
    n_img = len(ev.params.imgIds)
    images, found = {}, {}
    for i, img_id in enumerate(ev.params.imgIds):
        e = ev.evalImgs[i]
        gt_ids = [a["id"] for a in coco_gt.imgToAnns[img_id]]
        if e is None:
            images[img_id] = (0, 0, len(gt_ids))
            found.update({g: False for g in gt_ids})
            continue
        score = dict(zip(e["dtIds"], e["dtScores"]))
        keep = np.array(e["dtScores"]) >= threshold
        matched = np.array(e["dtMatches"][0]) > 0
        tp = int(np.count_nonzero(keep & matched))
        fp = int(np.count_nonzero(keep & ~matched))
        for g, d in zip(e["gtIds"], e["gtMatches"][0]):
            found[g] = bool(d > 0 and score[int(d)] >= threshold)
        images[img_id] = (tp, fp, len(gt_ids) - tp)
    assert len(images) == n_img
    return images, found


def f1(tp, fp, fn):
    return 1.0 if tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", default=os.path.join(common.OUTPUT_DIR, "metrics_val_ft.json"))
    parser.add_argument("--a", default="yolo26m_ft")
    parser.add_argument("--b", default="sam3")
    parser.add_argument("--ref", default="yolo26m", help="third model, for a reference correlation")
    parser.add_argument("--hard", type=float, default=0.10, help="share of images counted as hardest")
    parser.add_argument("--out", default=os.path.join(common.OUTPUT_DIR, "agreement_val.json"))
    args = parser.parse_args()

    metrics = json.load(open(args.metrics))
    coco_gt = quiet(COCO, metrics["gt"])
    img_ids = sorted(coco_gt.getImgIds())
    names = [args.a, args.b, args.ref]
    per_img, per_car = {}, {}
    for n in names:
        entry = metrics["models"][n]
        per_img[n], per_car[n] = per_image_and_car(coco_gt, json.load(open(entry["pred"])), entry["threshold"])

    report = {"metrics": metrics["gt"], "models": {n: metrics["models"][n]["threshold"] for n in names}}

    # 1. Car level: who finds which car
    cars = coco_gt.dataset["annotations"]
    fa = np.array([per_car[args.a][c["id"]] for c in cars])
    fb = np.array([per_car[args.b][c["id"]] for c in cars])
    area = np.array([c["area"] for c in cars])
    def table(mask):
        a, b = fa[mask], fb[mask]
        both, only_a, only_b, neither = (int(np.sum(a & b)), int(np.sum(a & ~b)), int(np.sum(~a & b)), int(np.sum(~a & ~b)))
        n = len(a)
        po = (both + neither) / n
        pe = (a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean()))
        return {"cars": n, "both": both, f"only_{args.a}": only_a, f"only_{args.b}": only_b, "neither": neither,
                f"recall_{args.a}": a.mean(), f"recall_{args.b}": b.mean(), "recall_either": (a | b).mean(),
                "cohen_kappa": (po - pe) / (1 - pe) if pe < 1 else 1.0,
                "mcnemar_p": stats.binomtest(only_a, only_a + only_b).pvalue if only_a + only_b else 1.0}
    report["cars"] = {"all": table(np.ones(len(cars), bool)),
                      **{s: table((area >= lo) & (area < hi)) for s, (lo, hi) in SIZES.items()}}

    # 2. Image level: per-image F1 and the hardest images
    f = {n: np.array([f1(*per_img[n][i]) for i in img_ids]) for n in names}
    with_cars = np.array([per_img[args.a][i][0] + per_img[args.a][i][2] > 0 for i in img_ids])
    def spearman(x, y):
        return float(stats.spearmanr(x[with_cars], y[with_cars]).statistic)
    k = int(round(args.hard * with_cars.sum()))
    idx = np.where(with_cars)[0]
    hardest = {n: set(idx[np.argsort(f[n][idx], kind="stable")[:k]]) for n in names}
    overlap = len(hardest[args.a] & hardest[args.b])
    report["images"] = {
        "images_with_cars": int(with_cars.sum()),
        f"spearman_f1_{args.a}_vs_{args.b}": spearman(f[args.a], f[args.b]),
        f"spearman_f1_{args.ref}_vs_{args.a}": spearman(f[args.ref], f[args.a]),
        f"spearman_f1_{args.ref}_vs_{args.b}": spearman(f[args.ref], f[args.b]),
        "hardest_k": k, "hardest_overlap": overlap, "hardest_overlap_by_chance": k * k / with_cars.sum(),
        "hardest_overlap_p": float(stats.hypergeom.sf(overlap - 1, int(with_cars.sum()), k, k)),
        f"mean_f1_{args.a}": float(f[args.a][with_cars].mean()), f"mean_f1_{args.b}": float(f[args.b][with_cars].mean()),
    }

    # 3. Where they disagree: F1 gap vs simple image properties
    file_of = {i: coco_gt.imgs[i]["file_name"] for i in img_ids}
    n_cars = np.array([len(coco_gt.imgToAnns[i]) for i in img_ids])
    small_share = np.array([np.mean([a["area"] < 32 ** 2 for a in coco_gt.imgToAnns[i]]) if coco_gt.imgToAnns[i] else np.nan
                            for i in img_ids])
    brightness = np.array([np.asarray(Image.open(os.path.join(common.DATASET_DIR, "images", "val", file_of[i]))
                                      .convert("L").reduce(8)).mean() for i in img_ids])
    gap = f[args.b] - f[args.a]
    def rho(x):
        m = with_cars & ~np.isnan(x)
        r = stats.spearmanr(gap[m], x[m])
        return {"spearman": float(r.statistic), "p": float(r.pvalue)}
    dark = with_cars & (brightness < np.percentile(brightness[with_cars], 25))
    light = with_cars & (brightness >= np.percentile(brightness[with_cars], 25))
    report["gap"] = {
        "definition": f"per-image F1 of {args.b} minus {args.a}",
        "vs_number_of_cars": rho(n_cars.astype(float)),
        "vs_share_of_small_cars": rho(small_share),
        "vs_brightness": rho(brightness),
        "mean_gap_darkest_quarter": float(gap[dark].mean()), "mean_gap_other_images": float(gap[light].mean()),
        "images_better_by_0.2": {args.a: int(np.sum(with_cars & (gap <= -0.2))), args.b: int(np.sum(with_cars & (gap >= 0.2)))},
        "largest_gaps": {
            f"{args.a}_better": [(file_of[img_ids[i]], round(float(gap[i]), 3), int(n_cars[i])) for i in np.argsort(gap)[:8]],
            f"{args.b}_better": [(file_of[img_ids[i]], round(float(gap[i]), 3), int(n_cars[i])) for i in np.argsort(-gap)[:8]],
        },
    }
    common.write_json(args.out, report)
    print(json.dumps(report, indent=1, default=float))


if __name__ == "__main__":
    main()
