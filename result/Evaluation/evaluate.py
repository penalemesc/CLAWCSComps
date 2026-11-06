# COCO box/mask AP, P/R/F1 at IoU 0.5 (threshold = max F1 on dev), pixel IoU/Dice, latency and memory, and paired-bootstrap 95% CIs.

import argparse
import contextlib
import io
import json
import os
from collections import defaultdict

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import common

REC_THRS = np.linspace(0.0, 1.0, 101)
# COCOeval area-range index
AREA_ALL, AREA_SMALL = 0, 1
MAX_DET = 100
CAR_TRAIN_ID = 13


def parse_named(values):
    out = {}
    for item in values or []:
        name, _, path = item.partition("=")
        out[name] = path
    return out


def quiet(fn, *args):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args)


def load_gt(path):
    coco = quiet(COCO, path)
    by_name = {img["file_name"]: img["id"] for img in coco.dataset["images"]}
    return coco, by_name


def to_results(pred, by_name, category_id, labels, iou_type):
    predicted = {img["file_name"] for img in pred["images"]}
    missing = set(by_name) - predicted
    if missing:
        raise SystemExit(f"{len(missing)} GT images have no inference record, e.g. {sorted(missing)[:3]}")
    results = []
    for det in pred["detections"]:
        if det["label"] not in labels or det["file_name"] not in by_name:
            continue
        res = {"image_id": by_name[det["file_name"]], "category_id": category_id,
               "bbox": det["bbox"], "score": det["score"]}
        if iou_type == "segm":
            res["segmentation"] = det["segmentation"]
        results.append(res)
    return results


def run_cocoeval(coco_gt, results, iou_type):
    """pycocotools stats plus per-image matches for the bootstrap."""
    coco_dt = quiet(coco_gt.loadRes, results) if results else COCO()
    ev = COCOeval(coco_gt, coco_dt, iou_type)
    quiet(ev.evaluate)
    quiet(ev.accumulate)
    quiet(ev.summarize)
    names = ["AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large", "AR1", "AR10", "AR100",
             "AR_small", "AR_medium", "AR_large"]
    stats = dict(zip(names, map(float, ev.stats)))

    n_img, n_iou = len(ev.params.imgIds), len(ev.params.iouThrs)
    per_area = {}
    for area in (AREA_ALL, AREA_SMALL):
        records = []
        for i in range(n_img):
            e = ev.evalImgs[area * n_img + i]
            if e is None:
                records.append((np.zeros(0), np.zeros((n_iou, 0), bool), np.zeros((n_iou, 0), bool), 0))
                continue
            records.append((np.asarray(e["dtScores"][:MAX_DET]),
                            np.asarray(e["dtMatches"][:, :MAX_DET]) > 0,
                            np.asarray(e["dtIgnore"][:, :MAX_DET]).astype(bool),
                            int(np.count_nonzero(np.asarray(e["gtIgnore"]) == 0))))
        per_area[area] = records
    return stats, per_area


def ap_per_iou(records, idx):
    """COCOeval.accumulate over a resampled set of images."""
    sel = [records[i] for i in idx]
    n_gt = sum(r[3] for r in sel)
    n_iou = sel[0][1].shape[0]
    if n_gt == 0:
        return np.full(n_iou, np.nan)
    scores = np.concatenate([r[0] for r in sel])
    order = np.argsort(-scores, kind="mergesort")
    matched = np.concatenate([r[1] for r in sel], axis=1)[:, order]
    ignored = np.concatenate([r[2] for r in sel], axis=1)[:, order]
    tps = np.cumsum(matched & ~ignored, axis=1, dtype=float)
    fps = np.cumsum(~matched & ~ignored, axis=1, dtype=float)
    ap = np.zeros(n_iou)
    for t in range(n_iou):
        if tps.shape[1] == 0:
            continue
        recall = tps[t] / n_gt
        precision = tps[t] / (tps[t] + fps[t] + np.spacing(1))
        precision = np.maximum.accumulate(precision[::-1])[::-1]
        inds = np.searchsorted(recall, REC_THRS, side="left")
        valid = inds < len(precision)
        q = np.zeros(len(REC_THRS))
        q[valid] = precision[inds[valid]]
        ap[t] = q.mean()
    return ap


def summary_from_records(per_area, idx):
    ap_all = ap_per_iou(per_area[AREA_ALL], idx)
    return {"AP": ap_all.mean(), "AP50": ap_all[0], "AP75": ap_all[5],
            "AP_small": ap_per_iou(per_area[AREA_SMALL], idx).mean()}


def prf_at(records, idx, threshold):
    """P/R/F1 at IoU 0.5 for scores >= threshold."""
    tp = fp = n_gt = 0
    for i in idx:
        scores, matched, ignored, gt = records[i]
        keep = (scores >= threshold) & ~ignored[0]
        tp += int(np.count_nonzero(keep & matched[0]))
        fp += int(np.count_nonzero(keep & ~matched[0]))
        n_gt += gt
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / n_gt if n_gt else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "F1": f1, "TP": tp, "FP": fp, "FN": n_gt - tp}


def best_f1_threshold(records):
    # Max-F1 score cut at IoU 0.5
    scores = np.concatenate([r[0] for r in records])
    matched = np.concatenate([r[1][0] & ~r[2][0] for r in records])
    counted = np.concatenate([~r[2][0] for r in records])
    n_gt = sum(r[3] for r in records)
    order = np.argsort(-scores, kind="mergesort")
    scores, matched, counted = scores[order], matched[order], counted[order]
    tp = np.cumsum(matched)
    kept = np.cumsum(counted)
    f1 = 2 * tp / np.maximum(kept + n_gt, 1)
    last_of_run = np.r_[scores[1:] != scores[:-1], True]
    best = np.argmax(np.where(last_of_run, f1, -1))
    return float(scores[best])


def gt_masks_from_coco(coco_gt):
    union = {}
    for img_id, img in coco_gt.imgs.items():
        rles = [coco_gt.annToRLE(a) for a in coco_gt.imgToAnns[img_id]]
        rles = [{"size": r["size"], "counts": r["counts"].encode("ascii") if isinstance(r["counts"], str) else r["counts"]}
                for r in rles]
        union[img["file_name"]] = (mask_utils.merge(rles) if rles else None, (img["height"], img["width"]))

    def get(name):
        rle, shape = union[name]
        return np.zeros(shape, bool) if rle is None else mask_utils.decode(rle).astype(bool)
    return get


def gt_masks_from_semantic(sem_dir):
    return lambda name: np.array(Image.open(os.path.join(
        sem_dir, os.path.splitext(name)[0] + "_train_id.png"))) == CAR_TRAIN_ID


def pixel_counts(pred, file_names, gt_mask, labels, threshold, fill):
    """Per image (intersection, predicted px, GT px). fill: "mask" or "box"."""
    kept = defaultdict(list)
    for det in pred["detections"]:
        if det["label"] in labels and det["score"] >= threshold:
            kept[det["file_name"]].append(det)
    counts = []
    for name in file_names:
        gt = gt_mask(name)
        union = np.zeros_like(gt)
        for det in kept[name]:
            if fill == "mask":
                rle = det["segmentation"]
                union |= mask_utils.decode({"size": rle["size"], "counts": rle["counts"].encode("ascii")}).astype(bool)
            else:
                x, y, w, h = det["bbox"]
                union[max(0, round(y)):max(0, round(y + h)), max(0, round(x)):max(0, round(x + w))] = True
        counts.append((np.count_nonzero(union & gt), np.count_nonzero(union), np.count_nonzero(gt)))
    return np.array(counts, dtype=np.int64)


def pixel_summary(counts, idx):
    inter, pred, gt = counts[idx].T
    union = pred + gt - inter
    has = union > 0
    return {"IoU": inter.sum() / max(union.sum(), 1), "Dice": 2 * inter.sum() / max((pred + gt).sum(), 1),
            "mean_image_IoU": float(np.mean(inter[has] / union[has])) if has.any() else float("nan"),
            "mean_image_Dice": float(np.mean(2 * inter[has] / (pred + gt)[has])) if has.any() else float("nan")}


def latency_summary(pred, file_names):
    keep = set(file_names)
    ms = np.array([img["latency_ms"] for img in pred["images"] if img["file_name"] in keep])
    return {"latency_mean_ms": ms.mean(), "latency_std_ms": ms.std(ddof=1), "latency_median_ms": float(np.median(ms)),
            "latency_p95_ms": float(np.percentile(ms, 95)), "fps": 1000.0 / ms.mean(),
            "peak_memory_mb": pred.get("peak_memory_mb"), "device": pred["env"].get("gpu", pred["env"]["device"])}


def evaluate_split(gt_path, preds, args, iou_type):
    coco_gt, by_name = load_gt(gt_path)
    category_id = coco_gt.getCatIds(catNms=[args.category])[0]
    out = {}
    for name, pred in preds.items():
        if iou_type == "segm" and not any("segmentation" in d for d in pred["detections"][:1]):
            continue
        labels = set(args.labels.get(name, args.category).split("+"))
        results = to_results(pred, by_name, category_id, labels, iou_type)
        out[name] = run_cocoeval(coco_gt, results, iou_type)
    return out, len(by_name)


def check_fast_path(name, stats, per_area, n):
    fast = summary_from_records(per_area, range(n))
    for key in ("AP", "AP50", "AP75", "AP_small"):
        if abs(fast[key] - stats[key]) > 1e-9 and not (np.isnan(fast[key]) and stats[key] < 0):
            raise SystemExit(f"bootstrap AP re-implementation disagrees with pycocotools for {name} {key}: "
                             f"{fast[key]} vs {stats[key]}")


def bootstrap(evals, pixels, n_images, thresholds, rounds, seed):
    rng = np.random.default_rng(seed)
    names = list(evals)
    draws = {n: defaultdict(list) for n in names}
    for _ in range(rounds):
        idx = rng.integers(0, n_images, n_images)
        for n in names:
            _, per_area = evals[n]
            for k, v in summary_from_records(per_area, idx).items():
                draws[n][k].append(v)
            if n in thresholds:
                draws[n]["F1"].append(prf_at(per_area[AREA_ALL], idx, thresholds[n])["F1"])
            for fill, counts in pixels.get(n, {}).items():
                for k, v in pixel_summary(counts, idx).items():
                    draws[n][f"pixel_{fill}_{k}"].append(v)
    ci = {"models": {n: {k: list(np.nanpercentile(v, [2.5, 97.5])) for k, v in draws[n].items()} for n in names},
          "differences": {}}
    for i, base in enumerate(names):
        for other in names[i + 1:]:
            shared = [k for k in draws[base] if k in draws[other]]
            ci["differences"][f"{other} - {base}"] = {
                k: {"ci95": list(np.nanpercentile(np.array(draws[other][k]) - np.array(draws[base][k]), [2.5, 97.5])),
                    "share_le_0": float(np.mean(np.array(draws[other][k]) - np.array(draws[base][k]) <= 0))}
                for k in shared}
    return ci


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", required=True)
    parser.add_argument("--pred", action="append", required=True, help="name=path; CIs are given for every pair")
    parser.add_argument("--dev-gt")
    parser.add_argument("--dev-pred", action="append", help="name=path on the dev split, for threshold selection")
    parser.add_argument("--threshold", action="append", help="name=value, overrides dev selection")
    parser.add_argument("--category", default="car")
    parser.add_argument("--labels", action="append",
                        help="name=car+truck: prediction labels that count as the category (default: the category)")
    parser.add_argument("--sem-masks", default=os.path.join(common.DATASET_DIR, "bw_masks", "val"),
                        help="pixel GT when the GT file has no masks; empty string to skip")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=os.path.join(common.OUTPUT_DIR, "metrics_val.json"))
    args = parser.parse_args()
    args.labels = parse_named(args.labels)

    pred_paths = parse_named(args.pred)
    preds = {n: json.load(open(p)) for n, p in pred_paths.items()}
    box, n_images = evaluate_split(args.gt, preds, args, "bbox")
    gt_has_masks = any("segmentation" in a for a in json.load(open(args.gt))["annotations"])
    mask, _ = evaluate_split(args.gt, preds, args, "segm") if gt_has_masks else ({}, 0)
    for name, (stats, per_area) in box.items():
        check_fast_path(name, stats, per_area, n_images)

    thresholds = {n: float(v) for n, v in parse_named(args.threshold).items()}
    if args.dev_gt:
        dev_preds = {n: json.load(open(p)) for n, p in parse_named(args.dev_pred).items()}
        dev_box, _ = evaluate_split(args.dev_gt, dev_preds, args, "bbox")
        for n, (_, per_area) in dev_box.items():
            thresholds.setdefault(n, best_f1_threshold(per_area[AREA_ALL]))

    report = {"gt": os.path.abspath(args.gt), "n_images": n_images, "category": args.category,
              "threshold_source": {n: ("given" if n in parse_named(args.threshold) else "dev max-F1")
                                   for n in thresholds}, "models": {}}
    coco_gt, _ = load_gt(args.gt)
    file_names = [coco_gt.imgs[i]["file_name"] for i in sorted(coco_gt.getImgIds())]
    if gt_has_masks:
        gt_mask, report["pixel_gt"] = gt_masks_from_coco(coco_gt), "GT instance masks"
    elif args.sem_masks:
        gt_mask, report["pixel_gt"] = gt_masks_from_semantic(args.sem_masks), args.sem_masks
    else:
        gt_mask = None
    pixels = defaultdict(dict)
    for name, (stats, per_area) in box.items():
        entry = {"pred": os.path.abspath(pred_paths[name]), "config": preds[name]["config"],
                 "labels": args.labels.get(name, args.category), "box": stats,
                 **latency_summary(preds[name], file_names)}
        if name in thresholds:
            entry["threshold"] = thresholds[name]
            entry["at_threshold_iou50"] = prf_at(per_area[AREA_ALL], range(n_images), thresholds[name])
            if gt_mask is not None:
                labels = set(entry["labels"].split("+"))
                fills = ["box"] + (["mask"] if "segmentation" in preds[name]["detections"][0] else [])
                for fill in fills:
                    pixels[name][fill] = pixel_counts(preds[name], file_names, gt_mask, labels,
                                                      thresholds[name], fill)
                    entry[f"pixel_{fill}"] = pixel_summary(pixels[name][fill], np.arange(n_images))
        if name in mask:
            entry["mask"] = mask[name][0]
        report["models"][name] = entry
    if args.bootstrap:
        report["bootstrap"] = {"rounds": args.bootstrap, "seed": args.seed,
                               "ci95": bootstrap(box, pixels, n_images, thresholds, args.bootstrap, args.seed)}
    common.write_json(args.out, report)
    print_table(report)
    print(f"\nwrote {args.out}")


def print_table(report):
    ci_all = report.get("bootstrap", {}).get("ci95", {})
    ci = ci_all.get("models", {})
    print(f"{report['category']} on {report['n_images']} images")
    header = f"{'model':<18}{'AP':>22}{'AP50':>22}{'AP75':>8}{'APs':>22}{'APm':>8}{'APl':>8}{'AR100':>8}"
    print(header)
    for name, m in report["models"].items():
        b, c = m["box"], ci.get(name, {})

        def cell(key):
            if key in c:
                return f"{b[key]:.3f} [{c[key][0]:.3f},{c[key][1]:.3f}]"
            return f"{b[key]:.3f}"
        print(f"{name:<18}{cell('AP'):>22}{cell('AP50'):>22}{b['AP75']:>8.3f}{cell('AP_small'):>22}"
              f"{b['AP_medium']:>8.3f}{b['AP_large']:>8.3f}{b['AR100']:>8.3f}")
    for name, m in report["models"].items():
        line = f"{name:<18}latency {m['latency_mean_ms']:.1f}±{m['latency_std_ms']:.1f} ms ({m['fps']:.1f} FPS) on {m['device']}"
        if "at_threshold_iou50" in m:
            p = m["at_threshold_iou50"]
            line += f" | thr {m['threshold']:.3f}: P {p['precision']:.3f} R {p['recall']:.3f} F1 {p['F1']:.3f}"
        if "mask" in m:
            line += f" | mask AP {m['mask']['AP']:.3f}"
        for fill in ("box", "mask"):
            if f"pixel_{fill}" in m:
                px, c = m[f"pixel_{fill}"], ci.get(name, {}).get(f"pixel_{fill}_IoU")
                line += f" | pixel IoU ({fill}) {px['IoU']:.3f}" + (f" [{c[0]:.3f},{c[1]:.3f}]" if c else "")
                line += f" Dice {px['Dice']:.3f}"
        print(line)
    for pair, d in ci_all.get("differences", {}).items():
        keys = [k for k in ("AP", "AP50", "AP_small", "F1", "pixel_box_IoU") if k in d]
        print(f"{pair}: " + ", ".join(f"d{k} [{d[k]['ci95'][0]:+.3f},{d[k]['ci95'][1]:+.3f}]" for k in keys))


if __name__ == "__main__":
    main()
