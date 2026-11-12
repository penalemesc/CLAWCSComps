# Fine-tunes SAM 3 for car detection on SharedGT and keeps the epoch with the best dev box AP.
# Our checkpoint: python train_sam3.py --out runs/sam3_ft_v2 --workers 6 --epochs 5 --unfreeze-last 2 --skip-initial-eval

import argparse
import contextlib
import hashlib
import io
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
import types

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch
import torch.nn.functional as F
from functools import partial
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
WORKSPACE = os.path.dirname(REPO)

FROZEN = ("backbone.vision_backbone.trunk.", "backbone.language_backbone.encoder.")
TRUNK_DEPTH = 32

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--weights", default=os.path.join(WORKSPACE, "sam3.pt"))
    p.add_argument("--train-json", default=os.path.join(REPO, "SharedGT", "coco", "train_car.json"))
    p.add_argument("--dev-json", default=os.path.join(REPO, "SharedGT", "coco", "dev_car.json"))
    p.add_argument("--images", default=os.path.join(REPO, "dataset", "images", "train"))
    p.add_argument("--prompt", default="car", help="must match the category name in the json")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-5, help="encoder, decoder and heads")
    p.add_argument("--lr-vision", type=float, default=1e-5, help="neck and unfrozen trunk blocks")
    p.add_argument("--lr-text", type=float, default=2e-6, help="text projection")
    p.add_argument("--layer-decay", type=float, default=0.9)
    p.add_argument("--wd", type=float, default=0.1)
    p.add_argument("--clip", type=float, default=0.1)
    p.add_argument("--warmup", type=int, default=300, help="optimizer steps")
    p.add_argument("--min-lr-ratio", type=float, default=0.1)
    p.add_argument("--min-size", type=int, default=480, help="smallest random resize")
    p.add_argument("--no-flip", action="store_true")
    p.add_argument("--no-masks", action="store_true", help="train boxes only")
    p.add_argument("--mask-points", type=int, default=12544, help="points per mask for the mask loss, 0 = full masks")
    p.add_argument("--no-o2m", action="store_true",
                   help="train without the one-to-many (DAC) decoder branch, so training matches inference")
    p.add_argument("--unfreeze-last", type=int, default=0, help="also train the last N trunk blocks")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--limit", type=int, default=None, help="train on a fixed subset")
    p.add_argument("--dev-limit", type=int, default=None)
    p.add_argument("--check-step", type=int, default=500, help="quick dev check after this many steps")
    p.add_argument("--check-n", type=int, default=50)
    p.add_argument("--max-steps", type=int, default=None, help="stop early (smoke tests)")
    p.add_argument("--max-hours", type=float, default=None, help="skip an epoch that would not finish in time")
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--skip-initial-eval", action="store_true")
    p.add_argument("--no-save", action="store_true")
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--check-labels", type=int, default=0)
    p.add_argument("--resume", default=None)
    return p.parse_args()


def pick_device(requested):
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def install_focal_loss_fallback():
    try:
        import triton
        return False
    except ImportError:
        pass

    def focal(inputs, targets, alpha=0.25, gamma=2):
        x, t = inputs.float(), targets.float()
        p = torch.sigmoid(x)
        bce = F.binary_cross_entropy_with_logits(x, t, reduction="none")
        p_t = p * t + (1 - p) * (1 - t)
        return (alpha * t + (1 - alpha) * (1 - t)) * (1 - p_t) ** gamma * bce

    mod = types.ModuleType("sam3.train.loss.sigmoid_focal_loss")
    mod.triton_sigmoid_focal_loss = focal
    mod.triton_sigmoid_focal_loss_reduce = lambda i, t, a=0.25, g=2: focal(i, t, a, g).sum()
    sys.modules[mod.__name__] = mod
    return True


def file_md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sam3_commit():
    import sam3
    try:
        from importlib.metadata import distribution
        info = json.loads(distribution("sam3").read_text("direct_url.json") or "{}")
        if info.get("vcs_info", {}).get("commit_id"):
            return info["vcs_info"]["commit_id"]
    except Exception:
        pass
    for d in (os.path.dirname(os.path.dirname(sam3.__file__)), os.path.join(WORKSPACE, "external", "sam3")):
        try:
            return subprocess.check_output(["git", "-C", d, "rev-parse", "HEAD"], text=True,
                                           stderr=subprocess.DEVNULL).strip()
        except Exception:
            continue
    return None


def autocast(device):
    return torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16)


def peak_mem_gb(device):
    if device.startswith("cuda"):
        return torch.cuda.max_memory_allocated() / 2**30
    if device == "mps":
        return torch.mps.driver_allocated_memory() / 2**30
    return 0.0


class Logger:
    def __init__(self, out):
        self.f = open(os.path.join(out, "train_log.txt"), "a")
        self.j = open(os.path.join(out, "train_log.jsonl"), "a")

    def __call__(self, msg):
        line = f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        self.f.write(line + "\n")
        self.f.flush()

    def record(self, **row):
        self.j.write(json.dumps(row) + "\n")
        self.j.flush()


def build_model(args):
    from sam3.model_builder import build_sam3_image_model

    # eval_mode=False
    model = build_sam3_image_model(checkpoint_path=None, load_from_HF=False, eval_mode=False,
                                   device="cpu", enable_segmentation=True)
    ckpt = torch.load(args.weights, map_location="cpu", weights_only=True, mmap=True)
    ckpt = ckpt.get("model", ckpt)
    state = {k[len("detector."):]: v for k, v in ckpt.items() if k.startswith("detector.")}
    missing, unexpected = model.load_state_dict(state, strict=False)
    unexpected = [k for k in unexpected if "sam2_convs" not in k]
    if missing or unexpected:
        raise SystemExit(f"{args.weights} does not fit: missing {missing[:5]}, unexpected {unexpected[:5]}")
    return model


def trunk_block(name):
    prefix = "backbone.vision_backbone.trunk.blocks."
    return int(name[len(prefix):].split(".")[0]) if name.startswith(prefix) else None


def no_grad(fn):
    def wrapped(*a, **k):
        with torch.no_grad():
            return fn(*a, **k)
    return wrapped


def trainable_mlp(mlp, x):
    x = F.gelu(mlp.fc1(x), approximate="tanh")
    return mlp.drop2(mlp.fc2(mlp.norm(mlp.drop1(x))))


def freeze(model, unfreeze_last):
    for name, p in model.named_parameters():
        block = trunk_block(name)
        p.requires_grad = not name.startswith(FROZEN) or (
            block is not None and block >= TRUNK_DEPTH - unfreeze_last)
    for i, blk in enumerate(model.backbone.vision_backbone.trunk.blocks):
        if i < TRUNK_DEPTH - unfreeze_last:
            blk.forward = no_grad(blk.forward)
        else:
            blk.mlp.forward = partial(trainable_mlp, blk.mlp)
    enc = model.backbone.language_backbone.encoder
    enc.forward = no_grad(enc.forward)


def set_train_mode(model):
    model.train()
    model.backbone.vision_backbone.trunk.eval()
    model.backbone.language_backbone.encoder.eval()


def param_groups(model, args):
    groups = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        block = trunk_block(name)
        if name.startswith("backbone.vision_backbone."):
            lr = args.lr_vision * (args.layer_decay ** (TRUNK_DEPTH - 1 - block) if block is not None else 1.0)
            kind = f"vision{block if block is not None else ''}"
        elif name.startswith("backbone.language_backbone."):
            lr, kind = args.lr_text, "text"
        else:
            lr, kind = args.lr, "main"
        decay = p.ndim > 1 and "pos_embed" not in name
        key = (kind, decay)
        if key not in groups:
            groups[key] = {"params": [], "lr": lr, "base_lr": lr, "weight_decay": args.wd if decay else 0.0,
                           "name": f"{kind}{'' if decay else '_nodecay'}"}
        groups[key]["params"].append(p)
    return list(groups.values())


def lr_factor(step, total, warmup, min_ratio):
    if step < warmup:
        return (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

def build_loss(with_masks, mask_points):
    from sam3.train.loss.loss_fns import Boxes, IABCEMdetr, Masks
    from sam3.train.loss.sam3_loss import Sam3LossWrapper
    from sam3.train.matcher import BinaryHungarianMatcherV2, BinaryOneToManyMatcher

    # terms and weights of the official config, mask loss on sampled points
    fns = [
        Boxes(weight_dict={"loss_bbox": 5.0, "loss_giou": 2.0}),
        IABCEMdetr(weak_loss=False, weight_dict={"loss_ce": 20.0, "presence_loss": 20.0}, pos_weight=10.0,
                   alpha=0.25, gamma=2, use_presence=True, pos_focal=False, pad_n_queries=200, pad_scale_pos=1.0),
    ]
    if with_masks:
        class SampledMasks(Masks):
            # the official version fails on an image with no cars
            def _sampled_loss(self, src_masks, target_masks, num_boxes):
                if src_masks.shape[0] == 0:
                    zero = src_masks.sum() * 0.0
                    return {"loss_mask": zero, "loss_dice": zero}
                return super()._sampled_loss(src_masks, target_masks, num_boxes)

        sampling = dict(num_sample_points=mask_points, oversample_ratio=3.0, importance_sample_ratio=0.75) \
            if mask_points else {}
        fns.append(SampledMasks(focal_alpha=0.25, focal_gamma=2.0, weight_dict={"loss_mask": 200.0, "loss_dice": 10.0},
                                compute_aux=False, **sampling))
    return Sam3LossWrapper(
        loss_fns_find=fns, normalization="local",
        matcher=BinaryHungarianMatcherV2(focal=True, cost_class=2.0, cost_bbox=5.0, cost_giou=2.0,
                                         alpha=0.25, gamma=2, stable=False),
        o2m_matcher=BinaryOneToManyMatcher(alpha=0.3, threshold=0.4, topk=4), o2m_weight=2.0,
        use_o2m_matcher_on_o2m_aux=False, loss_fn_semantic_seg=None, scale_by_find_batch_size=True)


def train_transforms(args, resolution=1008):
    from sam3.train.transforms import basic_for_api as T
    from sam3.train.transforms.basic import get_random_resize_max_size, get_random_resize_scales
    from sam3.train.transforms.filter_query_transforms import (FilterCrowds, FilterEmptyTargets,
                                                               FilterFindQueriesWithTooManyOut,
                                                               FlexibleFilterFindGetQueries)
    from sam3.train.transforms.segmentation import DecodeRle

    steps = [FlexibleFilterFindGetQueries(query_filter=FilterCrowds()), DecodeRle()]
    if not args.no_flip:
        steps.append(T.RandomHorizontalFlip(consistent_transform=False))
    steps += [
        T.RandomResizeAPI(sizes=get_random_resize_scales(resolution, args.min_size, False),
                          max_size=get_random_resize_max_size(resolution), square=True,
                          consistent_transform=False),
        T.PadToSizeAPI(size=resolution, consistent_transform=False),
        T.ToTensorAPI(),
        FlexibleFilterFindGetQueries(query_filter=FilterEmptyTargets()),
        T.NormalizeAPI(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        FlexibleFilterFindGetQueries(query_filter=FilterEmptyTargets()),
    ]
    return [T.ComposeAPI(transforms=steps),
            FlexibleFilterFindGetQueries(query_filter=FilterFindQueriesWithTooManyOut(max_num_objects=200))]


def seed_worker(base, worker_id):
    random.seed(base + worker_id)
    np.random.seed(base + worker_id)


def build_train_dataset(args):
    from sam3.train.data.coco_json_loaders import COCO_FROM_JSON
    from sam3.train.data.sam3_image_dataset import Sam3ImageDataset

    with open(args.train_json) as f:
        names = {c["name"] for c in json.load(f)["categories"]}
    if names != {args.prompt}:
        raise SystemExit(f"category names {names} differ from the prompt '{args.prompt}'")
    return Sam3ImageDataset(
        img_folder=args.images, ann_file=args.train_json, transforms=train_transforms(args),
        load_segmentation=not args.no_masks, max_ann_per_img=500000, multiplier=1, training=True,
        use_caching=False, coco_json_loader=partial(COCO_FROM_JSON, include_negatives=True),
        limit_ids=args.limit)


def collate(args):
    from sam3.train.data.collator import collate_fn_api
    return partial(collate_fn_api, repeats=1, dict_key="all", with_seg_masks=not args.no_masks)


def build_train_loader(args, device, epoch, dataset):
    g = torch.Generator()
    g.manual_seed(args.seed * 1000 + epoch)
    return torch.utils.data.DataLoader(
        dataset, batch_size=args.batch, shuffle=True, generator=g, num_workers=args.workers,
        pin_memory=device.startswith("cuda"), drop_last=True, persistent_workers=False,
        worker_init_fn=partial(seed_worker, args.seed * 1000 + epoch * 100), collate_fn=collate(args))


def check_labels(args):
    dataset = build_train_dataset(args)
    out_dir = os.path.join(args.out, "label_check")
    os.makedirs(out_dir, exist_ok=True)
    picks = random.Random(args.seed).sample(range(len(dataset)), min(args.check_labels, len(dataset)))
    for i, idx in enumerate(picks):
        raw = dataset.coco._raw_data[int(dataset.ids[idx])]
        b = collate(args)([dataset[idx]])["all"]
        img = ((b.img_batch[0].float() * 0.5 + 0.5).clamp(0, 1) * 255).byte().permute(1, 2, 0).numpy()
        h, w = img.shape[:2]
        tgt = b.find_targets[0]
        n = int(tgt.num_boxes[0])
        canvas = Image.fromarray(img)
        if tgt.segments is not None and n:
            m = tgt.segments[:n].any(0).numpy()
            img = img.copy()
            img[m] = (0.6 * img[m] + 0.4 * np.array([0, 255, 0])).astype(np.uint8)
            canvas = Image.fromarray(img)
        draw = ImageDraw.Draw(canvas)
        for cx, cy, bw, bh in tgt.boxes.view(-1, 4)[:n].float().tolist():
            draw.rectangle([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h],
                           outline=(255, 0, 0), width=2)
        canvas.save(os.path.join(out_dir, f"sample_{i:02d}.jpg"), quality=90)
        print(f"sample {i}: {raw['image']['file_name']} json cars {len(raw['annotations'])}, "
              f"target boxes {n}, masks {0 if tgt.segments is None else len(tgt.segments)}, "
              f"prompt {b.find_text_batch}, image {tuple(b.img_batch.shape[-2:])}")
    print(f"wrote {out_dir}")


@torch.no_grad()
def evaluate_dev(model, args, device, limit=None):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    from sam3.model.sam3_image_processor import Sam3Processor

    model.eval()
    processor = Sam3Processor(model, device=device, confidence_threshold=0.01)
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(args.dev_json)
    images = sorted(gt.dataset["images"], key=lambda im: im["file_name"])[:limit]
    dets, t0 = [], time.time()
    for im in images:
        image = Image.open(os.path.join(args.images, im["file_name"])).convert("RGB")
        with autocast(device):
            out = processor.set_text_prompt(prompt=args.prompt, state=processor.set_image(image))
        scores = out["scores"].float().cpu()
        order = torch.argsort(scores, descending=True)[:100]
        for (x1, y1, x2, y2), s in zip(out["boxes"].float().cpu()[order].tolist(), scores[order].tolist()):
            dets.append({"image_id": im["id"], "category_id": 1, "bbox": [x1, y1, x2 - x1, y2 - y1], "score": s})
    seconds = time.time() - t0
    keys = ["AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large", "AR1", "AR10", "AR100"]
    if not dets:
        return {**{k: 0.0 for k in keys}, "n_images": len(images), "n_dets": 0, "seconds": seconds}
    with contextlib.redirect_stdout(io.StringIO()):
        ev = COCOeval(gt, gt.loadRes(dets), "bbox")
        ev.params.imgIds = [im["id"] for im in images]
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    stats = dict(zip(keys, [float(x) for x in ev.stats[:9]]))
    return {**stats, "n_images": len(images), "n_dets": len(dets), "seconds": round(seconds, 1)}


def fmt(m):
    return f"AP {m['AP']:.4f} AP50 {m['AP50']:.4f} AP75 {m['AP75']:.4f} APs {m['AP_small']:.4f} ({m['n_images']} imgs, {m['seconds']:.0f}s)"


def save_best(model, path, meta):
    tmp = path + ".tmp"
    torch.save({"model": model.state_dict(), **meta}, tmp)
    os.replace(tmp, path)


def save_last(model, optimizer, path, meta):
    trainable = {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}
    tmp = path + ".tmp"
    torch.save({"trainable": trainable, "optimizer": optimizer.state_dict(), **meta}, tmp)
    os.replace(tmp, path)


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = pick_device(args.device)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    shim = install_focal_loss_fallback()

    if args.check_labels:
        return check_labels(args)

    log = Logger(args.out)
    with open(__file__, "rb") as f:
        script = f.read()
    run_info = {
        "args": vars(args), "device": device,
        "gpu": torch.cuda.get_device_name(0) if device.startswith("cuda") else platform.processor(),
        "torch": str(torch.__version__), "python": platform.python_version(), "sam3_commit": sam3_commit(),
        "focal_loss": "torch fallback" if shim else "triton", "script_sha256": hashlib.sha256(script).hexdigest(),
        "train_json_md5": file_md5(args.train_json), "dev_json_md5": file_md5(args.dev_json),
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(args.out, "run_info.json"), "w") as f:
        json.dump(run_info, f, indent=2)
    log(f"device {device} ({run_info['gpu']}), torch {torch.__version__}, sam3 {run_info['sam3_commit']}, "
        f"focal loss {run_info['focal_loss']}")

    model = build_model(args)
    freeze(model, args.unfreeze_last)
    if args.no_o2m:
        model.transformer.decoder.dac = False
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    log(f"trainable {n_train / 1e6:.1f}M of {n_all / 1e6:.1f}M parameters")
    model = model.to(device)

    if args.eval_only:
        m = evaluate_dev(model, args, device, args.dev_limit)
        log(f"dev {fmt(m)}")
        log.record(kind="dev", epoch=0, **m)
        return

    criterion = build_loss(not args.no_masks, args.mask_points).to(device)
    optimizer = torch.optim.AdamW(param_groups(model, args), betas=(0.9, 0.999))
    for g in optimizer.param_groups:
        log(f"  group {g['name']:16s} lr {g['base_lr']:.2e} wd {g['weight_decay']} "
            f"params {sum(p.numel() for p in g['params']) / 1e6:.2f}M")

    from sam3.model.utils.misc import copy_data_to_device

    start_epoch, step, best_ap, history = 0, 0, -1.0, []
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu", weights_only=True)
        missing = model.load_state_dict(ck["trainable"], strict=False).unexpected_keys
        if missing:
            raise SystemExit(f"resume keys do not fit: {missing[:5]}")
        optimizer.load_state_dict(ck["optimizer"])
        start_epoch, step, best_ap, history = ck["epoch"], ck["step"], ck["best_ap"], ck["history"]
        log(f"resumed from {args.resume}: {start_epoch} epochs done, step {step}, best dev AP {best_ap:.4f}")
    elif not args.skip_initial_eval:
        m = evaluate_dev(model, args, device, args.dev_limit)
        log(f"dev before training {fmt(m)}")
        log.record(kind="dev", epoch=0, step=0, **m)
        history.append({"epoch": 0, **m})

    dataset = build_train_dataset(args)
    steps_per_epoch = len(build_train_loader(args, device, 0, dataset)) // args.accum
    total_steps = args.epochs * steps_per_epoch
    log(f"{steps_per_epoch} optimizer steps per epoch (batch {args.batch} x accum {args.accum}), {total_steps} total")

    t_start, epoch_seconds, done = time.time(), None, False
    for epoch in range(start_epoch, args.epochs):
        if args.max_hours and epoch_seconds and (time.time() - t_start + epoch_seconds) / 3600 > args.max_hours:
            log(f"stopping before epoch {epoch + 1}: it would pass --max-hours {args.max_hours}")
            break
        loader = build_train_loader(args, device, epoch, dataset)
        set_train_mode(model)
        optimizer.zero_grad(set_to_none=True)
        t_epoch, t_last, sums, n_sum, bad, n_imgs = time.time(), time.time(), {}, 0, 0, 0
        if device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()

        for it, batch in enumerate(loader):
            b = copy_data_to_device(batch["all"], device, non_blocking=True)
            with autocast(device):
                stages = model(b)
                targets = [model.back_convert(x) for x in b.find_targets]
                losses = criterion(stages, targets)
            loss = losses.pop("core_loss")
            if not torch.isfinite(loss):
                bad += 1
                log(f"non-finite loss at epoch {epoch + 1} iter {it}, skipped ({bad} in a row)")
                optimizer.zero_grad(set_to_none=True)
                if bad >= 20:
                    raise SystemExit("too many non-finite losses")
                continue
            bad = 0
            (loss / args.accum).backward()
            n_imgs += len(b.img_batch)
            sums["loss"] = sums.get("loss", 0.0) + loss.item()
            for k, v in losses.items():
                if "_aux_" not in k and not k.endswith("_fs") and torch.is_tensor(v):
                    sums[k] = sums.get(k, 0.0) + v.item()
            n_sum += 1

            if (it + 1) % args.accum:
                continue
            factor = lr_factor(step, total_steps, args.warmup, args.min_lr_ratio)
            for g in optimizer.param_groups:
                g["lr"] = g["base_lr"] * factor
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], args.clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1

            if step % args.log_every == 0:
                avg = {k: v / n_sum for k, v in sums.items()}
                rate = (time.time() - t_last) / (args.log_every * args.accum * args.batch)
                eta = rate * (len(loader) * args.batch - n_imgs) / 60
                log(f"epoch {epoch + 1} step {step} " + " ".join(f"{k} {v:.4f}" for k, v in avg.items())
                    + f" | lr {args.lr * factor:.2e} {rate:.2f}s/img mem {peak_mem_gb(device):.1f}GB eta {eta:.0f}min")
                log.record(kind="train", epoch=epoch + 1, step=step, lr=args.lr * factor, s_per_img=rate, **avg)
                sums, n_sum, t_last = {}, 0, time.time()

            if step == args.check_step:
                m = evaluate_dev(model, args, device, args.check_n)
                log(f"quick dev check at step {step} {fmt(m)}")
                log.record(kind="check", epoch=epoch + 1, step=step, **m)
                set_train_mode(model)
                t_last = time.time()

            if args.max_steps and step >= args.max_steps:
                done = True
                break

        epoch_seconds = time.time() - t_epoch
        m = evaluate_dev(model, args, device, args.dev_limit)
        history.append({"epoch": epoch + 1, "step": step, "train_minutes": round(epoch_seconds / 60, 1), **m})
        better = m["AP"] > best_ap
        log(f"epoch {epoch + 1} done in {epoch_seconds / 60:.1f} min, dev {fmt(m)}{'  (best)' if better else ''}")
        log.record(kind="dev", epoch=epoch + 1, step=step, train_minutes=epoch_seconds / 60, **m)
        meta = {"epoch": epoch + 1, "step": step, "dev": m, "history": history, "run_info": run_info}
        if better:
            best_ap = m["AP"]
            if not args.no_save:
                save_best(model, os.path.join(args.out, "best.pt"), {**meta, "script": script.decode()})
                log(f"saved best.pt (epoch {epoch + 1})")
        if not args.no_save:
            save_last(model, optimizer, os.path.join(args.out, "last.pt"), {**meta, "best_ap": best_ap})
        if done:
            break

    with open(os.path.join(args.out, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    log(f"finished, best dev AP {best_ap:.4f}, total {(time.time() - t_start) / 3600:.2f} h")


if __name__ == "__main__":
    main()
