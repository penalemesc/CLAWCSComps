# Fine-tunes SAM 3 for car detection on SharedGT and keeps the epoch with the best dev box AP.
# Our checkpoint: python train_sam3.py --out runs/sam3_ft_v2 --workers 6 --epochs 5 --unfreeze-last 2 --skip-initial-eval

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

# Import the ungodly amount of constants
from config import (
    FROZEN, TRUNK_DEPTH,
    OUT_DIR, WEIGHTS, TRAIN_JSON, DEV_JSON, IMAGES, PROMPT,
    EPOCHS, BATCH, ACCUM, LR, LR_VISION, LR_TEXT, LAYER_DECAY, WD, CLIP,
    WARMUP, MIN_LR_RATIO, MIN_SIZE, MASK_POINTS, UNFREEZE_LAST, WORKERS, SEED,
    DEVICE, LIMIT, DEV_LIMIT, CHECK_STEP, CHECK_N, MAX_STEPS, MAX_HOURS, LOG_EVERY,
    CHECK_LABELS, RESUME, NO_FLIP, NO_MASKS, NO_O2M, SKIP_INITIAL_EVAL, NO_SAVE, EVAL_ONLY,
    WORKSPACE
)
from logger import Logger


def pick_device(requested):
    # Select device: requested > CUDA > MPS > CPU
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def install_focal_loss_fallback():
    # Use triton implementation of focal loss if available, otherwise create fallback
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
    # Compute MD5 hash of a file
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sam3_commit():
    # Get SAM3 Git commit hash
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
    # Return autocast context for mixed precision training
    return torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16)


def peak_mem_gb(device):
    # Get peak GPU/MPS memory usage in GB
    if device.startswith("cuda"):
        return torch.cuda.max_memory_allocated() / 2**30
    if device == "mps":
        return torch.mps.driver_allocated_memory() / 2**30
    return 0.0

def build_model():
    # Load SAM3 model and restore weights
    from sam3.model_builder import build_sam3_image_model

    model = build_sam3_image_model(checkpoint_path=None, load_from_HF=False, eval_mode=False,
                                   device="cpu", enable_segmentation=True)
    ckpt = torch.load(WEIGHTS, map_location="cpu", weights_only=True, mmap=True)
    ckpt = ckpt.get("model", ckpt)
    state = {k[len("detector."):]: v for k, v in ckpt.items() if k.startswith("detector.")}
    missing, unexpected = model.load_state_dict(state, strict=False)
    unexpected = [k for k in unexpected if "sam2_convs" not in k]
    if missing or unexpected:
        raise SystemExit(f"{WEIGHTS} does not fit: missing {missing[:5]}, unexpected {unexpected[:5]}")
    return model

def trunk_block(name):
    # Extract vision backbone block index from parameter name
    prefix = "backbone.vision_backbone.trunk.blocks."
    return int(name[len(prefix):].split(".")[0]) if name.startswith(prefix) else None

def no_grad(fn):
    # Decorator to run function with no gradient computation
    def wrapped(*a, **k):
        with torch.no_grad():
            return fn(*a, **k)
    return wrapped

def trainable_mlp(mlp, x):
    # Forward pass through MLP with gradient tracking
    x = F.gelu(mlp.fc1(x), approximate="tanh")
    return mlp.drop2(mlp.fc2(mlp.norm(mlp.drop1(x))))

def freeze(model, unfreeze_last):
    # Freeze most of backbone and language encoder, unfreeze last N blocks and decoder
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
    # Set model to train mode but keep backbone and text encoder in eval
    model.train()
    model.backbone.vision_backbone.trunk.eval()
    model.backbone.language_backbone.encoder.eval()

def param_groups(model):
    # Create optimizer parameter groups with  learning rates
    groups = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        block = trunk_block(name)
        if name.startswith("backbone.vision_backbone."):
            lr = LR_VISION * (LAYER_DECAY ** (TRUNK_DEPTH - 1 - block) if block is not None else 1.0)
            kind = f"vision{block if block is not None else ''}"
        elif name.startswith("backbone.language_backbone."):
            lr, kind = LR_TEXT, "text"
        else:
            lr, kind = LR, "main"
        decay = p.ndim > 1 and "pos_embed" not in name
        key = (kind, decay)
        if key not in groups:
            groups[key] = {"params": [], "lr": lr, "base_lr": lr, "weight_decay": WD if decay else 0.0,
                           "name": f"{kind}{'' if decay else '_nodecay'}"}
        groups[key]["params"].append(p)
    return list(groups.values())

def lr_factor(step, total, warmup, min_ratio):
    # Calculate learning rate multiplier 
    if step < warmup:
        return (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

def build_loss(with_masks, mask_points):
    # Build loss functions for detection and optionally segmentation
    from sam3.train.loss.loss_fns import Boxes, IABCEMdetr, Masks
    from sam3.train.loss.sam3_loss import Sam3LossWrapper
    from sam3.train.matcher import BinaryHungarianMatcherV2, BinaryOneToManyMatcher

    fns = [
        Boxes(weight_dict={"loss_bbox": 5.0, "loss_giou": 2.0}),
        IABCEMdetr(weak_loss=False, weight_dict={"loss_ce": 20.0, "presence_loss": 20.0}, pos_weight=10.0,
                   alpha=0.25, gamma=2, use_presence=True, pos_focal=False, pad_n_queries=200, pad_scale_pos=1.0),
    ]
    if with_masks:
        class SampledMasks(Masks):
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

def train_transforms(resolution=1008):
    # Build data augmentation pipeline for training
    from sam3.train.transforms import basic_for_api as T
    from sam3.train.transforms.basic import get_random_resize_max_size, get_random_resize_scales
    from sam3.train.transforms.filter_query_transforms import (FilterCrowds, FilterEmptyTargets,
                                                               FilterFindQueriesWithTooManyOut,
                                                               FlexibleFilterFindGetQueries)
    from sam3.train.transforms.segmentation import DecodeRle

    steps = [FlexibleFilterFindGetQueries(query_filter=FilterCrowds()), DecodeRle()]
    if not NO_FLIP:
        steps.append(T.RandomHorizontalFlip(consistent_transform=False))
    steps += [
        T.RandomResizeAPI(sizes=get_random_resize_scales(resolution, MIN_SIZE, False),
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
    # Seed RNG for data loading worker reproducibility
    random.seed(base + worker_id)
    np.random.seed(base + worker_id)

def build_train_dataset():
    # Load car detection dataset from COCO JSON
    from sam3.train.data.coco_json_loaders import COCO_FROM_JSON
    from sam3.train.data.sam3_image_dataset import Sam3ImageDataset

    with open(TRAIN_JSON) as f:
        names = {c["name"] for c in json.load(f)["categories"]}
    if names != {PROMPT}:
        raise SystemExit(f"category names {names} differ from the prompt '{PROMPT}'")
    return Sam3ImageDataset(
        img_folder=IMAGES, ann_file=TRAIN_JSON, transforms=train_transforms(),
        load_segmentation=not NO_MASKS, max_ann_per_img=500000, multiplier=1, training=True,
        use_caching=False, coco_json_loader=partial(COCO_FROM_JSON, include_negatives=True),
        limit_ids=LIMIT)

def collate():
    # Create collate function for training data batches
    from sam3.train.data.collator import collate_fn_api
    return partial(collate_fn_api, repeats=1, dict_key="all", with_seg_masks=not NO_MASKS)

def build_train_loader(device, epoch, dataset):
    # Create training data loader with per-epoch seeding
    g = torch.Generator()
    g.manual_seed(SEED * 1000 + epoch)
    return torch.utils.data.DataLoader(
        dataset, batch_size=BATCH, shuffle=True, generator=g, num_workers=WORKERS,
        pin_memory=device.startswith("cuda"), drop_last=True, persistent_workers=False,
        worker_init_fn=partial(seed_worker, SEED * 1000 + epoch * 100), collate_fn=collate())

def check_labels():
    # Visualize training dataset labels for debugging
    dataset = build_train_dataset()
    out_dir = os.path.join(OUT_DIR, "label_check")
    os.makedirs(out_dir, exist_ok=True)
    picks = random.Random(SEED).sample(range(len(dataset)), min(CHECK_LABELS, len(dataset)))
    for i, idx in enumerate(picks):
        raw = dataset.coco._raw_data[int(dataset.ids[idx])]
        b = collate()([dataset[idx]])["all"]
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
def evaluate_dev(model, device, limit=None):
    # Evaluate model on development set and return COCO metrics
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    from sam3.model.sam3_image_processor import Sam3Processor

    model.eval()
    processor = Sam3Processor(model, device=device, confidence_threshold=0.01)
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(DEV_JSON)
    images = sorted(gt.dataset["images"], key=lambda im: im["file_name"])[:limit]
    dets, t0 = [], time.time()
    for im in images:
        image = Image.open(os.path.join(IMAGES, im["file_name"])).convert("RGB")
        with autocast(device):
            out = processor.set_text_prompt(prompt=PROMPT, state=processor.set_image(image))
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
    # Format metrics for logging
    return f"AP {m['AP']:.4f} AP50 {m['AP50']:.4f} AP75 {m['AP75']:.4f} APs {m['AP_small']:.4f} ({m['n_images']} imgs, {m['seconds']:.0f}s)"

def save_best(model, path, meta):
    # Save best checkpoint with model state and metadata
    tmp = path + ".tmp"
    torch.save({"model": model.state_dict(), **meta}, tmp)
    os.replace(tmp, path)

def save_last(model, optimizer, path, meta):
    # Save last checkpoint with trainable params and optimizer state for resuming
    trainable = {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}
    tmp = path + ".tmp"
    torch.save({"trainable": trainable, "optimizer": optimizer.state_dict(), **meta}, tmp)
    os.replace(tmp, path)

def main():
    # Main training loop: setup, training, and evaluation
    os.makedirs(OUT_DIR, exist_ok=True)
    device = pick_device(DEVICE)
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    if device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    shim = install_focal_loss_fallback()

    if CHECK_LABELS:
        return check_labels()

    log = Logger(OUT_DIR)
    with open(__file__, "rb") as f:
        script = f.read()
    run_info = {
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if device.startswith("cuda") else platform.processor(),
        "torch": str(torch.__version__), "python": platform.python_version(), "sam3_commit": sam3_commit(),
        "focal_loss": "torch fallback" if shim else "triton", "script_sha256": hashlib.sha256(script).hexdigest(),
        "train_json_md5": file_md5(TRAIN_JSON), "dev_json_md5": file_md5(DEV_JSON),
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(OUT_DIR, "run_info.json"), "w") as f:
        json.dump(run_info, f, indent=2)
    log(f"device {device} ({run_info['gpu']}), torch {torch.__version__}, sam3 {run_info['sam3_commit']}, "
        f"focal loss {run_info['focal_loss']}")

    model = build_model()
    freeze(model, UNFREEZE_LAST)
    if NO_O2M:
        model.transformer.decoder.dac = False
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    log(f"trainable {n_train / 1e6:.1f}M of {n_all / 1e6:.1f}M parameters")
    model = model.to(device)

    if EVAL_ONLY:
        m = evaluate_dev(model, device, DEV_LIMIT)
        log(f"dev {fmt(m)}")
        log.record(kind="dev", epoch=0, **m)
        return

    criterion = build_loss(not NO_MASKS, MASK_POINTS).to(device)
    optimizer = torch.optim.AdamW(param_groups(model), betas=(0.9, 0.999))
    for g in optimizer.param_groups:
        log(f"  group {g['name']:16s} lr {g['base_lr']:.2e} wd {g['weight_decay']} "
            f"params {sum(p.numel() for p in g['params']) / 1e6:.2f}M")

    from sam3.model.utils.misc import copy_data_to_device

    start_epoch, step, best_ap, history = 0, 0, -1.0, []
    if RESUME:
        ck = torch.load(RESUME, map_location="cpu", weights_only=True)
        missing = model.load_state_dict(ck["trainable"], strict=False).unexpected_keys
        if missing:
            raise SystemExit(f"resume keys do not fit: {missing[:5]}")
        optimizer.load_state_dict(ck["optimizer"])
        start_epoch, step, best_ap, history = ck["epoch"], ck["step"], ck["best_ap"], ck["history"]
        log(f"resumed from {RESUME}: {start_epoch} epochs done, step {step}, best dev AP {best_ap:.4f}")
    elif not SKIP_INITIAL_EVAL:
        m = evaluate_dev(model, device, DEV_LIMIT)
        log(f"dev before training {fmt(m)}")
        log.record(kind="dev", epoch=0, step=0, **m)
        history.append({"epoch": 0, **m})

    dataset = build_train_dataset()
    steps_per_epoch = len(build_train_loader(device, 0, dataset)) // ACCUM
    total_steps = EPOCHS * steps_per_epoch
    log(f"{steps_per_epoch} optimizer steps per epoch (batch {BATCH} x accum {ACCUM}), {total_steps} total")

    t_start, epoch_seconds, done = time.time(), None, False
    for epoch in range(start_epoch, EPOCHS):
        # Check time limit
        if MAX_HOURS and epoch_seconds and (time.time() - t_start + epoch_seconds) / 3600 > MAX_HOURS:
            log(f"stopping before epoch {epoch + 1}: it would pass --max-hours {MAX_HOURS}")
            break
        loader = build_train_loader(device, epoch, dataset)
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
            # Skip steps with non-finite loss
            if not torch.isfinite(loss):
                bad += 1
                log(f"non-finite loss at epoch {epoch + 1} iter {it}, skipped ({bad} in a row)")
                optimizer.zero_grad(set_to_none=True)
                if bad >= 20:
                    raise SystemExit("too many non-finite losses")
                continue
            bad = 0
            (loss / ACCUM).backward()
            n_imgs += len(b.img_batch)
            sums["loss"] = sums.get("loss", 0.0) + loss.item()
            for k, v in losses.items():
                if "_aux_" not in k and not k.endswith("_fs") and torch.is_tensor(v):
                    sums[k] = sums.get(k, 0.0) + v.item()
            n_sum += 1

            # Optimizer step after gradient accumulation
            if (it + 1) % ACCUM:
                continue
            factor = lr_factor(step, total_steps, WARMUP, MIN_LR_RATIO)
            for g in optimizer.param_groups:
                g["lr"] = g["base_lr"] * factor
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], CLIP)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1

            if step % LOG_EVERY == 0:
                avg = {k: v / n_sum for k, v in sums.items()}
                rate = (time.time() - t_last) / (LOG_EVERY * ACCUM * BATCH)
                eta = rate * (len(loader) * BATCH - n_imgs) / 60
                log(f"epoch {epoch + 1} step {step} " + " ".join(f"{k} {v:.4f}" for k, v in avg.items())
                    + f" | lr {LR * factor:.2e} {rate:.2f}s/img mem {peak_mem_gb(device):.1f}GB eta {eta:.0f}min")
                log.record(kind="train", epoch=epoch + 1, step=step, lr=LR * factor, s_per_img=rate, **avg)
                sums, n_sum, t_last = {}, 0, time.time()

            if step == CHECK_STEP:
                m = evaluate_dev(model, device, CHECK_N)
                log(f"quick dev check at step {step} {fmt(m)}")
                log.record(kind="check", epoch=epoch + 1, step=step, **m)
                set_train_mode(model)
                t_last = time.time()

            if MAX_STEPS and step >= MAX_STEPS:
                done = True
                break

        # Evaluate at end of epoch
        epoch_seconds = time.time() - t_epoch
        m = evaluate_dev(model, device, DEV_LIMIT)
        history.append({"epoch": epoch + 1, "step": step, "train_minutes": round(epoch_seconds / 60, 1), **m})
        better = m["AP"] > best_ap
        log(f"epoch {epoch + 1} done in {epoch_seconds / 60:.1f} min, dev {fmt(m)}{'  (best)' if better else ''}")
        log.record(kind="dev", epoch=epoch + 1, step=step, train_minutes=epoch_seconds / 60, **m)
        meta = {"epoch": epoch + 1, "step": step, "dev": m, "history": history, "run_info": run_info}
        if better:
            best_ap = m["AP"]
            if not NO_SAVE:
                save_best(model, os.path.join(OUT_DIR, "best.pt"), {**meta, "script": script.decode()})
                log(f"saved best.pt (epoch {epoch + 1})")
        if not NO_SAVE:
            save_last(model, optimizer, os.path.join(OUT_DIR, "last.pt"), {**meta, "best_ap": best_ap})
        if done:
            break

    with open(os.path.join(OUT_DIR, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    log(f"finished, best dev AP {best_ap:.4f}, total {(time.time() - t_start) / 3600:.2f} h")

if __name__ == "__main__":
    main()
