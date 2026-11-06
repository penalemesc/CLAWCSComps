# Shared helpers for the zero-shot evaluation scripts
# Predictions are keyed by image file name

import json
import os
import time

import torch

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(EVAL_DIR)
DATASET_DIR = os.path.join(REPO_DIR, "dataset")
OUTPUT_DIR = os.path.join(EVAL_DIR, "outputs")
WORKSPACE_DIR = os.path.dirname(REPO_DIR)


def list_images(images_dir, limit=None, file_list=None):
    if file_list:
        names = sorted(line.strip() for line in open(file_list) if line.strip())
    else:
        names = sorted(f for f in os.listdir(images_dir) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    return names[:limit] if limit else names


def pick_device(requested=None):
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def sync(device):
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def reset_peak_memory(device):
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()


def memory_mb(device):
    # CUDA reports a peak
    if device.startswith("cuda"):
        return torch.cuda.max_memory_allocated() / 2**20
    if device == "mps":
        return torch.mps.driver_allocated_memory() / 2**20
    return float("nan")


class Timer:
    def __init__(self, device):
        self.device = device

    def __enter__(self):
        sync(self.device)
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        sync(self.device)
        self.ms = (time.perf_counter() - self.start) * 1000


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def environment_info(device):
    info = {"torch": torch.__version__, "device": device}
    if device.startswith("cuda"):
        info["gpu"] = torch.cuda.get_device_name(0)
    elif device == "mps":
        import platform
        info["gpu"] = f"Apple MPS ({platform.machine()}, macOS {platform.mac_ver()[0]})"
    return info
