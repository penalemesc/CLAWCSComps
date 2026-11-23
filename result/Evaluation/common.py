# Shared helpers for the evaluation scripts
# Predictions are keyed by image file name

import json
import os
import time

import torch

# Folder paths, all found from the location of this file
EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(EVAL_DIR)
DATASET_DIR = os.path.join(REPO_DIR, "dataset")
OUTPUT_DIR = os.path.join(EVAL_DIR, "outputs")
WORKSPACE_DIR = os.path.dirname(REPO_DIR)


# Input: images_dir (folder), limit (max number of images, optional), file_list (text file, one name per line, optional)
# Output: sorted list of image file names
# Does: takes the names from file_list if given, otherwise lists the jpg, jpeg and png files in images_dir
def list_images(images_dir, limit=None, file_list=None):
    if file_list:
        names = sorted(line.strip() for line in open(file_list) if line.strip())
    else:
        names = sorted(f for f in os.listdir(images_dir) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    return names[:limit] if limit else names


# Input: requested (device name, or None)
# Output: device name ("cuda", "mps" or "cpu")
# Does: returns the requested device, otherwise the best one available
def pick_device(requested=None):
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# Input: device name
# Does: waits until the GPU has finished its queued work, so that timings are correct
def sync(device):
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


# Input: device name
# Does: resets the peak memory counter on CUDA (does nothing on other devices)
def reset_peak_memory(device):
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()


# Input: device name
# Output: GPU memory in MiB (nan on CPU)
# Does: reports the peak allocated memory on CUDA, or the current driver memory on MPS
def memory_mb(device):
    if device.startswith("cuda"):
        return torch.cuda.max_memory_allocated() / 2**20
    if device == "mps":
        return torch.mps.driver_allocated_memory() / 2**20
    return float("nan")


# Input: device name
# Output: elapsed time in milliseconds, stored in .ms
# Does: times the code inside a "with" block, and syncs the GPU before and after
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


# Input: path (output file), obj (data that JSON can store)
# Does: writes to a temporary file and then renames it, so a crash never leaves a half-written file
def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


# Input: device name
# Output: dict with the torch version, the device and the GPU name
# Does: records the environment, to be saved with the results
def environment_info(device):
    info = {"torch": torch.__version__, "device": device}
    if device.startswith("cuda"):
        info["gpu"] = torch.cuda.get_device_name(0)
    elif device == "mps":
        import platform
        info["gpu"] = f"Apple MPS ({platform.machine()}, macOS {platform.mac_ver()[0]})"
    return info
    
