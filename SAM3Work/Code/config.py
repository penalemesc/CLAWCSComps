"""Configuration constants for SAM3 fine-tuning."""

import os

# Determine directory structure
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
WORKSPACE = os.path.dirname(REPO)

# Model architecture constants
FROZEN = ("backbone.vision_backbone.trunk.", "backbone.language_backbone.encoder.")
TRUNK_DEPTH = 32

# Output and data paths
OUT_DIR = os.path.join(WORKSPACE, "runs", "sam3_ft")
WEIGHTS = os.path.join(WORKSPACE, "sam3.pt")
TRAIN_JSON = os.path.join(REPO, "SharedGT", "coco", "train_car.json")
DEV_JSON = os.path.join(REPO, "SharedGT", "coco", "dev_car.json")
IMAGES = os.path.join(REPO, "dataset", "images", "train")

# Training text prompt
PROMPT = "car"

# Training hyperparameters
EPOCHS = 8
BATCH = 1
ACCUM = 4
LR = 2e-5
LR_VISION = 1e-5
LR_TEXT = 2e-6
LAYER_DECAY = 0.9
WD = 0.1
CLIP = 0.1

# Learning rate schedule
WARMUP = 300
MIN_LR_RATIO = 0.1

# Data augmentation
MIN_SIZE = 480
MASK_POINTS = 12544

# Model training options
UNFREEZE_LAST = 0
WORKERS = 4
SEED = 0

# Optional parameters
DEVICE = None
LIMIT = None
DEV_LIMIT = None
CHECK_STEP = 500
CHECK_N = 50
MAX_STEPS = None
MAX_HOURS = None
LOG_EVERY = 50
CHECK_LABELS = 0
RESUME = None

# Boolean flags
NO_FLIP = False
NO_MASKS = False
NO_O2M = False
SKIP_INITIAL_EVAL = False
NO_SAVE = False
EVAL_ONLY = False
