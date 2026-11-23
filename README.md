# CLAWtonomous Decisions: Vehicle Detection & Segmentation for Autonomous Driving

[![Carleton College Comps](https://img.shields.io/badge/Carleton%20College-CS%20Senior%20Comps-blue.svg)](https://www.carleton.edu/computer-science/)
[![Dataset](https://img.shields.io/badge/Dataset-BDD100K-orange.svg)](https://www.bdd100k.com/)
[![Models](https://img.shields.io/badge/Models-YOLO26%20%7C%20SAM%203%20%7C%20OpenCV-green.svg)]()

---

## Overview

Autonomous vehicle perception systems must balance **inference speed**, **hardware constraints**, and **segmentation precision**. This project investigates the architectural and practical trade-offs between three distinct paradigms for vehicle detection and segmentation using terrestrial driving footage from the [BDD100K](https://www.bdd100k.com/) dataset:

1. **Lightweight Real-Time Deep Learning:** **YOLO26m** (Ultralytics) — bounding-box detection optimized for low latency and minimal compute.
2. **Heavy Foundation Model:** **SAM 3** (Segment Anything Model 3, Meta) — promptable instance segmentation (~850M parameters) offering high pixel-level accuracy.
3. **Classical Computer Vision Baseline:** **Traditional Python OpenCV Algorithm** — non-neural vehicle segmentation using HSV color thresholding, geometric taillight pairing, and GrabCut graph-cut optimization.

We evaluated both zero-shot and fine-tuned versions of YOLO26 and SAM 3 against a shared ground truth, measuring detection accuracy, latency, and memory footprint on an NVIDIA RTX A2000 GPU.

---

## Key Results & Trade-Offs

Both models were evaluated on the 998-image test set using standardized COCO evaluation metrics on an NVIDIA RTX A2000 GPU (results recorded in [`result/final.csv`](result/final.csv)):

| Metric | YOLO26m (Zero-Shot) | YOLO26m (Fine-Tuned) | SAM 3 (Zero-Shot) | SAM 3 (Fine-Tuned) |
|---|:---:|:---:|:---:|:---:|
| **Box AP** | 0.463 | 0.534 | 0.532 | **0.573** |
| **AP50** | 0.661 | 0.741 | 0.766 | **0.801** |
| **AP75** | 0.490 | 0.567 | 0.563 | **0.606** |
| **AP Small** | 0.222 | 0.319 | 0.319 | **0.356** |
| **F1 Score** | 0.688 | 0.755 | 0.763 | **0.786** |
| **Pixel IoU (Box)** | 0.531 | 0.740 | 0.716 | **0.744** |
| **Latency (ms / frame)** | **10.4 ms** | **10.4 ms** | 429.1 ms | 426.7 ms |
| **Peak GPU VRAM (MB)** | **159 MB** | **159 MB** | 5,990 MB | 5,992 MB |

### Key Takeaways:
- **Speed vs. Precision:** SAM 3 fine-tuned achieves higher overall precision (AP₅₀ = 0.801 vs. 0.741), but YOLO26m runs **~41× faster** (10.4 ms vs. 426.7 ms, or ~96 FPS vs. ~2.3 FPS) with **~37× less VRAM** (159 MB vs. 5,992 MB).
- **Deployment Viability:** For real-time autonomous vehicle reaction loops, YOLO26m operates well within safety-critical latency budgets, while SAM 3 requires substantial acceleration or edge pruning for live deployment.
- **Fine-Tuning Impact:** Domain-specific fine-tuning on BDD100K significantly improved small-object recall (\(\text{AP}_{\text{small}}\) improved from 0.222 → 0.319 for YOLO and 0.319 → 0.356 for SAM 3).

---

## Repository Structure

```text
CLAWCSComps/
├── SAM3Work/             # SAM 3 model codebase & training pipeline (Amery & Walter)
│   └── Code/             # Dockerfile, fine-tuning scripts, and configuration
├── YOLOWork/             # YOLO26 model pipeline & inference handlers (Conrado & Luke)
│   ├── Code/             # Live webcam/OBS and file-based video inference scripts
│   ├── ConradoWorkNotes/ # Weekly engineering notes and logs
├── tradalgo/             # Classical CV vehicle segmentation baseline
│   ├── main.py           # Entry point for image segmentation
│   ├── colorThreshold.py # HSV thresholding and GrabCut segmentation
│   ├── edge.py           # Edge detection exploration
│   └── superpixel.py     # Superpixel segmentation exploration
├── SharedGT/             # Unified ground-truth labels and evaluation splits
│   ├── build_shared_gt.py # Generates shared labels from BDD100K panoptic data
│   ├── check_manifest.py  # Checksum verification for dataset integrity
│   ├── manifest.csv      # Split mappings and MD5 checksums for all images
│   ├── coco/             # COCO-format annotations (train, dev, test)
│   └── yolo/             # YOLO-format annotation builder and yaml configs
├── dataset/              # Sample images, masks, and dataset documentation
├── result/               # Evaluation framework and output data
│   ├── Evaluation/       # COCO evaluation, agreement analysis, and benchmarking
│   ├── graphs/           # Visualization plots and precision-recall curves
│   └── final.csv         # Consolidated benchmark results
├── website/              # Project website, interactive demos, and Comps posters
└── runs/                 # YOLO model training checkpoints and validation outputs
```

---

## Components & Subsystems

### 1. YOLO26 (`YOLOWork/`)
* Implements real-time inference workflows for both pre-recorded video files and live video feeds (via webcam or OBS Virtual Camera).
* Key scripts:
  - [`YOLOWork/Code/yoloBaseLiveHandler.py`](YOLOWork/Code/yoloBaseLiveHandler.py): Real-time stream processing with OpenCV and YOLO.
  - [`YOLOWork/Code/yoloBaseFileHandler.py`](YOLOWork/Code/yoloBaseFileHandler.py): File batch inference and annotated video generation.

### 2. SAM 3 (`SAM3Work/`)
* Implements the promptable instance segmentation pipeline using Meta's SAM 3 model with the text prompt `"car"`.
* Key scripts:
  - [`SAM3Work/Code/Dockerfile`](SAM3Work/Code/Dockerfile): Containerized environment for GPU training.
  - [`SAM3Work/Code/fine-tune.py`](SAM3Work/Code/fine-tune.py): Fine-tuning script with learning rate scheduling, logging, and checkpointing.

### 3. Traditional Computer Vision Baseline (`tradalgo/`)
* A rule-based, non-neural segmentation algorithm developed to contrast deep learning approaches with classical computer vision:
  - Detects rear-facing vehicles using red taillight pairs in HSV color space.
  - Generates bounding proposals and refines boundaries using OpenCV's GrabCut graph-cut algorithm.
* Key script:
  - [`tradalgo/main.py`](tradalgo/main.py): Runs traditional segmentation on input images and outputs binary masks.

### 4. Shared Ground Truth & Evaluation (`SharedGT/` & `result/`)
* Standardized splits derived from BDD100K panoptic segmentation (`bdd100k_pan_seg_labels_trainval.zip`) to guarantee fair evaluation across bounding box and mask models:
  - **Train:** 6,497 images (65,587 cars)
  - **Dev (Validation):** 499 images (5,019 cars) — used for hyperparameter and threshold selection
  - **Test:** 998 images (10,733 cars) — reserved strictly for final benchmark scoring
* Evaluator: [`result/Evaluation/evaluate.py`](result/Evaluation/evaluate.py) computes standard COCO mAP (AP, $\text{AP}_{50}$, $\text{AP}_{75}$, $\text{AP}_{\text{small}}$), F1 scores, pixel IoU, latency, and memory footprints.

### 5. Project Website (`website/`)
* Contains the complete static website presenting our research proposal, methodology, model explanations, side-by-side video demos, and final senior Comps posters:
  - [`website/index.html`](website/index.html): Abstract and project overview
  - [`website/projectDesc.html`](website/projectDesc.html): Project motivation and dataset details
  - [`website/yolo26.html`](website/yolo26.html) & [`website/sam3.html`](website/sam3.html): Model-specific breakdowns
  - [`website/opencv.html`](website/opencv.html): Tradition
