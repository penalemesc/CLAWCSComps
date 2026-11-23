# Code
This directory hosts all of the different programs that were used in the development of our YOLO26 models.


```text

Code/
├── Tests             # A set of tests that were created in the development of YOLO26
│     ├── FileTesting # Currently hosts only the video used in our final qualitative analysis, but used to hold other files that were used for testing in the development of yoloBaseFileHandler.py
│     ├── OtherTests  # Tests that are not strictly related to the full implementation of YOLO26, but were interesting to explore.
│     │       ├── ConsoleTests # A set of tests in which Conrado attempted to implement a global function that could be called from terminal to do different types of inferences
│     │       ├── otsuTest.py  # An implementation of Conrado's Otsu's thresholding assignment. It was used to learn how to best do real-time inferences, and what sorts of things might interfere with real time predictions
│     │       ├──

└── superpixel.py     # Superpixel segmentation exploration

main.py           # Entry point for image segmentation
│   ├── colorThreshold.py # HSV thresholding and GrabCut segmentation
│   ├── edge.py           # Edge detection exploration

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
