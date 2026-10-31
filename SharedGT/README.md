# SharedGT

Shared labels and splits for all four runs: YOLO and SAM 3, zero-shot and fine-tuned.

Labels come from BDD100K's panoptic segmentation (`bdd100k_pan_seg_labels_trainval.zip`): one box and one mask per car, for the same images as `dataset/`. `dataset/labels` is no longer used. 6 images have no panoptic label and are dropped (`excluded.txt`).

- `train` is `dataset/images/train` minus dev.
- `dev` is 499 fixed images from train. This is the validation set for fine-tuning: pick epochs, checkpoints and thresholds on it.
- `test` is `dataset/images/val` (BDD calls it val), only for final scores.

`manifest.csv` lists every image with its split and md5. Run `python SharedGT/check_manifest.py` to confirm your copy matches.

| split | images | cars |
|---|---|---|
| train | 6497 | 65587 |
| dev | 499 | 5019 |
| test | 998 | 10733 |

## Use

YOLO: run `python SharedGT/yolo/make_yolo_labels.py` once. It unzips `yolo/labels.zip` and links `yolo/images` to `dataset/images`. Then:

    yolo detect train data=SharedGT/yolo/car.yaml model=yolo26m.pt

`val` in the yaml is dev.

SAM 3: `coco/train_car.json` and `coco/dev_car.json` go with `img_folder: dataset/images/train/`, `coco/test_car.json` with `dataset/images/val/`. The prompt is `car`.

Score both models with the same COCO evaluator on `coco/test_car.json`.

Rebuild (needs the panoptic zip unpacked): `python SharedGT/build_shared_gt.py --pan-seg <...>/bdd100k/labels/pan_seg`
