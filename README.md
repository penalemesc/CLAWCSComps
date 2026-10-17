# SharedGT

Shared labels and splits for all four runs: YOLO and SAM 3, zero-shot and fine-tuned. No images are copied; everything points into `dataset/`.

- Boxes are our `dataset/labels`, car only.
- 6 images are dropped (`excluded.txt`). They are portrait photos whose masks don't match the image.
- `test` is `dataset/images/val`, only for final scores.
- `dev` is 499 fixed images from train, for picking epochs, checkpoints and thresholds.
- `train` is the rest of train.

| split | images | car boxes |
|---|---|---|
| train | 6497 | 27751 |
| dev | 499 | 2151 |
| test | 998 | 4357 |

## Use

YOLO (`val` in the yaml is dev):

    yolo detect train data=SharedGT/yolo/car.yaml model=yolo26m.pt classes=[1]

SAM 3: `coco/train_car.json` and `coco/dev_car.json` go with `img_folder: dataset/images/train/`, `coco/test_car.json` with `dataset/images/val/`. The prompt is `car`.

Score both models with the same COCO evaluator on `coco/test_car.json`. `splits/` has plain file-name lists if you need them.

Our boxes come from the segmentation masks, so touching cars share one box. This keeps box AP low for both models.

Rebuild: `python SharedGT/build_shared_gt.py`
