# YOLO Labels

The files in `labels/` use the standard **YOLO object-detection label format**.

Each image has a corresponding `.txt` file with the same base filename.

For example:

```text
images/train/0004a4c0-d4dff0ad.jpg
labels/train/0004a4c0-d4dff0ad.txt
```

Each line in a label file represents one detected object.

The format is:

```text
class_id x_center y_center width height
```

All coordinates are **normalized between 0 and 1** relative to the image dimensions.

## Class IDs

This dataset uses two YOLO classes:

```text
0 = person
1 = car
```

Therefore:

```text
0 0.35 0.52 0.08 0.21
```

represents a `person`, while:

```text
1 0.62 0.48 0.20 0.25
```

represents a `car`.

## Bounding Box Format

YOLO uses the following format:

```text
class_id
x_center
y_center
width
height
```

The coordinates are normalized:

```text
x_center = box center X / image width
y_center = box center Y / image height
width    = box width / image width
height   = box height / image height
```

For example, for a `1280 × 720` image:

```text
1 0.5 0.5 0.25 0.20
```

means:

* Class: `1` (`car`)
* Bounding box center: `(50%, 50%)`
* Bounding box width: `25%` of image width
* Bounding box height: `20%` of image height

The actual pixel bounding box would therefore be approximately:

```text
Center X = 640
Center Y = 360
Width    = 320
Height   = 144
```