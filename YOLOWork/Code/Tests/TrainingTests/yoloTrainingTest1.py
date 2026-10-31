# This code is based on the one presented here: https://learnopencv.com/train-yolov8-on-custom-dataset/

import cv2
import os
from ultralytics import YOLO

# This is a test that I (CPL) am running on my laptop, I do not want it to explode so I am using nano still, this test is mainly to see how YOLO is actually trained and whehter or not I can actually run it on here
model = YOLO("yolo26n.pt")

# Huizi said that we should keep the default settings for evaluation and inference, so I kept the imgsz to the default YOLO size, but I did change 
results = model.train(
    data = 'SharedGT/yolo/car.yaml',
    imgsz=640,
    epochs=1,
    batch=2,
    device="mps", #mps is gpu on apple silicon according to https://docs.ultralytics.com/modes/train
    name='yolo26Test2'
)