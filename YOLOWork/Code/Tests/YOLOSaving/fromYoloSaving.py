# I could not get this to work at all
# This code is based on the one present in the YOLO documentation: https://docs.ultralytics.com/modes/predict
# Import dependencies
from ultralytics import YOLO
import os
import cv2

# Define the output path for our videos
outputPath = "YOLOWork/Code/videoRecordings/"

# Define the camera that we will use
#cam = cv2.VideoCapture(1)

# Define the model we will use
model = YOLO("yolo26n.pt")

prediction = model.predict(1, verbose = False, stream = True, save=True)
# this is all from yolo
for result in prediction:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    # YOLO .show is bad when used with video
    #result.show()  # display to screen
    result.save(filename='result.mp4')  # save to disk