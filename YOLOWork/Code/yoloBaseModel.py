# This has a base implementation of the YOLO model. 
# All of the code present here (at the time of 29/9/26 (d/m/y))is created by CPL based on his previous research, unless otherwise specified

# Import dependencies
import cv2
import os
from ultralytics import YOLO

# Define the output path
output = "YOLOWork/Code/videoRecordings"

# Define camera object CHANGE THIS FOR YOUR OWN MACHINES, IN MINE (CPL), INDEX 2 IS THE ONE THAT HAS THE CAMERA THAT I ACTUALLY INTEND TO USE, IN MOST MACHINES IT IS 0, I am unsure at this point how to check other than simply run the program with different indexes
cam = cv2.VideoCapture(2)

# Define yolo model so that we can use it in predictions
model = YOLO("yolov8n.pt")

# Define the video recording function -- IMPLEMENT LATER

# White we do not quit the program, keep reading and showing the screen
while True:
    # cam.read∏() returns a tuple, a boolean which defines whether or not we could read the camera, and an image which is 'read' at the given time when it gives 
    ret, videoStream = cam.read()

    # If we cannot read the camera, send an error in console and exit the program
    if ret == False:
        print("Cannot read camera, exiting now")
        break

    # Use our model to predict on the video stream
    predictions = model(videoStream, imgsz = 640, verbose = False)
    predictedStream = predictions[0].plot()
    
    # Show our preview with the predictions that we make
    cv2.imshow('Preview', predictedStream)

    # Record user input, wait a milisecond so that the program can actually keep reading the videostream
    keyPressed = cv2.waitKey(1)

    # If we pressed q, quit
    if keyPressed == ord('q'):
        break
    # IMPLEMENT SAVING VIDEOS AT SOME POINT
    # elif keyPressed == ord('s'):

# Allow the camera to be used by other programs and destroy all preview windows
cam.release()
cv2.destroyAllWindows()