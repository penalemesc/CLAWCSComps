# This is just me (CPL) replicating my old tests using live feed, from what I can see from my old notes 
# I based it off of 
# https://opencv.org/reading-and-writing-videos-using-opencv/
import cv2
import os
from ultralytics import YOLO

# Define the output path
outputPath = "YOLOWork/Code/videoRecordings/"

# Define camera object CHANGE THIS FOR YOUR OWN MACHINES, IN MINE (CPL), INDEX 2 IS THE ONE THAT HAS THE CAMERA THAT I ACTUALLY INTEND TO USE, IN MOST MACHINES IT IS 0, I am unsure at this point how to check other than simply run the program with different indexes
cam = cv2.VideoCapture(0)

# Define yolo model so that we can use it in predictions, decided to use the segmentation model since that is something we might use in our project too
model = YOLO("yolo26n-seg.pt")

# I got this from https://github.com/orgs/ultralytics/discussions/5200 in the comment made by mariam-162
fps = float(cam.get(cv2.CAP_PROP_FPS))
# The w and h need to be the same as the input file, unless we resize the frame before we write
w = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the video codec to be mp4
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
outVid = cv2.VideoWriter(outputPath +"test.mp4", fourcc, fps, (w,h))
while True:
    # cam.read() returns a tuple, a boolean which defines whether or not we could read the camera (ret), and an image which is 'read' at the given time when it gives 
    ret, videoStream = cam.read()

    predictions = model(videoStream, imgsz = 640, verbose = False)
    predictedStream = predictions[0].plot()

    outVid.write(predictedStream)

    cv2.imshow("Preview", predictedStream)

    # Record user input, wait a milisecond so that the program can actually keep reading the videostream
    keyPressed = cv2.waitKey(1)
    
    # If we pressed q, quit
    if keyPressed == ord('q'):
        break

# Allow the camera to be used by other programs and destroy all preview windows
cam.release()
outVid.release()
cv2.destroyAllWindows()