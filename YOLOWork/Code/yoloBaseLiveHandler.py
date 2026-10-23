# This has a base implementation of the YOLO model. 
# All of the code present here (at the time of 29/9/26 (d/m/y))is created by CPL based on his previous research, unless otherwise specified

# Import dependencies
import cv2
import os
from ultralytics import YOLO

# Define the output path
outputPath = "YOLOWork/Code/yoloPredictions/livePredictions/"

# Define camera object CHANGE THIS FOR YOUR OWN MACHINES, IN MINE (CPL), INDEX 2 IS THE ONE THAT HAS THE CAMERA THAT I ACTUALLY INTEND TO USE, IN MOST MACHINES IT IS 0, I am unsure at this point how to check other than simply run the program with different indexes
cam = cv2.VideoCapture(1)

# Define yolo model so that we can use it in predictions
model = YOLO("yolo26n.pt")

# Define the real-time prediction function
def realTimeFeed(cam, model):
    # While we do not quit the program, keep reading and showing the screen
    while True:
        # cam.read() returns a tuple, a boolean which defines whether or not we could read the camera, and an image which is 'read' at the given time when it gives 
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

# Infer in real time and save the inferences as an mp4 video for qualitative analysis
# Input is the ca
def realTimeSave(camera, model):
    # Get the camera width, height and Frames Per Second so that we can properly set up our save file
    w = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(camera.get(cv2.CAP_PROP_FPS))
    # Define our video codex as mp4
    fourCC = cv2.VideoWriter_fourcc(*'mp4v')
    # Define our video writer, so that we can actually store the videos somewhere
    outVid = cv2.VideoWriter((outputPath + 'liveVideo.mp4'), fourCC, fps/2, (w,h))   # Unfortunately I have not been able to find a way to read user input without showing said input to the user in the case of live feed, so CPL arbitrarily decided to halve fps as a test and it worked well enough on his bad PC that it was left as such.
    while True: 
         # cam.read() returns a tuple, a boolean which defines whether or not we could read the camera, and an image which is 'read' at the given time when it gives 
        ret, videoStream = camera.read()
        
        # If we cannot read the camera, send an error in console and exit the program
        if ret == False:
            print("Cannot read camera, exiting now")
            break
        
        # Use our model to predict on the video stream
        predictions = model(videoStream, imgsz = 640, verbose = False)
        predictedStream = predictions[0].plot()

        outVid.write(predictedStream)

        # Show our preview with the predictions that we make
        cv2.imshow('Preview', predictedStream)
            
        # Record user input, wait a milisecond so that the program can actually keep reading the videostream
        keyPressed = cv2.waitKey(1)
        
        # If we pressed q, quit
        if keyPressed == ord('q'):
            break

realTimeSave(cam, model)
# Allow the camera to be used by other programs and destroy all preview windows
cam.release()
cv2.destroyAllWindows()