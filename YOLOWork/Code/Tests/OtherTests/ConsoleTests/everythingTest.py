# Import dependencies
import cv2
import os
from ultralytics import YOLO
import time

# Define the input path
inputPath = "SharedGT/yolo/images/val/"
# We probably want this to be the actual validating files, however because I (CPL) have not finished training at this point (its going to take 45 hours on my personal computer), I just put my own files for now

# Define the input files 
files = os.listdir(inputPath)

# Define output path
outputPath = "YOLOWork/Code/yoloPredictions/filePredictions/"

# This is here for testing
# Define the YOLO model to be used
boxModel = YOLO("YOLOWork/Code/Tests/TrainingTests/boundingCarYOLO26/weights/boundingCarYOLO.pt") 

# Print out menu options
def printMenu():
    print("BOUNDING BOX - LIVE FEED")
    print("1: Run live-feed bounding box inference on a connected camera")
    print("2: Run and Save live-feed bounding box inferences from a connected camera\n")
    print("SEGMENTATION - LIVE FEED")
    print("3: Run live-feed segmentation inference on a connected camera")
    print("4: Run and Save live-feed segmentation inference on a connected camera\n")
    print("BOUNDING BOX - FILES")
    print("5: Run bounding box inference on a set of files")
    print("6: Run and Save bounding box inference on a set of files\n")
    print("SEGMENTATION - FILES")
    print("7: Run segmentation inference on a set of files")
    print("8: Run and Save segmentation inference on a set of files\n")
    print("q: Exit this program")

# Separate base printing so that we can re-print the menu to screen without printing the base
def printBase():
    print("Welcome to our YOLO implementation")
    print("Press the number attached to one of the following options with \nwhich to run our program\n")

# Infers on files and displays the result of the inference to the user. 
# It is meant as a simple fast test
# Input is a set of files, the path where those files are and the model that is used for predicting
def predictFiles(inFiles, inPath, model):
    # Whilst we still have files that weh have not predicted on
    for file in inFiles:
        print(file)
        isImage = False
        if file.find(".jpg") != -1 or file.find(".png") != -1:
            isImage = True
        elif file.find(".mp4") != -1 or file.find(".avi") != -1:
            isImage = False
        else: 
            print("File \"" + file + "\"not accepted, please remove it from the source folder and run this program again")
            exit()
        print(isImage)
        if isImage == True:
            prediction = model((inPath + file), imgsz = 640, verbose = False)
            predictedFile = prediction[0].plot()
            cv2.imshow("Predicted Image", predictedFile)
            # 0 because it is not a stream and we want to actually wait until the user sees the image
            keyPressed = cv2.waitKey(0)
            # We also do not want to do anything with user input, we just want to move on to the next image

        else: 
            cam = cv2.VideoCapture(inPath + file)
            while True:
                ret, videoStream = cam.read()
                if ret == False:
                    break
                prediction = model(videoStream, imgsz = 640, verbose = False)
                predictedFile = prediction[0].plot()
                cv2.imshow("Predicted Video", predictedFile)
                keyPressed = cv2.waitKey(1)
                if keyPressed == ord('n'):
                    break
            cam.release()


def saveFromFile(inFiles, inPath, outPath, model):
    for file in inFiles:
        isImage = False
        if file.find(".jpg") != -1 or file.find(".png") != -1:
            isImage = True
        elif file.find(".mp4") != -1:
            fourCC = cv2.VideoWriter_fourcc(*'mp4v')
        elif file.find(".avi") != -1:
            fourCC = cv2.VideoWriter_fourcc(*'XVID')
        else: 
            print("File \"" + file + "\" not accepted, please remove it from the source folder, and run this program again")
            exit()

        if isImage == True:
            prediction = model((inPath + file), imgsz = 640, verbose = False)
            predictedFile = prediction[0].plot()
            cv2.imwrite((outPath + file), predictedFile)
        else: 
            cam = cv2.VideoCapture(inPath + file)
            w = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cam.get(cv2.CAP_PROP_FPS))
            outVid = cv2.VideoWriter((outPath + file), fourCC, fps, (w,h))
            while True:
                ret, videoStream = cam.read()
                if ret == False:
                    break
                predictions = model(videoStream, imgsz = 640, verbose = False)
                predictedStream = predictions[0].plot()
                outVid.write(predictedStream)

    cam.release()
    outVid.release()

# Define the real-time prediction function
def realTimeFeed(camera, model):
    # While we do not quit the program, keep reading and showing the screen
    while True:
        # cam.read() returns a tuple, a boolean which defines whether or not we could read the camera, and an image which is 'read' at the given time when it is called
        ret, videoStream = camera.read()

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
    camera.release()
    cv2.destroyAllWindows()
        

# Infer in real time and save the inferences as an mp4 video for qualitative analysis
# Input is the camera that we are using and the model that we want to use
def realTimeSave(camera, outPath, model):
    # Get the camera width, height and Frames Per Second so that we can properly set up our save file
    w = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(camera.get(cv2.CAP_PROP_FPS))
    # Define our video codex as mp4
    fourCC = cv2.VideoWriter_fourcc(*'mp4v')
    # Define our video writer, so that we can actually store the videos somewhere
    outVid = cv2.VideoWriter((outPath + 'liveVideo.mp4'), fourCC, fps/2, (w,h))   # Unfortunately I have not been able to find a way to read user input without showing said input to the user in the case of live feed, so CPL arbitrarily decided to halve fps as a test and it worked well enough on his bad PC that it was left as such.
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
    camera.release()
    outVid.release()
    cv2.destroyAllWindows()


while True:
    printMenu()
    usrInput = input()
    if usrInput == "1":
        camera = cv2.VideoCapture(1)
        realTimeFeed(camera, boxModel)
    elif usrInput == "2":
        camera = cv2.VideoCapture(1)
        realTimeSave(camera, outputPath, boxModel)
    else:
        exit()
    cv2.destroyAllWindows()
    time.sleep(2)