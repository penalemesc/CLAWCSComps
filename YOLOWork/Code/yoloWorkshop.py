# CPL could not think of a better title for the file in which all things YOLO should live, and so here we are

# Import dependencies 
import time
import cv2
import os
from ultralytics import YOLO
from yoloBaseLiveHandler import realTimeFeed, realTimeSave
from yoloBaseFileHandler import predictFiles, saveFromFile
from consoleHelper import printBase, printMenu

# Define the model that we will use for our bounding box tests
boxModel = YOLO("YOLOWork/Code/Tests/TrainingTests/carYOLO26v2/weights/boundingCarYOLO.pt")

# Define the model that we will use for our mask -- TO DO: TRAIN THIS
#maskModel = YOLO("noModelYet")

# For simplicity's sake the camera, models, input and output paths will be defined through code
inputPath = "YOLOWork/Code/Tests/FileTesting/"

outputPath = "YOLOWork/Code/yoloPredictions/livePredictions/"


printBase()

while True:
    printMenu()
    usrInput = input()
    if usrInput == "1":
        camera = cv2.VideoCapture(1)
        realTimeFeed(camera, boxModel)
        time.sleep(250)
    elif usrInput == "2":
        camera = cv2.VideoCapture(1)
        realTimeSave(camera, outputPath, boxModel)
    else:
        exit()
        
