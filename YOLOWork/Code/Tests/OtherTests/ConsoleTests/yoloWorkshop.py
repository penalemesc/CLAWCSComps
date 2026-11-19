
# This does not currently work, I am unsure why that is 

# Import dependencies 
import time
import cv2
import os
from ultralytics import YOLO
from yoloBaseLiveHandler import realTimeFeed, realTimeSave
from yoloBaseFileHandler import predictFiles, saveFromFile
from YOLOWork.Code.Tests.FunTests.consoleHelper import printBase, printMenu

# Define the model that we will use for our bounding box tests
boxModel = YOLO("YOLOWork/Code/Tests/TrainingTests/carYOLO26v2/weights/boundingCarYOLO.pt")

# Define the model that we will use for our mask -- TO DO: TRAIN THIS
#maskModel = YOLO("noModelYet")

# For simplicity's sake the camera, models, input and output paths will be defined through code
inputPath = "YOLOWork/Code/Tests/FileTesting/"

outputPath = "YOLOWork/Code/yoloPredictions/livePredictions/"


printBase()
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
        
