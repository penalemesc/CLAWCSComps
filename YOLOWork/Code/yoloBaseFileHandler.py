# Import dependencies
import cv2
import os
from ultralytics import YOLO

# Define the input path
#inputPath = "SharedGT/yolo/images/val/"
inputPath = "YOLOWork/Code/Tests/fileInput/"
# We probably want this to be the actual validating files, however because I (CPL) have not finished training at this point (its going to take 45 hours on my personal computer), I just put my own files for now

# Define the input files 
files = os.listdir(inputPath)

# Define output path
outputPath = "YOLOWork/Code/yoloPredictions/filePredictions/"

# This is here for testing
# Define the YOLO model to be used
boxModel = YOLO("YOLOWork/Code/Tests/TrainingTests/boundingCarYOLO26/weights/boundingCarYOLO.pt")

baseModel = YOLO("yolo26m.pt")

# Infers on files and displays the result of the inference to the user. 
# It is meant as a simple fast test
# Input is a set of files, the path where those files are and the model that is used for predicting
def predictFiles(inFiles, inPath, model):
    # Whilst we still have files that we have not predicted on, keep reading
    for file in inFiles:
        print(file)
        isImage = False
        if file.find(".jpg") != -1 or file.find(".png") != -1 or file.find(".jpeg") != 1:
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
        if file.find(".jpg") != -1 or file.find(".png") != -1 or file.find(".jpeg") != 1: 
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
            cv2.imwrite((outPath + 'car'+ file), predictedFile)
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


#predictFiles(files, inputPath, baseModel)
saveFromFile(files, inputPath, outputPath, boxModel)

cv2.destroyAllWindows()