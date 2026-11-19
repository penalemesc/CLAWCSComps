# This code is a test to see how well a theoretical third "traditional" algorithm 
# could work in real-time image processing.
# It was done to see the feasability of adding a third algorithm to our project 
# All code is based on CPL's Otsu's threshold assigment or in the "yoloBase.py" file
# which was all created by CPL (based on whatever sources he used)

# TO DO: ALL COMMENTS ON THIS FILE
import cv2

cam = cv2.VideoCapture(1)

while True: 
    ret, videoStream = cam.read()

    grayscale = cv2.cvtColor(videoStream, cv2.COLOR_BGR2GRAY)
    # So fun little thing, because of how I do the otsu thresholding (poorly), resizing the image is a somewhat good idea for improving performance
    grayscale = cv2.resize(grayscale, (320, 200))
    """ distribution = grayscale.ravel()
    mean = distribution.mean() """
    # I did a test, it 
    otsuThresh = 150 #round(mean)


    for x in range(len(grayscale)):
        for y in range(len(grayscale[x])):
            if grayscale[x,y] > otsuThresh:
                grayscale[x,y] = 255
            else:
                grayscale[x,y] = 0

    # Resizing it back up produces interesting results 
    grayscale = cv2.resize(grayscale, (1200, 750))
    cv2.imshow("Otsu's Image", grayscale)
    keyPress = cv2.waitKey(1)
    if keyPress == ord('q'):
        break

# Allow the camera to be used by other programs and destroy all preview windows
cam.release()
cv2.destroyAllWindows()