def printMenu():
    print("1: Run live-feed bounding box inference on a connected camera")
    print("2: Run and Save live-feed bounding box inferences from a connected camera")
    print("3: Run live-feed segmentation inference on a connected camera")
    print("4: Run and Save live-feed segmentation inference on a connected camera")
    print("5: Run bounding box inference on a set of files")
    print("6: Run and Save bounding box inference on a set of files")
    print("7: Run segmentation inference on a set of files")
    print("8: Run and Save segmentation inference on a set of files")
def printBase():
    print("Welcome to our YOLO implementation")
    print("Press the number attached to one of the following options with \nwhich to run our program\n")

def handleUserinput():
    # If we write inside of input() it will print it to the console allowing for a much more seamless 
    x = input("Your choice: ")
    if x == "1":
        print("test success")
    elif x != "1":
        print("Cool it works")

printBase()
printMenu()
handleUserinput()