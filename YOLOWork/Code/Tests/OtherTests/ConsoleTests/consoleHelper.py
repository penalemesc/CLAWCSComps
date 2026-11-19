# Originally a test to figure out a good looking menu for YOLO, became the base for the proper menu,
# My idea is to just copy paste a finished version of this to whichever file I define as the final menu/proper executable
# and in that final file use inports from the different files

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

""" # TO DO: implement the actual function described here
# User input handler, takes in an integer and starts executing whichever option was selected
# If a non-integer or a non-valid option is selected, then offer the user the option to write a new input
# if return is "Q" or "q" quit the program
def handleUserinput():
    # If we write inside of input() it will print it to the console allowing for a much more seamless 
    x = input("Your choice: ")
    if x == "1":
        print("test success")
    elif x != "1":
        print("Cool it works") """
