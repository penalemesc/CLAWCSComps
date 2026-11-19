# All code written by CPL
# Simple test to see if I can accept input from console in python (had not done it since sophomore year)
# I do this because I want to accept input from console to see which of the functions in yolo we need to run (whether it is simple live detection, live detection + saving output, from file detection or from file detection + saving output)
import numpy
while True:
    print("Hi please write a number")
    userInput = input()
    
    if userInput == "q":
        break
    elif userInput.isdigit() == False:
        print("Wrong type of input, goodbye")
        exit()
    # This feels like a bad solution to the problem https://inventwithpython.com/blog/how-to-test-multiple-variables-against-a-single-value-and-single-variables-against-multiple-values.html
    # I need to research how python works more because I should be able to just use != 1 | 2 | 3, but that does not work at all which was weird
    elif int(userInput) not in (1, 2, 3):
        print("Not a valid option, goodbye")
        exit()
    print("Your input:", userInput)