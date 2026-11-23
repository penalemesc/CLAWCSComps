# Code
This directory hosts all of the different programs that were used in the development of our YOLO26 models.

## Folder Structure
```text
Code/
├── Tests                        # A set of tests that were created in the development of YOLO26
│     ├── FileTesting            # Currently hosts only the video used in our final qualitative analysis, but used to hold other files that were used for testing in the development of yoloBaseFileHandler.py
│     ├── OtherTests             # Tests that are not strictly related to the full implementation of YOLO26, but were interesting to explore.
│     │       ├── ConsoleTests   # A set of tests in which Conrado attempted to implement a global function that could be called from terminal to do different types of inferences
│     │       ├── otsuTest.py    # An implementation of Conrado's Otsu's thresholding assignment. It was used to learn how to best do real-time inferences, and what sorts of things might interfere with real time predictions
│     │       ├── TrainingTest   # A set of tests that were done to figure out how to best fine-tune YOLO26 to create our 'carYOLO26' model
│     └──     └── YOLOSaving     # A set of tests that attempted to figure out how to save inferences directly from code
├── yoloPredictions              # Stores a set of inferences that were saved directly from code, using both base YOLO26 and carYOLO26
├── yoloBaseFileHandler.py       # A file with two functions which create inferences based on input in the form of files (images or videos), and either show them to the user or store them in storage
├── yoloBaseLiveHandler.py       # A file with two functions which create live inferences and show them to the user, one of them can also save them to storage
└── Readme.md                    # The file you are currently reading
```
