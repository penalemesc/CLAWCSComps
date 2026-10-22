# Import dependencies
from ultralytics import YOLO

# Define model that we are using as a base
model = YOLO("yolo26m.pt")
 
# Define training parameters, with the device being mps because CPL (creator of this training) is using an apple silicon device for training
results = model.train(data = 'SharedGT/yolo/car.yaml', imgsz = 640, project = '../../CLAWCSComps/YOLOWork/Code/Tests/TrainingTests/', epochs = 25, device=0, name='carYOLO26v2')
# Thank you AndreaPi on github for answering your own question on github: https://github.com/ultralytics/ultralytics/issues/3527