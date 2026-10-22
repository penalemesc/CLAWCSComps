# Import dependencies
from ultralytics import YOLO

# Define model that we are using as a base
model = YOLO("yolo26m.pt")
 
# Define training parameters, with the device being mps because CPL (creator of this training) is using an apple silicon device for training
results = model.train(data = 'SharedGT/yolo/car.yaml', imgsz = 640, epochs = 25, device=0, name='carYOLO26')