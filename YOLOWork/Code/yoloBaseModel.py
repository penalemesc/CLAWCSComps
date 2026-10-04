# This has a base implementation of the YOLO model. 
# All of the code present here (at the time of 29/9/26 (d/m/y))is created by CPL based on his previous research, unless otherwise specified

# Import dependencies
import cv2
import os
from ultralytics import YOLO

# Define the output path
output = "YOLOWork/Code/videoRecordings"

# Define camera object

# Define yolo model so that we can use it in predictions

# Define the video recording function -- IMPLEMENT LATER

# White we do not quit the program, keep reading and showing the screen
while True:
    # I forgot that I haven't used an actual camera from a laptop in like a year, which means that I need to figure out how to do that in a non-pi setting
    # videoStream = 

    # Show our preview 
    cv2.imshow()

    # Record user input
    keyPressed = cv2.waitKey(0)
    
    if keyPressed == ord('q'):
        # If we pressed q, quit
        break
    # IMPLEMENT SAVING VIDEOS AT SOME POINT
    # elif keyPressed == ord('s'):


cv2.destroyAllWindows()

#  Working with waymo dataset notes
#  validate on pc - gcloud auth application-default login
#  TensorFlow's tf.data.TFRecordDataset can read files directly from a Google Cloud Storage path prefix (gs://). The file stays in the cloud, and data streams frame-by-frame directly into your system memory

# Code for tensorflow
# import tensorflow as tf
# from waymo_open_dataset import dataset_pb2 as open_dataset

# # The official Waymo bucket path (ensure your account has access)
# # Replace with the exact split/shard path you need
# # Luke will find the bucket we can use.
# gcs_file_path = "gs://waymo_open_dataset_motion/v_1_2_0/uncompressed/scenario/training/training.tfrecord-00000-of-01000"

# # 1. Create a dataset that streams directly from GCS
# # buffer_size configures a prefetch buffer to keep the stream moving smoothly
# raw_dataset = tf.data.TFRecordDataset(gcs_file_path, buffer_size=100 * 1024 * 1024) # 100MB buffer

# def parse_waymo_scenario(raw_record):
#     """Parses a single raw scenario record without downloading the file."""
#     # Create the internal protocol buffer object
#     scenario = open_dataset.Scenario()
#     scenario.ParseFromString(raw_record.numpy())
    
#     # Example extraction: Extract object trajectories for motion/tracking
#     # If training YOLO/SAM, extract camera image bytes here instead
#     tracks = scenario.tracks
#     map_features = scenario.map_features
    
#     return tracks, map_features

# # 2. Iterate through the stream on-the-fly
# print("Starting stream...")
# for raw_record in raw_dataset.take(5):  # Grabbing 5 scenarios as an example
#     tracks, map_features = parse_waymo_scenario(raw_record)
#     print(f"Successfully streamed scenario with {len(tracks)} tracked agents!")


# Code for pytorch
# import torch
# from torch.utils.data import IterableDataset, DataLoader

# class WaymoStreamDataset(IterableDataset):
#     def __init__(self, tf_dataset):
#         self.tf_dataset = tf_dataset

#     def __iter__(self):
#         for raw_record in self.tf_dataset:
#             # Parse your data, convert to numpy/tensors
#             # For YOLO: extract image, resize, find bounding boxes
#             # For SAM: extract image and prompt points/masks
            
#             # Yield image tensor and target annotations 
#             yield image_tensor, target_tensor

# # Wrap the streamed TF dataset into PyTorch
# torch_stream = WaymoStreamDataset(raw_dataset)
# train_loader = DataLoader(torch_stream, batch_size=4)

# # Your typical training loop
# for batch_idx, (images, targets) in enumerate(train_loader):
#     # outputs = model(images)
#     # loss = criterion(outputs, targets)
#     pass
