# SAM3 Fine-Tuning Implementation

## Overview

This directory contains the code for fine-tuning SAM3. My implementation fine-tunes SAM3 specifically to detect cars given the prompt "car" using COCO-format annotations. The training pipeline runs in a Docker container with GPU support and uses PyTorch for model training.

## Files

### `fine-tune.py`
This is the most important file here, as it runs the entire fine-tuning process for SAM3. It loads the pre-trained SAM3 model, configures optimizer settings, and trains the model on car detection using COCO annotations. It also tracks validation metrics on a development set and automatically saves the best checkpoint based on bounding box AP.

### `config.py`
Contains all configuration constants and hyperparameters for training, including model architecture settings, paths to datasets and weights, and training hyperparameters (learning rates, epochs, batch size, etc). This centralized configuration file allows easy modification of training settings without editing the main training script.

### `logger.py`
A custom logging class that records training progress and metrics to both human-readable text files and JSON. The logger includes timestamps for each log entry and handles structured metric recording for analysis and visualization of training progress.

### `Dockerfile`
Specifies the Docker container environment for training, including all system dependencies and Python packages. 

## Running the Training Script

```bash
docker build --tag 'sam3' .

docker run -it --rm --gpus all --shm-size=8g \
  --volume="$PWD:/app" \
  --volume="/path_to_data_set:/dataset" \
  sam3 python fine-tune.py
```
