# How to run training script

docker build --tag 'sam3' .

docker run -it --rm --gpus all --shm-size=8g \
  --volume="$PWD:/app" \
  --volume="/path_to_data_set:/dataset" \
  sam3 python fine-tune.py
