# How to run training script

docker build --tag 'sam3' .

docker run -it --rm --gpus all --shm-size=8g \
  --volume="$PWD:/app" \
  --volume="/home/steinsmithw/Desktop/CLAWCSComps/dataset:/dataset" \
  sam3 python finetune_sam3_native.py
