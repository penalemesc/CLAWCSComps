# Setup Training Env
## Setup Docker on Local Machine
Cd into this directory
```console
docker build --tag 'sam3' .
```
Run containers
```console
docker run --detach 'sam3' .
docker run --rm sam3
```

### Use GPU?
Run containers with GPU:
```console
docker run --rm --gpus all sam3
```

Test GPU access on lab pc
```console
nvidia-smi
docker run --rm --gpus all sam3 python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"
```
