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


