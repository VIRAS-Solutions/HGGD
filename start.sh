#!/bin/bash

# Image name to start
IMAGE_NAME="hggd"

xhost +local:docker


# Docker run command to start container
#docker run -it --network="host" -v="$(pwd):/workspace/models/HGGD"  --runtime=nvidia --gpus=all $IMAGE_NAME:latest /bin/bash

docker run -it \
    --env="DISPLAY=$DISPLAY" \
    --env="QT_X11_NO_MITSHM=1" \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    --network="host" \
    --volume="$(pwd):/workspace/models/HGGD" \
    --runtime=nvidia \
    --gpus=all \
    $IMAGE_NAME:latest /bin/bash


xhost -local:docker
