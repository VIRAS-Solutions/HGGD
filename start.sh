#!/bin/bash

# Image name to start
IMAGE_NAME="hggd"

# Docker run command to start container
docker run -it --network="host" -v="$(pwd):/workspace/models/HGGD" --runtime=nvidia $IMAGE_NAME:latest /bin/bash
