#!/bin/bash

# Image name
IMAGE_NAME="hggd"

# Docker build command
docker build --network=host -t $IMAGE_NAME .
