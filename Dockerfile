FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8

# ROS2 Setup
RUN apt update && apt install -y curl gnupg2 lsb-release
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add -
RUN echo "deb http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" > /etc/apt/sources.list.d/ros2-latest.list

RUN apt update && apt install -y \
    ros-galactic-desktop \
    python3-colcon-common-extensions \
    python3-rosdep \
    python3-rosinstall-generator \
    python3-vcstool \
    python3-pip \
    python3.8 \
    python3.8-dev \
    python3.8-venv \
    git \
    build-essential \
    cmake \
    libglib2.0-0 \
    libgl1-mesa-glx \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgtk2.0-dev \
    libusb-1.0-0-dev \
    wget \
    && rm -rf /var/lib/apt/lists/*


# Python 3.8 als Standard setzen
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1
RUN python3 -m pip install --upgrade pip


# Installiere requirements.txt aus dem Modellverzeichnis
WORKDIR /workspace/models/HGGD
# COPY src/models/HGGD/requirements.txt .
COPY requirements.txt .
RUN pip install --no-cache-dir --ignore-installed -r requirements.txt

# 'python' program command execute Python 3
RUN apt-get update 
RUN apt install python-is-python3

# PyTorch + PyTorch3D
RUN pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 \
    --extra-index-url https://download.pytorch.org/whl/cu113
RUN pip install fvcore
RUN pip install --no-index --no-cache-dir pytorch3d \
    -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu113_pyt1110/download.html

# ROS2 Environment konfigurieren
RUN echo "source /opt/ros/galactic/setup.bash" >> /root/.bashrc

# Colcon-Workspace vorbereiten
RUN mkdir -p /ros2/ros2_ws/src

WORKDIR /ros2/ros2_ws

# Portfreigaben für Debugging und Visualisierung
#EXPOSE 8888
#EXPOSE 6006

# Default-Shell
CMD ["bash"]
