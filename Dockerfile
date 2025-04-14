# Base Image (Ubuntu 20.04)
FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04

# Umgebungsvariablen setzen
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8

RUN apt update -y
# Systemabhängigkeiten und Python installieren
RUN apt-get update && apt-get install -y \
    python3.8 \
    python3.8-dev \
    python3-pip \
    python3.8-venv \
    git \
    build-essential \
    cmake \
    curl \
    ca-certificates \
    libgl1-mesa-glx \
    libsm6 \
    libxext6 \
    libxrender-dev \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Python3.8 als Standard-Python einrichten
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1

# pip auf die neueste Version aktualisieren
RUN python3 -m pip install --upgrade pip

# Abhängigkeiten installieren
WORKDIR /workspace

# Kopiere requirements.txt aus deinem Repo (unter src/models/HGGD)
#COPY src/models/HGGD/requirements.txt .
COPY requirements.txt .

RUN pip install -r requirements.txt


# Installiere pytorch und pytorch3d manuell
RUN pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 \
    --extra-index-url https://download.pytorch.org/whl/cu113

RUN pip install fvcore
RUN pip install --no-index --no-cache-dir pytorch3d \
    -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu113_pyt1110/download.html

# Setze das Arbeitsverzeichnis auf das HGGD-Projektverzeichnis
WORKDIR /workspace/models/HGGD

# Update
RUN apt-get update 

# 'python' program command execute Python 3
RUN apt install  python-is-python3

# Libglib2.0 install
RUN apt-get update && apt-get install -y libglib2.0-0


# Definiere Standardkommando, wenn der Container startet
#CMD ["bash", "demo.sh"]
CMD ["bash"]
