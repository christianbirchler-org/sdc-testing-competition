FROM nvidia/cuda:12.2.0-base-ubuntu22.04

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive 
ENV PYTHONUNBUFFERED=1
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-dev \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*


RUN pip3 install --no-cache-dir \
    torch==2.1.0+cu121 --extra-index-url https://download.pytorch.org/whl/cu121 \
    torch-scatter torch-sparse torch-geometric -f https://data.pyg.org/whl/torch-2.1.0+cu121.html \
    numpy==1.24.3 \
    networkx==3.1 \
    scipy==1.10.1 \
    grpcio==1.67.1 \
    grpcio-tools==1.67.1 \
    scikit-learn==1.3.0

COPY . .

ENTRYPOINT ["python3", "graph_selector.py"]