FROM nvidia/cuda:12.6.0-cudnn-devel-ubuntu22.04
ARG DEBIAN_FRONTEND=noninteractive

WORKDIR /app

#set up environment
RUN apt-get update && apt-get install --no-install-recommends --no-install-suggests -y curl wget software-properties-common && add-apt-repository ppa:deadsnakes/ppa
# RUN apt-get update
RUN apt-get -y install unzip libpng-dev zlib1g-dev nano
RUN apt-get -y install python3.11-dev
RUN  curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

# Install the required packages
RUN python3.11 -m pip install --upgrade pip && \
    pip install tensorflow[and-cuda]==2.18.0 && \
    pip install pandas==2.2.3 && \
    pip install numpy==2.0.2 && \
    pip install matplotlib==3.9.2 && \
    pip install shapely==2.0.6 && \
    pip install scikit-learn==1.5.2 && \
    pip install scipy==1.14.1 && \
    pip install tqdm==4.67.1 && \
    pip install grpcio-tools

COPY . .

ENTRYPOINT ["python3.11", "drvn_test_selector.py"]
