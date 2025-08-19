FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3-dev \
    gcc \
    g++ \
    && rm -rf /var/apt/lists/*

RUN pip install --no-cache-dir \
    numpy \
    scikit-learn \
    protobuf \
    grpcio \
    grpcio-tools

COPY . .

ENTRYPOINT ["python", "ensemble_selector.py"]