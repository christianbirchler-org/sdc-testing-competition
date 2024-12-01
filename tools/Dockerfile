FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    grpcio \
    grpcio-tools \
    numpy \
    shapely \
    scikit-learn \
    matplotlib

COPY . .

ENTRYPOINT ["python", "curvature_selector.py"]
