FROM python:3.13

WORKDIR /app

# Install PyTorch CPU-only (smaller image, sufficient for inference)
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir \
    grpcio==1.74.0 \
    numpy>=1.26.0 \
    scipy>=1.12.0

COPY . .

ENTRYPOINT ["python", "main.py"]
