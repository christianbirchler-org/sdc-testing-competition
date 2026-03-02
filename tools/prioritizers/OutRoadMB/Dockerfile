FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir grpcio>=1.74.0 grpcio-tools>=1.74.0 numpy

COPY . .

ENTRYPOINT ["python", "main.py"]
