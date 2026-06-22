# Use a lightweight Python image with PyTorch support
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Install necessary system libraries for NumPy/gRPC
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install the exact libraries you need
RUN pip install --no-cache-dir \
    torch \
    numpy \
    grpcio \
    grpcio-tools

# Copy only the necessary "Winning" files
COPY main.py .
COPY crash_model.pth .
COPY competition_2026_pb2.py .
COPY competition_2026_pb2_grpc.py .

# Expose the gRPC port
EXPOSE 50051

# Run your tool
CMD ["python", "main.py"]
