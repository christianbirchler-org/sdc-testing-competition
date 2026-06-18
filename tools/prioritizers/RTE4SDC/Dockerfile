FROM python:3.13
WORKDIR /app
RUN pip install grpcio-tools onnxruntime numpy
COPY . .
ENTRYPOINT ["python", "main.py"]