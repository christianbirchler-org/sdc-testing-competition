FROM python:3.13

WORKDIR /app

RUN pip install grpcio-tools onnxruntime==1.24.2 numpy scipy

COPY . .

ENTRYPOINT ["python", "eagle_semble.py"]
