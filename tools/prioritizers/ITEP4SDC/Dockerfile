FROM python:3.13

WORKDIR /app

RUN pip install grpcio-tools onnxruntime==1.23.2 numpy scipy joblib scikit-learn

COPY . .

ENTRYPOINT ["python", "main.py"]
