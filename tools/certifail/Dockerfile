FROM python:3.11

WORKDIR /app

RUN pip install grpcio-tools pandas joblib scikit-learn xgboost

COPY . .

ENTRYPOINT ["python", "CertiFail.py"]
