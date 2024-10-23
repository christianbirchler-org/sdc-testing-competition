FROM python:3.11

WORKDIR /app

RUN pip install grpcio-tools pandas joblib scikit-learn

COPY . .

ENTRYPOINT ["python", "rf_static_test_selector.py"]
