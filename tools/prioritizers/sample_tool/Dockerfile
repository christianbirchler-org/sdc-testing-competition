FROM python:3.13

WORKDIR /app

RUN pip install grpcio-tools

COPY . .

ENTRYPOINT ["python", "sample_test_prioritizer.py"]
