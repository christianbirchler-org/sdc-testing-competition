FROM python:3.11

WORKDIR /app

RUN pip install grpcio-tools numpy Shapely

COPY . .

ENTRYPOINT ["python", "evaluation.py", "-t", "./sample_tests/sdc-test-data.json"]
