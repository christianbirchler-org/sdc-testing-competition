FROM python:3.11

WORKDIR /app

RUN pip install grpcio-tools

COPY . .

ENTRYPOINT ["python", "evaluation.py"]
