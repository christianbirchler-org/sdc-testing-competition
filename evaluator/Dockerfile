FROM python:3.11

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

ENTRYPOINT ["python", "evaluation.py", "-t", "./sample_tests/sdc-test-data.json"]
