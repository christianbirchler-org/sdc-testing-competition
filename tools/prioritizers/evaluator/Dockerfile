FROM python:3.13

WORKDIR /app


COPY . .

RUN pip install -r requirements.txt

ENTRYPOINT ["python", "evaluation.py"]
