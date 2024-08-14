FROM python:3.11

WORKDIR /competition

COPY . .

CMD ["python", "evaluation.py"]
