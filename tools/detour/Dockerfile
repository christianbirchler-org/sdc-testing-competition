FROM python:3.11

WORKDIR /app

RUN pip install grpcio-tools
RUN pip install numpy
RUN pip install scipy

COPY . .

ENTRYPOINT ["python", "detour.py"]
