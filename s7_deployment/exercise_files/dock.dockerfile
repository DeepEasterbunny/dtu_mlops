FROM python:3.11-slim

EXPOSE 80

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install fastapi
RUN pip install pydantic
RUN pip install uvicorn
RUN pip install opencv-python

COPY main.py main.py
COPY mail.json mail.json

CMD exec uvicorn main:app --port 80 --host 0.0.0.0 --workers 1