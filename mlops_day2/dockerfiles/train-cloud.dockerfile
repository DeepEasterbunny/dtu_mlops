# Base image
FROM python:3.11-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY mlops_day2/requirements.txt requirements.txt
COPY mlops_day2/pyproject.toml pyproject.toml
COPY mlops_day2/src/ src/
COPY mlops_day2/reports/ reports/

WORKDIR /
RUN pip install -r requirements.txt --no-cache-dir
RUN pip install . --no-deps --no-cache-dir
RUN mkdir models

ENTRYPOINT ["python", "-u", "src/mlops_day2/train_cloud.py"]
