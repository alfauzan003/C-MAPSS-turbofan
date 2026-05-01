FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

RUN pip install --upgrade pip && pip install \
    "mlflow>=3,<4" \
    "psycopg2-binary>=2.9,<3" \
    "boto3>=1.34,<2"

EXPOSE 5000

# Tracking server is started via the docker-compose `command:` field.
