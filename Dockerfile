# Single image used by simulator, prefect-worker, ingestion-api, prediction-api.
# They differ only by their command at runtime.

FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps (psycopg needs libpq; xgboost needs libgomp)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libpq5 \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Copy project files (pyproject first for better caching once deps are stable)
COPY pyproject.toml ./
COPY pdm/ ./pdm/
COPY migrations/ ./migrations/
COPY alembic.ini ./

# Install the package in editable mode WITH dev extras.
# Editable so host-mounted volumes in dev override the baked source.
RUN pip install --upgrade pip && pip install -e ".[dev]"

# Default to a no-op so docker-compose `command:` overrides decide what runs
CMD ["python", "-c", "print('pdm image: override the command in docker-compose.yml')"]
