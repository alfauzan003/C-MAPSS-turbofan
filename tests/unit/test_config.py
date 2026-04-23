"""Tests for pdm.config — Settings class loads env vars correctly."""

import os
from unittest.mock import patch

from pdm.config import Settings


def test_settings_load_from_env():
    env = {
        "POSTGRES_HOST": "db.example",
        "POSTGRES_PORT": "5433",
        "POSTGRES_USER": "u",
        "POSTGRES_PASSWORD": "p",
        "POSTGRES_DB": "d",
        "MINIO_ENDPOINT_URL": "http://minio.example:9000",
        "MINIO_ACCESS_KEY": "ak",
        "MINIO_SECRET_KEY": "sk",
        "MINIO_BUCKET_RAW": "raw-data",
        "MINIO_BUCKET_ARTIFACTS": "mlflow-artifacts",
        "LOG_LEVEL": "DEBUG",
    }
    with patch.dict(os.environ, env, clear=True):
        s = Settings()

    assert s.postgres_host == "db.example"
    assert s.postgres_port == 5433
    assert s.postgres_user == "u"
    assert s.postgres_db == "d"
    assert s.minio_endpoint_url == "http://minio.example:9000"
    assert s.minio_bucket_raw == "raw-data"
    assert s.log_level == "DEBUG"


def test_settings_database_url_property():
    env = {
        "POSTGRES_HOST": "h",
        "POSTGRES_PORT": "5432",
        "POSTGRES_USER": "u",
        "POSTGRES_PASSWORD": "p",
        "POSTGRES_DB": "d",
        "MINIO_ENDPOINT_URL": "http://m:9000",
        "MINIO_ACCESS_KEY": "ak",
        "MINIO_SECRET_KEY": "sk",
        "MINIO_BUCKET_RAW": "raw",
        "MINIO_BUCKET_ARTIFACTS": "art",
    }
    with patch.dict(os.environ, env, clear=True):
        s = Settings()
    assert s.database_url == "postgresql+psycopg://u:p@h:5432/d"
