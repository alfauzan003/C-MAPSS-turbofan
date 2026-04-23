"""Settings — single source of truth for runtime configuration.

All env vars are loaded here. Other modules import `get_settings()`,
never `os.environ` directly. Cached via lru_cache so the env is read once.
"""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Postgres
    postgres_host: str = Field(...)
    postgres_port: int = Field(default=5432)
    postgres_user: str = Field(...)
    postgres_password: str = Field(...)
    postgres_db: str = Field(...)

    # MinIO / S3
    minio_endpoint_url: str = Field(...)
    minio_access_key: str = Field(...)
    minio_secret_key: str = Field(...)
    minio_bucket_raw: str = Field(default="raw-data")
    minio_bucket_artifacts: str = Field(default="mlflow-artifacts")

    # Logging
    log_level: str = Field(default="INFO")

    @property
    def database_url(self) -> str:
        return (
            f"postgresql+psycopg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings instance. Env is read once per process."""
    return Settings()
