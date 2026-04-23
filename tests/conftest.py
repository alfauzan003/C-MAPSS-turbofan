"""Shared pytest fixtures.

Integration tests assume `docker compose up -d postgres minio` is running.
The session-scoped `db_engine` fixture connects to the `pdm_test` database
(created by `scripts/postgres-init/01-create-test-db.sql`), drops & recreates
the public schema at session start, runs migrations, and yields an engine.
"""

import os
from collections.abc import Iterator

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, text
from sqlalchemy.orm import Session

from pdm.config import Settings, get_settings
from pdm.db import build_engine, get_engine, get_sessionmaker


def _test_settings() -> Settings:
    """Settings pointed at the pdm_test database.

    Uses env vars from the dev shell (e.g. POSTGRES_HOST=localhost when running
    pytest on the host against a docker-compose-exposed port).
    """
    # Force test DB regardless of what .env says
    os.environ["POSTGRES_DB"] = "pdm_test"
    # Default to localhost when running on the host; CI/devcontainer can override
    os.environ.setdefault("POSTGRES_HOST", "localhost")
    os.environ.setdefault("POSTGRES_PORT", "5432")
    os.environ.setdefault("POSTGRES_USER", "pdm")
    os.environ.setdefault("POSTGRES_PASSWORD", "pdm")
    os.environ.setdefault("MINIO_ENDPOINT_URL", "http://localhost:9000")
    os.environ.setdefault("MINIO_ACCESS_KEY", "minioadmin")
    os.environ.setdefault("MINIO_SECRET_KEY", "minioadmin")
    # Reset the cached settings/engine so they pick up the override
    get_settings.cache_clear()
    get_engine.cache_clear()
    get_sessionmaker.cache_clear()
    return get_settings()


@pytest.fixture(scope="session")
def db_engine() -> Iterator[Engine]:
    """Session-scoped engine pointed at pdm_test, with migrations applied.

    Drops and recreates the public schema at session start so each test session
    starts clean. We don't drop it at teardown (keeping it lets you inspect
    state after a failed run).
    """
    settings = _test_settings()
    engine = build_engine(settings.database_url)

    # Reset schema
    with engine.begin() as conn:
        conn.execute(text("DROP SCHEMA IF EXISTS public CASCADE"))
        conn.execute(text("CREATE SCHEMA public"))

    # Apply migrations
    cfg = Config("alembic.ini")
    cfg.set_main_option("sqlalchemy.url", settings.database_url)
    command.upgrade(cfg, "head")

    yield engine

    engine.dispose()


@pytest.fixture()
def db_session(db_engine: Engine) -> Iterator[Session]:
    """Per-test session wrapped in a transaction that rolls back at teardown."""
    connection = db_engine.connect()
    transaction = connection.begin()
    session = Session(bind=connection, expire_on_commit=False)
    try:
        yield session
    finally:
        session.close()
        transaction.rollback()
        connection.close()
