"""Tests for pdm.db — engine factory + Base declarative base."""

from sqlalchemy import Engine

from pdm.db import Base, build_engine


def test_build_engine_uses_database_url():
    engine = build_engine("postgresql+psycopg://u:p@h:5432/d")
    assert isinstance(engine, Engine)
    assert str(engine.url) == "postgresql+psycopg://u:***@h:5432/d"


def test_base_is_declarative_base():
    # Has the SQLAlchemy 2.x DeclarativeBase metadata
    assert hasattr(Base, "metadata")
    assert hasattr(Base, "registry")
