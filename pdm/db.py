"""SQLAlchemy 2.x engine + session factory + declarative Base.

Other modules:
    from pdm.db import Base, get_engine, get_sessionmaker
    Session = get_sessionmaker()
    with Session() as session:
        ...
"""

from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from pdm.config import get_settings


class Base(DeclarativeBase):
    """Declarative base for all ORM models in the pdm package."""


def build_engine(url: str) -> Engine:
    """Create a SQLAlchemy engine for the given URL."""
    return create_engine(url, pool_pre_ping=True, future=True)


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    """Return the process-wide engine, lazily built from settings."""
    return build_engine(get_settings().database_url)


@lru_cache(maxsize=1)
def get_sessionmaker() -> sessionmaker[Session]:
    """Return the process-wide sessionmaker."""
    return sessionmaker(bind=get_engine(), expire_on_commit=False, future=True)


@contextmanager
def session_scope() -> Iterator[Session]:
    """Context manager that commits on success, rolls back on exception."""
    Session_ = get_sessionmaker()
    session = Session_()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
