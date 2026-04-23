"""Smoke test: alembic migrations apply (and downgrade) cleanly."""

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, inspect

from pdm.config import get_settings


@pytest.mark.integration
def test_alembic_upgrade_head_is_idempotent(db_engine: Engine):
    # db_engine fixture already ran upgrade head. Running again should be a no-op.
    cfg = Config("alembic.ini")
    cfg.set_main_option("sqlalchemy.url", get_settings().database_url)
    command.upgrade(cfg, "head")  # should not raise

    # alembic_version table exists after first run
    insp = inspect(db_engine)
    assert "alembic_version" in insp.get_table_names()


@pytest.mark.integration
def test_alembic_can_downgrade_to_base(db_engine: Engine):
    cfg = Config("alembic.ini")
    cfg.set_main_option("sqlalchemy.url", get_settings().database_url)
    # Phase 0 has no real migrations yet, so downgrade-to-base + upgrade-to-head is trivial.
    command.downgrade(cfg, "base")
    command.upgrade(cfg, "head")
