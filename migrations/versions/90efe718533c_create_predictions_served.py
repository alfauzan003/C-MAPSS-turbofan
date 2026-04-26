"""create predictions served

Revision ID: 90efe718533c
Revises: e3e436e8bb0c
Create Date: 2026-04-26 09:36:49.067212

"""
from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '90efe718533c'
down_revision: str | None = 'e3e436e8bb0c'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("CREATE SCHEMA IF NOT EXISTS predictions")
    op.create_table(
        "served",
        sa.Column("id", sa.BigInteger(), autoincrement=True, primary_key=True),
        sa.Column("engine_id", sa.Integer(), nullable=False),
        sa.Column("predicted_rul", sa.Float(), nullable=False),
        sa.Column("model_name", sa.String(64), nullable=False),
        sa.Column("model_version", sa.String(32), nullable=False),
        sa.Column("input_fingerprint", sa.String(64), nullable=False),
        sa.Column("n_input_rows", sa.Integer(), nullable=False),
        sa.Column("latency_ms", sa.Float(), nullable=False),
        sa.Column(
            "served_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        schema="predictions",
    )
    op.create_index("ix_served_engine_id", "served", ["engine_id"], schema="predictions")
    op.create_index("ix_served_model_version", "served", ["model_version"], schema="predictions")
    op.create_index("ix_served_served_at", "served", ["served_at"], schema="predictions")


def downgrade() -> None:
    op.drop_index("ix_served_served_at", table_name="served", schema="predictions")
    op.drop_index("ix_served_model_version", table_name="served", schema="predictions")
    op.drop_index("ix_served_engine_id", table_name="served", schema="predictions")
    op.drop_table("served", schema="predictions")
    op.execute("DROP SCHEMA IF EXISTS predictions CASCADE")
