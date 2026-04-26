"""create predictions drift_reports

Revision ID: 5d70d7c75074
Revises: 90efe718533c
Create Date: 2026-04-26 15:54:35.862930

"""
from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '5d70d7c75074'
down_revision: str | None = '90efe718533c'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "drift_reports",
        sa.Column("id", sa.BigInteger(), autoincrement=True, primary_key=True),
        sa.Column("model_name", sa.String(64), nullable=False),
        sa.Column("model_version", sa.String(32), nullable=False),
        sa.Column("window_start", sa.DateTime(timezone=True), nullable=False),
        sa.Column("window_end", sa.DateTime(timezone=True), nullable=False),
        sa.Column("n_baseline_rows", sa.Integer(), nullable=False),
        sa.Column("n_compare_rows", sa.Integer(), nullable=False),
        sa.Column("psi_per_feature", sa.JSON(), nullable=False),
        sa.Column("max_psi", sa.Float(), nullable=False),
        sa.Column("alert", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        schema="predictions",
    )
    op.create_index(op.f("ix_predictions_drift_reports_model_version"), "drift_reports", ["model_version"], schema="predictions")
    op.create_index(op.f("ix_predictions_drift_reports_max_psi"), "drift_reports", ["max_psi"], schema="predictions")
    op.create_index(op.f("ix_predictions_drift_reports_created_at"), "drift_reports", ["created_at"], schema="predictions")


def downgrade() -> None:
    op.drop_index(op.f("ix_predictions_drift_reports_created_at"), table_name="drift_reports", schema="predictions")
    op.drop_index(op.f("ix_predictions_drift_reports_max_psi"), table_name="drift_reports", schema="predictions")
    op.drop_index(op.f("ix_predictions_drift_reports_model_version"), table_name="drift_reports", schema="predictions")
    op.drop_table("drift_reports", schema="predictions")
