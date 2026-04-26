"""ORM models for the predictions schema.

`served`: one row per prediction returned to a client.
`drift_reports`: one row per drift-detection window evaluation.
"""
from datetime import datetime

from sqlalchemy import BigInteger, Boolean, DateTime, Float, Integer, JSON, String, func
from sqlalchemy.orm import Mapped, mapped_column

from pdm.db import Base


class ServedPrediction(Base):
    __tablename__ = "served"
    __table_args__ = ({"schema": "predictions"},)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)

    # What was predicted
    engine_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    predicted_rul: Mapped[float] = mapped_column(Float, nullable=False)

    # Identity / lineage
    model_name: Mapped[str] = mapped_column(String(64), nullable=False)
    model_version: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    input_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    n_input_rows: Mapped[int] = mapped_column(Integer, nullable=False)

    # Quality / ops
    latency_ms: Mapped[float] = mapped_column(Float, nullable=False)
    served_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), index=True
    )


class DriftReport(Base):
    __tablename__ = "drift_reports"
    __table_args__ = ({"schema": "predictions"},)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)

    model_name: Mapped[str] = mapped_column(String(64), nullable=False)
    model_version: Mapped[str] = mapped_column(String(32), nullable=False, index=True)

    window_start: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    window_end: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    n_baseline_rows: Mapped[int] = mapped_column(Integer, nullable=False)
    n_compare_rows: Mapped[int] = mapped_column(Integer, nullable=False)

    # {"sensor_3": 0.12, "sensor_7": 0.31, ...}
    psi_per_feature: Mapped[dict] = mapped_column(JSON, nullable=False)
    max_psi: Mapped[float] = mapped_column(Float, nullable=False, index=True)
    alert: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, index=True
    )
