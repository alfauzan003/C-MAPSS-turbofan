"""ORM model for the engineered training-set rows.

Only persists *target + identity*. The actual feature columns are wide and
volatile — we materialize them on demand from raw_sensor.readings using
pdm.features. This table records (engine, cycle, training_run, rul) so
runs can reference exactly which rows they trained on.
"""

from datetime import datetime

from sqlalchemy import BigInteger, DateTime, Integer, String, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column

from pdm.db import Base


class EngineWindow(Base):
    __tablename__ = "engine_window"
    __table_args__ = (
        UniqueConstraint(
            "training_run_id", "engine_id", "cycle",
            name="uq_engine_window_run_engine_cycle",
        ),
        {"schema": "features"},
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    training_run_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    engine_id: Mapped[int] = mapped_column(Integer, nullable=False)
    cycle: Mapped[int] = mapped_column(Integer, nullable=False)
    rul: Mapped[int] = mapped_column(Integer, nullable=False)
    parquet_uri: Mapped[str] = mapped_column(String(512), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
