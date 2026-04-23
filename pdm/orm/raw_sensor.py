"""ORM model for raw_sensor.readings — one row per (engine_id, cycle) ingestion event.

Schema = `raw_sensor`. Table = `readings`.
"""

from datetime import datetime

from sqlalchemy import BigInteger, DateTime, Float, Integer, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column

from pdm.db import Base


class SensorReading(Base):
    __tablename__ = "readings"
    __table_args__ = (
        UniqueConstraint("engine_id", "cycle", name="uq_readings_engine_cycle"),
        {"schema": "raw_sensor"},
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)

    # Identity
    engine_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    cycle: Mapped[int] = mapped_column(Integer, nullable=False)

    # Operational settings
    op_setting_1: Mapped[float] = mapped_column(Float, nullable=False)
    op_setting_2: Mapped[float] = mapped_column(Float, nullable=False)
    op_setting_3: Mapped[float] = mapped_column(Float, nullable=False)

    # Sensors 1..21
    sensor_1: Mapped[float] = mapped_column(Float, nullable=False)
    sensor_2: Mapped[float] = mapped_column(Float, nullable=False)
    sensor_3: Mapped[float] = mapped_column(Float, nullable=False)
    sensor_4: Mapped[float] = mapped_column(Float, nullable=False)
    sensor_5: Mapped[float] = mapped_column(Float, nullable=False)
    sensor_6: Mapped[float] = mapped_column(Float, nullable=False)
    sensor_7: Mapped[float] = mapped_column(Float, nullable=False)
    sensor_8: Mapped[float] = mapped_column(Float, nullable=False)
    sensor_9: Mapped[float] = mapped_column(Float, nullable=False)
    sensor_10: Mapped[float] = mapped_column(Float, nullable=False)
    sensor_11: Mapped[float] = mapped_column(Float, nullable=False)
    sensor_12: Mapped[float] = mapped_column(Float, nullable=False)
    sensor_13: Mapped[float] = mapped_column(Float, nullable=False)
    sensor_14: Mapped[float] = mapped_column(Float, nullable=False)
    sensor_15: Mapped[float] = mapped_column(Float, nullable=False)
    sensor_16: Mapped[float] = mapped_column(Float, nullable=False)
    sensor_17: Mapped[float] = mapped_column(Float, nullable=False)
    sensor_18: Mapped[float] = mapped_column(Float, nullable=False)
    sensor_19: Mapped[float] = mapped_column(Float, nullable=False)
    sensor_20: Mapped[float] = mapped_column(Float, nullable=False)
    sensor_21: Mapped[float] = mapped_column(Float, nullable=False)

    # Timestamps
    ts: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), index=True
    )
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
