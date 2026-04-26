"""Pydantic models for the API wire format.

These mirror the C-MAPSS row layout but are HTTP-side only — they do not import
SQLAlchemy. The API layer translates between SensorReadingIn and the ORM model.
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, PositiveInt


class SensorReadingIn(BaseModel):
    """Single sensor row posted by the simulator (or any client)."""

    engine_id: PositiveInt
    cycle: PositiveInt

    op_setting_1: float
    op_setting_2: float
    op_setting_3: float

    sensor_1: float
    sensor_2: float
    sensor_3: float
    sensor_4: float
    sensor_5: float
    sensor_6: float
    sensor_7: float
    sensor_8: float
    sensor_9: float
    sensor_10: float
    sensor_11: float
    sensor_12: float
    sensor_13: float
    sensor_14: float
    sensor_15: float
    sensor_16: float
    sensor_17: float
    sensor_18: float
    sensor_19: float
    sensor_20: float
    sensor_21: float

    # If client provides a ts we use it; otherwise the DB server_default fills now().
    ts: datetime | None = Field(default=None)


class SensorReadingOut(BaseModel):
    """Acknowledgment returned by the API after a successful insert."""

    id: int
    engine_id: int
    cycle: int


class HealthStatus(BaseModel):
    status: Literal["ok", "degraded", "down"]
    detail: str | None = None


class PredictReadingRow(BaseModel):
    """One row in a predict request — same shape as SensorReadingIn but no ts."""

    engine_id: PositiveInt
    cycle: PositiveInt

    op_setting_1: float
    op_setting_2: float
    op_setting_3: float

    sensor_1: float
    sensor_2: float
    sensor_3: float
    sensor_4: float
    sensor_5: float
    sensor_6: float
    sensor_7: float
    sensor_8: float
    sensor_9: float
    sensor_10: float
    sensor_11: float
    sensor_12: float
    sensor_13: float
    sensor_14: float
    sensor_15: float
    sensor_16: float
    sensor_17: float
    sensor_18: float
    sensor_19: float
    sensor_20: float
    sensor_21: float


class PredictRequest(BaseModel):
    """A window of recent readings for a single engine."""

    readings: list[PredictReadingRow] = Field(..., min_length=1, max_length=500)


class PredictResponse(BaseModel):
    """Wire response returned by POST /predict."""

    engine_id: int
    predicted_rul: float
    model_name: str
    model_version: str
    n_input_rows: int
    latency_ms: float
