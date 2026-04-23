"""Tests for pdm.schemas — Pydantic wire-format models."""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from pdm.schemas import HealthStatus, SensorReadingIn, SensorReadingOut


def _payload(**overrides) -> dict:
    base = {
        "engine_id": 1,
        "cycle": 1,
        "op_setting_1": -0.0007,
        "op_setting_2": -0.0004,
        "op_setting_3": 100.0,
    }
    for i in range(1, 22):
        base[f"sensor_{i}"] = float(i)
    base.update(overrides)
    return base


def test_sensor_reading_in_accepts_full_payload():
    p = _payload()
    m = SensorReadingIn(**p)
    assert m.engine_id == 1
    assert m.cycle == 1
    assert m.sensor_21 == 21.0


def test_sensor_reading_in_rejects_missing_sensor():
    p = _payload()
    del p["sensor_5"]
    with pytest.raises(ValidationError) as exc:
        SensorReadingIn(**p)
    assert "sensor_5" in str(exc.value)


def test_sensor_reading_in_rejects_negative_engine_id():
    p = _payload(engine_id=0)
    with pytest.raises(ValidationError):
        SensorReadingIn(**p)


def test_sensor_reading_in_rejects_negative_cycle():
    p = _payload(cycle=0)
    with pytest.raises(ValidationError):
        SensorReadingIn(**p)


def test_sensor_reading_in_optional_ts_defaults_to_none():
    p = _payload()
    m = SensorReadingIn(**p)
    assert m.ts is None


def test_sensor_reading_in_accepts_iso_ts():
    p = _payload(ts="2026-04-22T12:00:00+00:00")
    m = SensorReadingIn(**p)
    assert m.ts == datetime(2026, 4, 22, 12, 0, tzinfo=timezone.utc)


def test_sensor_reading_out_serializes():
    out = SensorReadingOut(id=42, engine_id=1, cycle=1)
    j = out.model_dump()
    assert j == {"id": 42, "engine_id": 1, "cycle": 1}


def test_health_status_ok():
    h = HealthStatus(status="ok")
    assert h.status == "ok"
