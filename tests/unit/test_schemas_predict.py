"""Tests for predict-endpoint schemas."""

import pytest
from pydantic import ValidationError

from pdm.schemas import PredictReadingRow, PredictRequest


def _row(engine_id: int = 1, cycle: int = 1) -> dict:
    base = {
        "engine_id": engine_id,
        "cycle": cycle,
        "op_setting_1": 0.0,
        "op_setting_2": 0.0,
        "op_setting_3": 100.0,
    }
    for i in range(1, 22):
        base[f"sensor_{i}"] = float(i)
    return base


def test_predict_request_accepts_single_row():
    req = PredictRequest(readings=[_row()])
    assert len(req.readings) == 1


def test_predict_request_accepts_window():
    req = PredictRequest(readings=[_row(cycle=c) for c in range(1, 11)])
    assert len(req.readings) == 10


def test_predict_request_rejects_empty():
    with pytest.raises(ValidationError):
        PredictRequest(readings=[])


def test_predict_request_rejects_too_long():
    rows = [_row(cycle=c) for c in range(1, 502)]
    with pytest.raises(ValidationError):
        PredictRequest(readings=rows)
