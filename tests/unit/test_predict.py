"""Unit tests for PredictService — uses a fake model to avoid MLflow."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from pdm.predict import PredictService, _fingerprint, _readings_to_df
from pdm.schemas import PredictReadingRow


@dataclass
class _FakeModel:
    """Returns 50.0 for every input row, deterministically."""

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        # Return one prediction per row; service uses the last one.
        return np.full(len(X), 50.0)


def _row(cycle: int = 1) -> PredictReadingRow:
    base = {
        "engine_id": 1,
        "cycle": cycle,
        "op_setting_1": 0.0,
        "op_setting_2": 0.0,
        "op_setting_3": 100.0,
    }
    for i in range(1, 22):
        base[f"sensor_{i}"] = float(i)
    return PredictReadingRow(**base)


def test_readings_to_df_preserves_order():
    rows = [_row(cycle=c) for c in [1, 2, 3]]
    df = _readings_to_df(rows)
    assert list(df["cycle"]) == [1, 2, 3]
    assert df.shape[0] == 3


def test_fingerprint_is_deterministic_and_input_dependent():
    rows1 = [_row(cycle=c) for c in [1, 2, 3]]
    rows2 = [_row(cycle=c) for c in [1, 2, 3]]
    rows3 = [_row(cycle=c) for c in [1, 2, 4]]
    assert _fingerprint(rows1) == _fingerprint(rows2)
    assert _fingerprint(rows1) != _fingerprint(rows3)


def test_predict_service_returns_last_window_prediction():
    svc = PredictService(model=_FakeModel(), model_name="fake", model_version="1")
    rows = [_row(cycle=c) for c in range(1, 11)]
    out = svc.predict(rows)
    assert out.predicted_rul == 50.0
    assert out.model_name == "fake"
    assert out.model_version == "1"
    assert out.n_input_rows == 10
    assert out.latency_ms >= 0


def test_predict_service_rejects_empty_rows():
    svc = PredictService(model=_FakeModel(), model_name="fake", model_version="1")
    with pytest.raises(ValueError, match="non-empty"):
        svc.predict([])


def test_predict_service_rejects_mixed_engine_ids():
    svc = PredictService(model=_FakeModel(), model_name="fake", model_version="1")
    row1 = _row(cycle=1)
    # Create a row with engine_id=2
    base = {
        "engine_id": 2,
        "cycle": 2,
        "op_setting_1": 0.0,
        "op_setting_2": 0.0,
        "op_setting_3": 100.0,
    }
    for i in range(1, 22):
        base[f"sensor_{i}"] = float(i)
    row2 = PredictReadingRow(**base)
    with pytest.raises(ValueError, match="engine_id"):
        svc.predict([row1, row2])
