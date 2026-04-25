"""Tests for compute_rul — derives RUL from run-to-failure traces."""

import pandas as pd

from pdm.features.rul import compute_rul


def test_rul_for_single_engine():
    df = pd.DataFrame({"engine_id": [1, 1, 1, 1], "cycle": [1, 2, 3, 4]})
    out = compute_rul(df)
    # max cycle = 4; RUL = 4 - cycle
    assert list(out["rul"]) == [3, 2, 1, 0]


def test_rul_handles_multiple_engines():
    df = pd.DataFrame(
        {"engine_id": [1, 1, 2, 2, 2], "cycle": [1, 2, 1, 2, 3]}
    )
    out = compute_rul(df)
    assert list(out["rul"]) == [1, 0, 2, 1, 0]


def test_rul_caps_at_max():
    df = pd.DataFrame({"engine_id": [1] * 5, "cycle": [1, 2, 3, 4, 5]})
    out = compute_rul(df, max_rul=2)
    assert list(out["rul"]) == [2, 2, 2, 1, 0]


def test_rul_preserves_other_columns():
    df = pd.DataFrame(
        {"engine_id": [1, 1], "cycle": [1, 2], "sensor_1": [10.0, 11.0]}
    )
    out = compute_rul(df)
    assert "sensor_1" in out.columns
    assert list(out["sensor_1"]) == [10.0, 11.0]


def test_rul_empty_input_returns_empty():
    df = pd.DataFrame({"engine_id": [], "cycle": []})
    out = compute_rul(df)
    assert "rul" in out.columns
    assert len(out) == 0
