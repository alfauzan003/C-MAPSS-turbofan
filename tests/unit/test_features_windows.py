"""Tests for compute_windows — rolling stats + lag features per engine."""

import pandas as pd

from pdm.features.windows import compute_windows


def _frame(engine_id: int, cycles: list[int], sensor_1: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "engine_id": [engine_id] * len(cycles),
            "cycle": cycles,
            "sensor_1": sensor_1,
        }
    )


def test_compute_windows_adds_rolling_mean():
    df = _frame(1, cycles=[1, 2, 3, 4, 5], sensor_1=[10.0, 12.0, 14.0, 16.0, 18.0])
    out = compute_windows(df, sensor_cols=["sensor_1"], windows=(3,), lags=())
    assert "sensor_1_roll3_mean" in out.columns
    # Rolling mean(3) at cycle 3 = mean(10,12,14) = 12.0
    row = out[out["cycle"] == 3].iloc[0]
    assert row["sensor_1_roll3_mean"] == 12.0


def test_compute_windows_adds_rolling_std():
    df = _frame(1, cycles=[1, 2, 3, 4], sensor_1=[1.0, 2.0, 3.0, 4.0])
    out = compute_windows(df, sensor_cols=["sensor_1"], windows=(3,), lags=())
    # Rolling std(3) at cycle 3 = std(1,2,3) = 1.0 (sample std)
    row = out[out["cycle"] == 3].iloc[0]
    assert round(row["sensor_1_roll3_std"], 5) == 1.0


def test_compute_windows_adds_lag():
    df = _frame(1, cycles=[1, 2, 3], sensor_1=[10.0, 20.0, 30.0])
    out = compute_windows(df, sensor_cols=["sensor_1"], windows=(), lags=(1,))
    row = out[out["cycle"] == 2].iloc[0]
    assert row["sensor_1_lag1"] == 10.0


def test_compute_windows_per_engine_no_cross_contamination():
    """Engine 2's lag at cycle 1 must NOT see engine 1's data."""
    df = pd.concat(
        [
            _frame(1, cycles=[1, 2], sensor_1=[100.0, 200.0]),
            _frame(2, cycles=[1, 2], sensor_1=[1.0, 2.0]),
        ],
        ignore_index=True,
    )
    out = compute_windows(df, sensor_cols=["sensor_1"], windows=(), lags=(1,))
    e2_c1 = out[(out["engine_id"] == 2) & (out["cycle"] == 1)].iloc[0]
    # Lag1 of first cycle is undefined (NaN) — must NOT leak engine 1's value
    assert pd.isna(e2_c1["sensor_1_lag1"])


def test_compute_windows_adds_time_since_start():
    df = _frame(1, cycles=[1, 5, 10], sensor_1=[0.0, 0.0, 0.0])
    out = compute_windows(df, sensor_cols=["sensor_1"], windows=(), lags=())
    assert "time_since_start" in out.columns
    assert list(out["time_since_start"]) == [0, 4, 9]


def test_compute_windows_preserves_input_columns():
    df = _frame(1, cycles=[1, 2], sensor_1=[1.0, 2.0])
    df["op_setting_1"] = [0.1, 0.2]
    out = compute_windows(df, sensor_cols=["sensor_1"], windows=(2,), lags=(1,))
    assert "op_setting_1" in out.columns
    assert "engine_id" in out.columns
    assert "cycle" in out.columns


def test_compute_windows_empty_input():
    df = pd.DataFrame({"engine_id": [], "cycle": [], "sensor_1": []})
    out = compute_windows(df, sensor_cols=["sensor_1"], windows=(3,), lags=(1,))
    assert "sensor_1_roll3_mean" in out.columns
    assert "sensor_1_lag1" in out.columns
    assert len(out) == 0
