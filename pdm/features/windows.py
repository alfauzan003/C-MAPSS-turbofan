"""Rolling-window statistics, lag features, and time-since-start.

All operations are grouped by `engine_id` so per-engine traces don't bleed
into each other.
"""

from collections.abc import Iterable

import pandas as pd

DEFAULT_WINDOWS: tuple[int, ...] = (5, 10, 20)
DEFAULT_LAGS: tuple[int, ...] = (1, 2, 5)


def compute_windows(
    df: pd.DataFrame,
    sensor_cols: Iterable[str],
    windows: tuple[int, ...] = DEFAULT_WINDOWS,
    lags: tuple[int, ...] = DEFAULT_LAGS,
) -> pd.DataFrame:
    """Add rolling-mean, rolling-std, lag, and time-since-start columns.

    For each sensor column and each window size w, adds:
        <sensor>_roll<w>_mean, <sensor>_roll<w>_std

    For each lag k, adds:
        <sensor>_lag<k>

    Always adds `time_since_start` = cycle - min(cycle) per engine.

    Operations are grouped by `engine_id` so windows/lags never leak across engines.
    """
    if df.empty:
        out = df.copy()
        for col in sensor_cols:
            for w in windows:
                out[f"{col}_roll{w}_mean"] = pd.Series(dtype="float64")
                out[f"{col}_roll{w}_std"] = pd.Series(dtype="float64")
            for lag in lags:
                out[f"{col}_lag{lag}"] = pd.Series(dtype="float64")
        out["time_since_start"] = pd.Series(dtype="int64")
        return out

    out = df.copy()
    grouped = out.groupby("engine_id", sort=False, group_keys=False)

    for col in sensor_cols:
        for w in windows:
            out[f"{col}_roll{w}_mean"] = grouped[col].transform(
                lambda s, w=w: s.rolling(window=w, min_periods=1).mean()
            )
            out[f"{col}_roll{w}_std"] = grouped[col].transform(
                lambda s, w=w: s.rolling(window=w, min_periods=2).std()
            )
        for lag in lags:
            out[f"{col}_lag{lag}"] = grouped[col].shift(lag)

    out["time_since_start"] = grouped["cycle"].transform(lambda s: s - s.min()).astype("int64")
    return out
