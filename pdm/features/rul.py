"""Derive Remaining Useful Life (RUL) target from run-to-failure traces.

For training, RUL(engine, cycle) = max_cycle(engine) - cycle.
We optionally cap RUL at `max_rul` — a standard regularization in C-MAPSS
literature that prevents the model from chasing very high RUL values that
are not informative for maintenance decisions.
"""

import pandas as pd


def compute_rul(df: pd.DataFrame, max_rul: int | None = None) -> pd.DataFrame:
    """Add a `rul` column to `df`.

    `df` must contain `engine_id` and `cycle` columns. All other columns
    are preserved. Output is the same shape as input plus one column.
    """
    if df.empty:
        out = df.copy()
        out["rul"] = pd.Series(dtype="int64")
        return out

    max_per_engine = df.groupby("engine_id")["cycle"].transform("max")
    out = df.copy()
    out["rul"] = (max_per_engine - df["cycle"]).astype("int64")
    if max_rul is not None:
        out["rul"] = out["rul"].clip(upper=max_rul)
    return out
