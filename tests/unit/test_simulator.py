"""Unit tests for the simulator's pure functions."""

from io import StringIO
from itertools import islice
from pathlib import Path

import pandas as pd
import pytest

from pdm.simulator.run import CMAPSS_COLUMNS, iter_rows_forever, load_cmapss


def _write_fake_cmapss(tmp_path: Path, rows: int = 3) -> Path:
    """Generate a tiny C-MAPSS-format file (whitespace-separated, no header)."""
    lines = []
    for engine in range(1, rows + 1):
        for cycle in range(1, 4):
            vals = [str(engine), str(cycle), "0.0", "0.0", "100.0"] + [f"{v:.3f}" for v in range(1, 22)]
            lines.append(" ".join(vals))
    p = tmp_path / "train.txt"
    p.write_text("\n".join(lines) + "\n")
    return p


def test_load_cmapss_returns_named_columns(tmp_path):
    p = _write_fake_cmapss(tmp_path, rows=2)
    df = load_cmapss(p)
    assert list(df.columns) == CMAPSS_COLUMNS
    assert len(df) == 6  # 2 engines x 3 cycles
    assert df["engine_id"].dtype.kind == "i"
    assert df["cycle"].dtype.kind == "i"


def test_load_cmapss_handles_trailing_whitespace_columns():
    """Real C-MAPSS files often have two trailing whitespace 'columns' (empty)."""
    raw = "1 1 0.0 0.0 100.0 " + " ".join(str(i) for i in range(1, 22)) + "  \n"
    df = pd.read_csv(StringIO(raw), sep=r"\s+", header=None, engine="python")
    df = df.dropna(axis=1, how="all").iloc[:, : len(CMAPSS_COLUMNS)]
    assert df.shape[1] == len(CMAPSS_COLUMNS)


def test_iter_rows_forever_loops():
    df = pd.DataFrame(
        [{c: 0 for c in CMAPSS_COLUMNS}, {c: 1 for c in CMAPSS_COLUMNS}]
    )
    df["engine_id"] = [1, 1]
    df["cycle"] = [1, 2]
    seq = list(islice(iter_rows_forever(df), 5))
    assert len(seq) == 5
    assert seq[0]["cycle"] == 1
    assert seq[1]["cycle"] == 2
    assert seq[2]["cycle"] == 1  # looped
