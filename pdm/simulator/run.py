"""Sensor simulator — drips C-MAPSS rows to ingestion-api.

Reads `data/cmapss/train_FD001.txt`, posts one row at a time, sleeps
`INGEST_INTERVAL_SEC`, loops back to the start when the file ends.

Run: `python -m pdm.simulator.run`
"""

from __future__ import annotations

import os
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import httpx
import pandas as pd

from pdm.logging import configure_logging, get_logger

CMAPSS_COLUMNS = (
    ["engine_id", "cycle", "op_setting_1", "op_setting_2", "op_setting_3"]
    + [f"sensor_{i}" for i in range(1, 22)]
)


def load_cmapss(path: Path) -> pd.DataFrame:
    """Read a whitespace-separated C-MAPSS file into a DataFrame with named columns."""
    df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    # The raw files often have trailing whitespace producing two empty columns
    df = df.dropna(axis=1, how="all").iloc[:, : len(CMAPSS_COLUMNS)]
    df.columns = CMAPSS_COLUMNS
    df["engine_id"] = df["engine_id"].astype(int)
    df["cycle"] = df["cycle"].astype(int)
    return df


def iter_rows_forever(df: pd.DataFrame) -> Iterator[dict]:
    """Yield row dicts forever, starting over when we hit the end of the file."""
    while True:
        for row in df.to_dict(orient="records"):
            yield row


def post_one(client: httpx.Client, url: str, payload: dict, log) -> None:
    try:
        r = client.post(url, json=payload, timeout=5.0)
    except httpx.HTTPError as e:
        log.warning("post_failed", error=str(e))
        return
    if r.status_code == 201:
        log.info("posted", engine_id=payload["engine_id"], cycle=payload["cycle"])
    elif r.status_code == 409:
        # Duplicate is fine when we loop the file — just skip.
        log.debug("duplicate_skipped", engine_id=payload["engine_id"], cycle=payload["cycle"])
    else:
        log.warning("post_unexpected", status=r.status_code, body=r.text[:200])


def main() -> int:
    configure_logging(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        service="simulator",
    )
    log = get_logger("simulator")

    data_path = Path(os.environ.get("CMAPSS_TRAIN_FILE", "data/cmapss/train_FD001.txt"))
    api_url = os.environ.get("INGESTION_API_URL", "http://ingestion-api:8000") + "/sensor-readings"
    interval = float(os.environ.get("INGEST_INTERVAL_SEC", "5"))

    if not data_path.exists():
        log.error("seed_file_missing", path=str(data_path))
        return 1

    df = load_cmapss(data_path)
    log.info("loaded_seed", rows=len(df), engines=df["engine_id"].nunique())

    with httpx.Client() as client:
        for payload in iter_rows_forever(df):
            post_one(client, api_url, payload, log)
            time.sleep(interval)


if __name__ == "__main__":
    sys.exit(main() or 0)
