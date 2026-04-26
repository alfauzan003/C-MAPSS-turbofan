"""PredictService — wraps a loaded model + the feature pipeline."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any

import pandas as pd

from pdm.features import compute_windows
from pdm.logging import get_logger
from pdm.schemas import PredictReadingRow, PredictResponse

SENSOR_COLS = [f"sensor_{i}" for i in range(1, 22)]


def _readings_to_df(rows: list[PredictReadingRow]) -> pd.DataFrame:
    return pd.DataFrame([r.model_dump() for r in rows])


def _fingerprint(rows: list[PredictReadingRow]) -> str:
    """Stable SHA-256 of the (sorted-key) JSON serialization of the input."""
    blob = json.dumps([r.model_dump() for r in rows], sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:32]


@dataclass
class PredictResult:
    engine_id: int
    predicted_rul: float
    model_name: str
    model_version: str
    n_input_rows: int
    latency_ms: float
    input_fingerprint: str

    def to_response(self) -> PredictResponse:
        return PredictResponse(
            engine_id=self.engine_id,
            predicted_rul=self.predicted_rul,
            model_name=self.model_name,
            model_version=self.model_version,
            n_input_rows=self.n_input_rows,
            latency_ms=self.latency_ms,
        )


class PredictService:
    """In-memory wrapper around a trained model.

    Constructed once at app startup. `predict()` is synchronous and returns
    a PredictResult; the caller is responsible for persisting it.
    """

    def __init__(self, model: Any, model_name: str, model_version: str):
        self.model = model
        self.model_name = model_name
        self.model_version = model_version
        self._log = get_logger("predict")

    def predict(self, rows: list[PredictReadingRow]) -> PredictResult:
        if not rows:
            raise ValueError("rows must be non-empty")
        engine_ids = {r.engine_id for r in rows}
        if len(engine_ids) > 1:
            raise ValueError(f"all rows must share an engine_id, got {engine_ids}")
        engine_id = next(iter(engine_ids))

        t0 = time.perf_counter()
        df = _readings_to_df(rows)
        feats = compute_windows(df, sensor_cols=SENSOR_COLS)

        # Drop rows whose lag/rolling features are still NaN (start-of-window),
        # but never return empty — fall back to the last row with NaN-fill of zero.
        feature_cols = [c for c in feats.columns if c not in ("engine_id", "cycle")]
        clean = feats.dropna(subset=feature_cols)
        if clean.empty:
            clean = feats.fillna(0.0)

        # Predict on all rows of the window; report the LAST (most recent) prediction.
        X = clean.drop(columns=["engine_id", "cycle"]).astype("float64")
        y = self.model.predict(X)
        last_pred = float(y[-1])
        latency_ms = (time.perf_counter() - t0) * 1000

        fp = _fingerprint(rows)
        self._log.info(
            "predicted",
            engine_id=engine_id,
            predicted_rul=last_pred,
            n_input=len(rows),
            latency_ms=latency_ms,
            model_version=self.model_version,
        )
        return PredictResult(
            engine_id=engine_id,
            predicted_rul=last_pred,
            model_name=self.model_name,
            model_version=self.model_version,
            n_input_rows=len(rows),
            latency_ms=latency_ms,
            input_fingerprint=fp,
        )


def load_production_service(tracking_uri: str, name: str = "pdm-rul") -> "PredictService":
    """Load the current champion model from MLflow and wrap it."""
    from pdm.models.registry import load_production
    loaded = load_production(tracking_uri, name=name)
    return PredictService(
        model=loaded.model,
        model_name=name,
        model_version=loaded.version,
    )
