"""prediction-api — serves RUL predictions from the current Production model.

Endpoints:
    POST /predict        -> 200 PredictResponse, also writes to predictions.served
    POST /reload-model   -> 200 {"loaded": "<version>"} | 500 (keeps old model)
    GET  /health         -> 200 {"status": "ok" | "degraded", ...}
    GET  /metrics        -> 200 HTML dashboard (volume, latency, drift)
    GET  /evaluate       -> 200 HTML evaluation page (predicted vs actual RUL on FD001)
    GET  /docs           -> Swagger UI (auto)
"""

from __future__ import annotations

import json
import os
from collections.abc import Generator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import HTMLResponse
from jinja2 import Environment, FileSystemLoader, select_autoescape
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from pdm.config import get_settings
from pdm.db import get_engine, get_sessionmaker
from pdm.features.windows import compute_windows
from pdm.logging import configure_logging, get_logger
from pdm.models.evaluate import cmapss_score, mae, rmse
from pdm.orm import ServedPrediction
from pdm.predict import PredictService, load_production_service
from pdm.schemas import HealthStatus, PredictRequest, PredictResponse
from pdm.simulator.run import load_cmapss


def _tracking_uri() -> str:
    return os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging(level=get_settings().log_level, service="prediction-api")
    log = get_logger("prediction-api")

    app.state.predict_service = None
    app.state.startup_error = None
    try:
        app.state.predict_service = load_production_service(_tracking_uri())
        log.info("model_loaded", version=app.state.predict_service.model_version)
    except Exception as e:
        app.state.startup_error = str(e)
        log.warning("model_load_failed_starting_degraded", error=str(e))

    yield


app = FastAPI(
    title="PDM Prediction API",
    version="0.1.0",
    description="Serves RUL predictions from the current MLflow Production model.",
    lifespan=lifespan,
)

_TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "templates"
_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "cmapss"
_jinja = Environment(
    loader=FileSystemLoader(str(_TEMPLATE_DIR)),
    autoescape=select_autoescape(["html"]),
)


_SENSOR_COLS = [f"sensor_{i}" for i in range(1, 22)]


def _build_evaluate_context(svc: PredictService) -> dict:
    """Load FD001 test set, run predictions for every cycle, build template context."""
    test_df = load_cmapss(_DATA_DIR / "test_FD001.txt")
    rul_labels = pd.read_csv(
        _DATA_DIR / "RUL_FD001.txt", sep=r"\s+", header=None, engine="python"
    ).squeeze()
    # rul_labels is 0-indexed; engine ids are 1-indexed
    rul_label_map: dict[int, int] = {
        int(engine_id): int(rul_labels.iloc[engine_id - 1])
        for engine_id in test_df["engine_id"].unique()
    }

    feats = compute_windows(test_df, _SENSOR_COLS)
    feature_cols = [c for c in feats.columns if c not in ("engine_id", "cycle")]
    X = feats[feature_cols].fillna(0).astype("float64")
    preds = svc.model.predict(X)  # raw model — avoids per-engine PredictService overhead
    feats = feats.copy()
    feats["predicted_rul"] = preds

    # Scatter data: last cycle per engine vs RUL label file
    scatter = []
    for engine_id, group in feats.groupby("engine_id"):
        last_row = group.sort_values("cycle").iloc[-1]
        scatter.append({
            "engine_id": int(engine_id),
            "actual": rul_label_map[int(engine_id)],
            "predicted": round(float(last_row["predicted_rul"]), 1),
        })

    # Degradation curve data: all cycles, derive true RUL per cycle
    # true_rul_at_cycle = RUL_label + (cycles remaining to last cycle in test sequence)
    degradation: dict[str, list[dict]] = {}
    for engine_id, group in feats.groupby("engine_id"):
        group = group.sort_values("cycle")
        max_cycle = int(group["cycle"].max())
        rul_label = rul_label_map[int(engine_id)]
        true_rul_series = rul_label + (max_cycle - group["cycle"])
        records = group[["cycle", "predicted_rul"]].copy()
        records["actual_rul"] = true_rul_series.values
        records["predicted_rul"] = records["predicted_rul"].round(1)
        degradation[str(int(engine_id))] = records[["cycle", "actual_rul", "predicted_rul"]].to_dict(orient="records")

    # Metrics on last-cycle predictions vs labels
    actual_arr = np.array([s["actual"] for s in scatter], dtype=float)
    predicted_arr = np.array([s["predicted"] for s in scatter], dtype=float)

    return {
        "model_name": svc.model_name,
        "model_version": svc.model_version,
        "rmse": round(rmse(actual_arr, predicted_arr), 2),
        "mae": round(mae(actual_arr, predicted_arr), 2),
        "cmapss_score": round(cmapss_score(actual_arr, predicted_arr), 1),
        "n_engines": len(scatter),
        "scatter_json": json.dumps(scatter),
        "degradation_json": json.dumps(degradation),
        "engine_ids": sorted(rul_label_map.keys()),
        "error": None,
    }


def _service(request: Request) -> PredictService:
    svc = request.app.state.predict_service
    if svc is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"model not loaded: {request.app.state.startup_error or 'unknown'}",
        )
    return svc


def _session() -> Generator[Session, None, None]:
    """Generator-based dependency so FastAPI closes the session after each request."""
    sess = get_sessionmaker()()
    try:
        yield sess
    finally:
        sess.close()


SessionDep = Annotated[Session, Depends(_session)]
ServiceDep = Annotated[PredictService, Depends(_service)]


@app.get("/health", response_model=HealthStatus)
def health(request: Request) -> HealthStatus:
    db_ok = True
    db_err: str | None = None
    try:
        with get_engine().connect() as c:
            c.execute(text("SELECT 1"))
    except SQLAlchemyError as e:
        db_ok = False
        db_err = f"db: {type(e).__name__}"

    model_ok = request.app.state.predict_service is not None

    if db_ok and model_ok:
        return HealthStatus(status="ok")
    detail = []
    if not db_ok:
        detail.append(db_err or "db not reachable")
    if not model_ok:
        detail.append("model not loaded")
    return HealthStatus(status="degraded", detail="; ".join(detail))


@app.post("/predict", response_model=PredictResponse, status_code=status.HTTP_200_OK)
def predict(payload: PredictRequest, svc: ServiceDep, session: SessionDep) -> PredictResponse:
    try:
        result = svc.predict(payload.readings)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e)) from e
    row = ServedPrediction(
        engine_id=result.engine_id,
        predicted_rul=result.predicted_rul,
        model_name=result.model_name,
        model_version=result.model_version,
        input_fingerprint=result.input_fingerprint,
        n_input_rows=result.n_input_rows,
        latency_ms=result.latency_ms,
    )
    try:
        session.add(row)
        session.commit()
    except SQLAlchemyError as e:
        session.rollback()
        log = get_logger("prediction-api")
        log.error("prediction_log_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="failed to persist prediction",
        ) from e
    return result.to_response()


@app.post("/reload-model")
def reload_model(request: Request) -> dict:
    log = get_logger("prediction-api")
    try:
        new_svc = load_production_service(_tracking_uri())
    except Exception as e:
        log.error("reload_failed_keeping_old", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"reload failed: {e!s}",
        ) from e
    request.app.state.predict_service = new_svc
    log.info("model_reloaded", version=new_svc.model_version)
    return {"loaded": new_svc.model_version}


@app.get("/metrics", response_class=HTMLResponse)
def metrics_page(request: Request) -> HTMLResponse:
    """Server-rendered HTML metrics dashboard."""
    svc = request.app.state.predict_service
    model_name = svc.model_name if svc else "(none)"
    model_version = svc.model_version if svc else "(none)"

    try:
        with get_engine().connect() as c:
            stats = c.execute(text("""
                SELECT
                    count(*) AS total,
                    COUNT(DISTINCT engine_id) AS engines,
                    COALESCE(percentile_cont(0.5)  WITHIN GROUP (ORDER BY latency_ms), 0) AS p50,
                    COALESCE(percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_ms), 0) AS p95
                FROM predictions.served
                WHERE served_at >= NOW() - INTERVAL '24 hours'
            """)).mappings().one()

            drift_row = c.execute(text("""
                SELECT id, model_version, window_start, window_end,
                       psi_per_feature, max_psi, alert, created_at
                FROM predictions.drift_reports
                ORDER BY created_at DESC
                LIMIT 1
            """)).mappings().first()
    except SQLAlchemyError as e:
        log = get_logger("prediction-api")
        log.error("metrics_db_query_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="metrics unavailable: database error",
        ) from e

    drift_ctx = None
    if drift_row:
        psi_dict = dict(drift_row["psi_per_feature"])
        psi_sorted = sorted(psi_dict.items(), key=lambda kv: -kv[1])
        window_hours = round(
            (drift_row["window_end"] - drift_row["window_start"]).total_seconds() / 3600, 1
        )
        drift_ctx = {
            "created_at": drift_row["created_at"].strftime("%Y-%m-%d %H:%M UTC"),
            "max_psi": float(drift_row["max_psi"]),
            "alert": bool(drift_row["alert"]),
            "psi_sorted": psi_sorted,
            "window_hours": window_hours,
        }

    template = _jinja.get_template("metrics.html")
    html = template.render(
        model_name=model_name,
        model_version=model_version,
        counts={"total": stats["total"], "engines": stats["engines"]},
        latency={"p50": float(stats["p50"]), "p95": float(stats["p95"])},
        drift=drift_ctx,
    )
    return HTMLResponse(html)


@app.get("/evaluate", response_class=HTMLResponse)
def evaluate_page(request: Request) -> HTMLResponse:
    """Server-rendered HTML page: predicted vs actual RUL on FD001 test set."""
    svc = request.app.state.predict_service
    template = _jinja.get_template("evaluate.html")

    if svc is None:
        html = template.render(error=request.app.state.startup_error or "model not loaded")
        return HTMLResponse(html)

    try:
        ctx = _build_evaluate_context(svc)
    except FileNotFoundError as e:
        log = get_logger("prediction-api")
        log.error("evaluate_data_missing", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"evaluation data not found: {e}",
        ) from e

    html = template.render(**ctx)
    return HTMLResponse(html)
