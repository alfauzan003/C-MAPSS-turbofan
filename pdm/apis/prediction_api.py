"""prediction-api — serves RUL predictions from the current Production model.

Endpoints:
    POST /predict        -> 200 PredictResponse, also writes to predictions.served
    POST /reload-model   -> 200 {"loaded": "<version>"} | 500 (keeps old model)
    GET  /health         -> 200 {"status": "ok" | "degraded", ...}
    GET  /docs           -> Swagger UI (auto)
"""

from __future__ import annotations

import os
from collections.abc import Generator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Request, status
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from pdm.config import get_settings
from pdm.db import get_engine, get_sessionmaker
from pdm.logging import configure_logging, get_logger
from pdm.orm import ServedPrediction
from pdm.predict import PredictService, load_production_service
from pdm.schemas import HealthStatus, PredictRequest, PredictResponse


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
