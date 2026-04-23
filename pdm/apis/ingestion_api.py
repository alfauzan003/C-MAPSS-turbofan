"""ingestion-api — FastAPI service that accepts sensor readings.

Endpoints:
    POST /sensor-readings  -> 201 {"id", "engine_id", "cycle"}
    GET  /health           -> 200 {"status": "ok" | "degraded", ...}
"""

from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, status
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session

from pdm.config import get_settings
from pdm.db import get_engine, get_sessionmaker
from pdm.logging import configure_logging, get_logger
from pdm.orm import SensorReading
from pdm.schemas import HealthStatus, SensorReadingIn, SensorReadingOut


@asynccontextmanager
async def lifespan(_: FastAPI):
    configure_logging(level=get_settings().log_level, service="ingestion-api")
    log = get_logger("ingestion-api")
    log.info("starting", db_host=get_settings().postgres_host)
    yield
    log.info("stopping")


app = FastAPI(
    title="PDM Ingestion API",
    version="0.1.0",
    description="Accepts C-MAPSS-style sensor readings and persists them to raw_sensor.readings.",
    lifespan=lifespan,
)


def _session():
    session = get_sessionmaker()()
    try:
        yield session
    finally:
        session.close()


SessionDep = Annotated[Session, Depends(_session)]


@app.get("/health", response_model=HealthStatus)
def health() -> HealthStatus:
    try:
        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
        return HealthStatus(status="ok")
    except SQLAlchemyError as e:
        return HealthStatus(status="degraded", detail=f"db: {type(e).__name__}")


@app.post(
    "/sensor-readings",
    response_model=SensorReadingOut,
    status_code=status.HTTP_201_CREATED,
)
def ingest(payload: SensorReadingIn, session: SessionDep) -> SensorReadingOut:
    log = get_logger("ingestion-api")
    row = SensorReading(**payload.model_dump(exclude_none=True))
    session.add(row)
    try:
        session.commit()
    except IntegrityError as e:
        session.rollback()
        # Duplicate (engine_id, cycle) — return 409 with helpful detail
        log.warning("duplicate_reading", engine_id=payload.engine_id, cycle=payload.cycle)
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"reading already exists for engine_id={payload.engine_id}, cycle={payload.cycle}",
        ) from e
    session.refresh(row)
    log.info("ingested", id=row.id, engine_id=row.engine_id, cycle=row.cycle)
    return SensorReadingOut(id=row.id, engine_id=row.engine_id, cycle=row.cycle)
