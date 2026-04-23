# CLAUDE.md — Session continuity for the PDM project

## Project summary

Predictive maintenance ML pipeline on the NASA C-MAPSS turbofan dataset.
Predicts Remaining Useful Life (RUL) with XGBoost; orchestrated by Prefect;
served via FastAPI; tracked in MLflow; deployed via docker-compose on a VPS.

Spec: `docs/superpowers/specs/2026-04-22-predictive-maintenance-pipeline-design.md`
Plans: `docs/superpowers/plans/`

## Tech stack (locked)

- **Language:** Python 3.12–3.14 (pyproject.toml: `>=3.12,<3.15`)
- **DB:** PostgreSQL via psycopg[binary] v3, SQLAlchemy 2.x, Alembic
- **ORM style:** `DeclarativeBase`, `Mapped`, `mapped_column`
- **Config:** pydantic-settings `BaseSettings`, loaded from `.env`
- **API:** FastAPI with lifespan, generator-based `Depends` for sessions
- **Storage:** MinIO (S3-compatible)
- **Logging:** structlog JSON, configured once at startup via `configure_logging()`
- **Orchestration:** Prefect (Phase 2+)
- **ML:** XGBoost (Phase 3+)
- **Tracking:** MLflow (Phase 3+)
- **Infra:** docker-compose v2, VPS

## Repository layout

```
pdm/                      # Main package
  config.py               # Settings (pydantic-settings), get_settings() lru_cache
  db.py                   # Base (DeclarativeBase), build_engine, get_engine, get_sessionmaker, session_scope
  logging.py              # configure_logging(), get_logger()
  schemas.py              # Pydantic schemas: SensorReadingIn, SensorReadingOut, HealthStatus
  orm/
    __init__.py            # exports SensorReading
    raw_sensor.py          # SensorReading ORM model → raw_sensor.readings
  apis/
    ingestion_api.py       # FastAPI app: POST /sensor-readings, GET /health
  simulator/
    run.py                 # C-MAPSS CSV loader + posting loop
migrations/
  env.py                  # Alembic env; imports get_settings() + pdm.orm for metadata
  versions/f3e7be5076bd_create_raw_sensor_readings.py
scripts/
  postgres-init/
    00-hba-md5.sh          # Sets pg_hba.conf to trust auth (fixes Windows psycopg auth)
    01-create-test-db.sql  # Creates pdm_test database
data/cmapss/              # NASA C-MAPSS files (train/test/RUL FD001-FD004) — committed
tests/
  conftest.py             # db_engine (session, drops raw_sensor+public, runs migrations), db_session (per-test rollback)
  unit/                   # 17 tests — no Docker needed
  integration/            # 7 tests — require Docker postgres on port 5433
```

## Environment setup

### `.env` (not committed — use `.env.example` as template)
```
POSTGRES_HOST=localhost
POSTGRES_PORT=5433          # Docker maps 5433→5432 to avoid Windows native postgres on 5432
POSTGRES_USER=pdm
POSTGRES_PASSWORD=pdm
POSTGRES_DB=pdm
MINIO_ENDPOINT_URL=http://localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
```

### Windows gotcha: port 5433
Docker Compose override maps postgres to host port **5433** (not 5432) because Windows
may have a native postgres process on 5432. All tests and `.env` default to 5433.
`docker-compose.override.yml` handles this locally; the VPS base `docker-compose.yml`
keeps 5432 container-internal only.

### Auth: trust
`scripts/postgres-init/00-hba-md5.sh` rewrites `pg_hba.conf` to use `trust` for all
host connections so psycopg[binary] on Windows doesn't hit scram-sha-256 compatibility issues.

## Running things

```bash
# Start infrastructure (postgres + minio)
docker compose up -d postgres minio

# Unit tests only (no Docker needed)
python -m pytest tests/unit/ -v

# All tests (Docker must be running)
python -m pytest tests/ -v

# Full stack
docker compose up -d --build

# Watch ingestion
docker compose logs -f ingestion-api simulator
```

## Current status (2026-04-24)

| Phase | Status |
|-------|--------|
| Phase 0 — Foundations | Complete |
| Phase 1 — Ingestion Path | Complete |
| Phase 2 — Feature Engineering / Prefect | Not started |
| Phase 3 — ML / MLflow | Not started |
| Phase 4 — Serving / FastAPI RUL endpoint | Not started |

### What's running
- **postgres** — `pdm` database + `raw_sensor.readings` table via Alembic migration
- **minio** — S3-compatible object store
- **ingestion-api** — FastAPI on :8000, accepts POST /sensor-readings
- **simulator** — Posts C-MAPSS FD001 rows to ingestion-api every 5s

## Key bugs fixed (for future reference)

1. **Python 3.14 compat**: `pyproject.toml` widened to `>=3.12,<3.15`
2. **Windows port conflict**: Docker postgres mapped to 5433, not 5432
3. **pg_hba.conf auth**: `00-hba-md5.sh` sets trust auth for host connections
4. **Session leak in FastAPI**: `_session()` must be a generator (`yield`) so FastAPI closes it after each request — otherwise TRUNCATE in tests hangs waiting for lock
5. **conftest schema cleanup**: `db_engine` fixture must drop `raw_sensor` schema before `public`, otherwise migration fails on second run
6. **Alembic migration**: Must manually add `CREATE SCHEMA IF NOT EXISTS raw_sensor` to upgrade(); auto-generated migration omits it

## Next steps (Phase 2)

See `docs/superpowers/plans/` for the Phase 2 feature engineering plan.
Key tasks: Prefect flow, feature computation (EWMA, lag features, RUL labels), write to `features` schema.
