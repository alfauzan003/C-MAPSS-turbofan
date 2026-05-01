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
  storage.py              # write_parquet(df, bucket, key), read_parquet(uri) — MinIO via boto3/pyarrow
  orm/
    __init__.py            # exports SensorReading, EngineWindow, ServedPrediction, DriftReport
    raw_sensor.py          # SensorReading ORM model → raw_sensor.readings
    features.py            # EngineWindow ORM model → features.engine_window
    predictions.py         # ServedPrediction + DriftReport ORM models → predictions.* schema
  apis/
    ingestion_api.py       # FastAPI app: POST /sensor-readings, GET /health
    prediction_api.py      # FastAPI app: POST /predict, POST /reload-model, GET /health, GET /metrics
  simulator/
    run.py                 # C-MAPSS CSV loader + posting loop
  features/
    __init__.py            # exports compute_rul, compute_windows
    rul.py                 # compute_rul(df, max_rul) → adds "rul" column
    windows.py             # compute_windows(df, sensor_cols, windows, lags)
  models/
    evaluate.py            # rmse(), mae(), cmapss_score()
    train.py               # train_and_log() → TrainingResult; compare_and_promote_decision(); promote(); trigger_reload()
    registry.py            # load_production(), promote_to_production() — uses aliases not stages
  monitoring/
    __init__.py            # exports compute_psi, compute_psi_per_column
    drift.py               # compute_psi() — Population Stability Index; PSI_ALERT_THRESHOLD=0.25
  flows/
    training_flow.py       # Prefect flow: fetch → features → snapshot → train → auto-promote
    monitoring_flow.py     # Prefect flow: champion model → baseline parquet → PSI → DriftReport
    _serve.py              # Single worker entrypoint: serves both training + monitoring deployments
  predict.py               # PredictService, PredictResult, load_production_service()
  schemas.py               # Pydantic schemas incl. PredictRequest, PredictResponse
  templates/
    metrics.html           # Jinja2 template for GET /metrics dashboard
migrations/
  env.py                  # Alembic env; imports get_settings() + pdm.orm for metadata
  versions/f3e7be5076bd_create_raw_sensor_readings.py
  versions/e3e436e8bb0c_create_features_engine_window.py
  versions/90efe718533c_create_predictions_served.py
  versions/5d70d7c75074_create_predictions_drift_reports.py
scripts/
  postgres-init/
    00-hba-md5.sh          # Sets pg_hba.conf to trust auth (fixes Windows psycopg auth)
    01-create-test-db.sql  # Creates pdm_test database
    02-create-mlflow-db.sql  # Creates mlflow database (idempotent)
    03-create-prefect-db.sql # Creates prefect database (idempotent)
data/cmapss/              # NASA C-MAPSS files (train/test/RUL FD001-FD004) — committed
tests/
  conftest.py             # db_engine (drops predictions+features+raw_sensor+public schemas, runs migrations)
  unit/                   # 56 tests — no Docker needed
  integration/            # requires Docker full stack (postgres + minio + mlflow)
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

## Current status (2026-05-01)

| Phase | Status |
|-------|--------|
| Phase 0 — Foundations | Complete |
| Phase 1 — Ingestion Path | Complete |
| Phase 2 — Feature Engineering / Prefect / MLflow | Complete |
| Phase 3 — Prediction Path / FastAPI RUL endpoint | Complete |
| Phase 4 — Monitoring + Auto-Promotion | Complete |
| Phase 5 — Polish + VPS Deploy | Not started |

### What's running (full stack)
- **postgres** — `pdm` + `mlflow` + `prefect` databases
- **minio** — S3-compatible; buckets `raw-data` (parquet snapshots) + `mlflow-artifacts`
- **mlflow** — tracking + model registry on :5000; image `pdm-mlflow:dev`
- **prefect-server** — workflow server on :4200
- **prefect-worker** — runs `pdm.flows._serve` → hosts `pdm-training` (6h) + `pdm-monitoring` (24h)
- **ingestion-api** — FastAPI on :8000
- **prediction-api** — FastAPI on :8001; loads champion model from MLflow on startup
- **simulator** — Posts C-MAPSS FD001 rows every 5s

## Key bugs fixed (for future reference)

1. **Python 3.14 compat**: `pyproject.toml` widened to `>=3.12,<3.15`
2. **Windows port conflict**: Docker postgres mapped to 5433, not 5432
3. **pg_hba.conf auth**: `00-hba-md5.sh` sets trust auth for host connections
4. **Session leak in FastAPI**: `_session()` must be a generator (`yield`) so FastAPI closes it after each request — otherwise TRUNCATE in tests hangs waiting for lock
5. **conftest schema cleanup**: `db_engine` fixture must drop `raw_sensor` schema before `public`, otherwise migration fails on second run
6. **Alembic migration**: Must manually add `CREATE SCHEMA IF NOT EXISTS raw_sensor` to upgrade(); auto-generated migration omits it
7. **MLflow v3 client/server mismatch**: container was on v2; local on v3 → `BAD_REQUEST (integer = character varying)`. Fix: pin `"mlflow>=3,<4"` in both `pyproject.toml` and `docker/mlflow.Dockerfile`.
8. **MLflow v3 registry API change**: `get_latest_versions`/`transition_model_version_stage` deprecated. Use `search_model_versions` + `set_registered_model_alias("champion")` / `get_model_version_by_alias`. See `pdm/models/registry.py`.
9. **DataFrame fragmentation**: column-by-column assignment to a wide DataFrame triggers `PerformanceWarning`. Collect new columns in a `dict[str, pd.Series]`, then do one `pd.concat`. See `pdm/features/windows.py`.
10. **Docker image staleness**: if a container uses an old image after `docker compose build`, run `docker compose up -d <service>` — Compose will recreate it with the new image.
11. **prefect-worker image rebuild**: worker runs `pdm.flows._serve`; if flows/ didn't exist at last build, you'll get `ModuleNotFoundError`. Fix: `docker compose build` then `docker compose up -d prefect-worker`.
12. **prediction-api lifespan**: `app.state.predict_service = None` and `app.state.startup_error = None` must both be set BEFORE the try block, so degraded startup is safe even on happy path.
13. **FastAPI session dependency must be generator**: `_session()` must use `yield`, not `return Session`. Otherwise FastAPI never closes the session and TRUNCATE in tests hangs on lock.
14. **MLflow v3 alias API — never use deprecated stages**: use `client.get_model_version_by_alias(name, alias="champion")` and `client.set_registered_model_alias(name, alias, version)`. Do NOT use `get_latest_versions(stages=["Production"])` or `transition_model_version_stage`.
15. **conftest schema teardown order**: drop `predictions` schema FIRST (before features/raw_sensor/public), otherwise foreign-key constraints from predictions→raw_sensor block the drop.
16. **Integration tests bypass MLflow** by injecting a fake `PredictService` directly onto `app.state` after `TestClient` construction — lifespan runs before the `with TestClient(app) as c:` block so overwrite immediately after.
17. **train/serve skew — `time_since_start`**: computed from `min(cycle)` in the prediction window, not the true start of the engine's life. Documented as known limitation in `PredictService.predict`.
18. **NaN fallback for short windows**: lag-5 features make the first 5 rows NaN; if fewer than 6 rows passed to `/predict`, the model still runs but the warning `nan_in_features` is logged.

## Next steps (Phase 5)

See `docs/superpowers/plans/2026-04-22-phase-5-polish-vps-deploy.md`.
Key tasks: `.env.example` audit, `deploy_vps.sh`, deploy to VPS, Caddy reverse proxy, demo URL, README screenshots.
