# Predictive Maintenance ML Pipeline

An end-to-end MLOps system for predicting the **Remaining Useful Life (RUL)** of jet engines using the [NASA C-MAPSS turbofan dataset](https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository). A sensor stream flows through ingestion, feature engineering, model training, RUL prediction, drift monitoring, and live dashboards — all running in Docker.

## Table of contents

- [What it does](#what-it-does)
- [Stack](#stack)
- [Architecture](#architecture)
- [Getting started](#getting-started)
- [Services & ports](#services--ports)
- [API reference](#api-reference)
- [Dashboards](#dashboards)
- [ML model](#ml-model)
- [Automated workflows](#automated-workflows)
- [Drift monitoring](#drift-monitoring)
- [Database schema](#database-schema)
- [Configuration](#configuration)
- [Running tests](#running-tests)
- [Useful commands](#useful-commands)
- [Project structure](#project-structure)

---

## What it does

1. **Ingest** — A simulator continuously posts turbofan sensor readings to a FastAPI endpoint, storing them in PostgreSQL.
2. **Train** — A Prefect flow (every 6 h) engineers 169+ rolling and lag features from the raw readings, trains an XGBoost regressor, and auto-promotes the model if RMSE improves by ≥ 2%.
3. **Serve** — A prediction API loads the current champion model from MLflow and returns RUL predictions for incoming engine windows. Every prediction is logged to the database for monitoring.
4. **Monitor** — A second Prefect flow (every 24 h) computes Population Stability Index (PSI) comparing recent serving inputs to the training baseline, and raises an alert if any feature shifts beyond the threshold.
5. **Observe** — Two browser dashboards: a live `/metrics` page (prediction volume, latency, drift status) and an `/evaluate` page (scatter plot + degradation curves on the FD001 test set).

---

## Stack

| Layer | Technology |
|---|---|
| Language | Python 3.12 |
| API | FastAPI + Uvicorn |
| Database | PostgreSQL 16 |
| ORM / migrations | SQLAlchemy 2 + Alembic |
| Object storage | MinIO (S3-compatible) |
| ML | XGBoost 2 + scikit-learn + pandas |
| Experiment tracking | MLflow 3 |
| Orchestration | Prefect 3 |
| Logging | structlog (JSON) |
| Packaging | Docker + docker-compose v2 |

---

## Architecture

```
simulator (C-MAPSS FD001, every 5 s)
  └─ POST /sensor-readings
       └─ ingestion-api
            └─ raw_sensor.readings  (PostgreSQL)

prefect-worker — training_flow (every 6 h)
  raw_sensor.readings
    └─ feature engineering (169 features: rolling means/stds + lags)
         └─ parquet snapshot  →  MinIO  (raw-data bucket)
              └─ XGBoost train  →  MLflow  (pdm-rul model + run metrics)
                   └─ auto-promote to "champion" alias if RMSE improves ≥ 2%
                        └─ POST /reload-model  →  prediction-api

client
  └─ POST /predict
       └─ prediction-api (loads champion model from MLflow)
            └─ predictions.served  (PostgreSQL)

prefect-worker — monitoring_flow (every 24 h)
  predictions.served + raw_sensor.readings
    └─ PSI vs training baseline parquet  (MinIO)
         └─ predictions.drift_reports  (PostgreSQL)
              └─ GET /metrics  →  live dashboard
```

---

## Getting started

### Prerequisites

- **Docker Desktop** running (Docker Engine 24+)
- **Python 3.12+** — only if you want to run tests locally
- Git

### 1. Clone

```bash
git clone <repo-url>
cd Project2
```

### 2. Configure environment

```bash
cp .env.example .env
```

The defaults work out of the box. Only notable gotcha:

> **Windows:** Postgres maps to host port **5433** (not 5432) to avoid conflicts with native Windows postgres. The `.env.example` default already uses 5433 — no change needed.

### 3. Start the full stack

```bash
docker compose up -d --build
```

This builds the app image, runs Alembic migrations, creates MinIO buckets, and starts all services. First build takes 2–3 minutes. Wait ~30 s for health checks to pass.

### 4. Trigger the first training run

The prediction API cannot serve until a champion model exists. Trigger training immediately:

1. Open http://localhost:4200 (Prefect UI)
2. Go to **Deployments** → `pdm-training/training-default` → **Quick run**

After the flow completes (~1 min), the champion model is promoted and the prediction API is ready.

### 5. Verify everything is working

```bash
# Check all services are healthy
docker compose ps

# Stream ingestion logs
docker compose logs -f ingestion-api simulator

# Count rows accumulating
docker compose exec postgres psql -U pdm -c "SELECT count(*) FROM raw_sensor.readings;"

# Make a test prediction
curl -s -X POST http://localhost:8001/predict \
  -H 'Content-Type: application/json' \
  -d '{"readings": [{"engine_id": 1, "cycle": 50, "op_setting_1": 0.0, "op_setting_2": 0.0, "op_setting_3": 100.0, "sensor_1": 518.67, "sensor_2": 641.82, "sensor_3": 1589.7, "sensor_4": 1400.6, "sensor_5": 14.62, "sensor_6": 21.61, "sensor_7": 554.36, "sensor_8": 2388.02, "sensor_9": 9046.19, "sensor_10": 1.3, "sensor_11": 47.47, "sensor_12": 521.66, "sensor_13": 2388.02, "sensor_14": 8138.62, "sensor_15": 8.4195, "sensor_16": 0.03, "sensor_17": 392, "sensor_18": 2388, "sensor_19": 100.0, "sensor_20": 38.86, "sensor_21": 23.3619}]}'
```

---

## Services & ports

| Service | Host port | URL | Notes |
|---|---|---|---|
| ingestion-api | 8000 | http://localhost:8000/docs | Swagger UI |
| prediction-api | 8001 | http://localhost:8001/docs | Swagger UI |
| MLflow UI | 5000 | http://localhost:5000 | Experiment runs + model registry |
| Prefect UI | 4200 | http://localhost:4200 | Flow runs + deployments |
| MinIO console | 9001 | http://localhost:9001 | Object storage (user: `minioadmin` / `minioadmin`) |
| MinIO S3 API | 9000 | — | Internal S3 endpoint |
| Postgres | 5433 | `psql postgres://pdm:pdm@localhost:5433/pdm` | Direct access |

---

## API reference

### Ingestion API — port 8000

#### `POST /sensor-readings`

Accepts a single sensor reading from an engine.

**Request body**

| Field | Type | Constraints |
|---|---|---|
| `engine_id` | int | > 0 |
| `cycle` | int | > 0 |
| `op_setting_1` | float | — |
| `op_setting_2` | float | — |
| `op_setting_3` | float | — |
| `sensor_1` … `sensor_21` | float (×21) | — |
| `ts` | datetime (ISO 8601) | optional; defaults to `NOW()` |

**Responses**

| Code | Meaning |
|---|---|
| 201 | Reading stored; returns `{id, engine_id, cycle}` |
| 409 | Duplicate — `(engine_id, cycle)` already exists |
| 422 | Validation error |

#### `GET /health`

Returns `{status: "ok" | "degraded", detail: string | null}`. Checks DB connectivity.

---

### Prediction API — port 8001

#### `POST /predict`

Predicts RUL for a single engine using its recent sensor window.

**Request body**

```json
{
  "readings": [
    {
      "engine_id": 1,
      "cycle": 50,
      "op_setting_1": 0.0,
      "op_setting_2": 0.0,
      "op_setting_3": 100.0,
      "sensor_1": 518.67,
      "... sensor_2 through sensor_21 ...": "..."
    }
  ]
}
```

- `readings`: 1–500 rows; all must share the same `engine_id`
- Order matters — rows should be in ascending cycle order for correct lag/rolling features

**Response**

```json
{
  "engine_id": 1,
  "predicted_rul": 87.4,
  "model_name": "pdm-rul",
  "model_version": "3",
  "n_input_rows": 1,
  "latency_ms": 12.3
}
```

- Returns the prediction for the **last (most recent) cycle** in the window
- Every call is logged to `predictions.served` for drift monitoring
- If fewer than 6 rows are passed, lag-5 features will be NaN (zero-filled with a warning)

**Responses**

| Code | Meaning |
|---|---|
| 200 | Prediction returned |
| 422 | Validation error or mixed engine IDs |
| 503 | Model not loaded (no champion in MLflow yet) |

#### `POST /reload-model`

Hot-reloads the current `champion` alias from MLflow without restarting the container. Returns `{loaded: "<version>"}`. Called automatically after auto-promotion; safe to call manually at any time.

#### `GET /health`

Returns `{status: "ok" | "degraded", detail: string | null}`. Checks both DB connectivity and model loaded state.

#### `GET /metrics`

HTML dashboard — see [Dashboards](#dashboards).

#### `GET /evaluate`

HTML evaluation page — see [Dashboards](#dashboards).

---

## Dashboards

### `/metrics` — Live prediction dashboard

http://localhost:8001/metrics

Server-rendered page (refresh to update). Shows:

**Prediction volume (last 24 h)**
- Total predictions served
- Distinct engines seen
- p50 latency (ms)
- p95 latency (ms)

**Latest drift report**
- Report timestamp (UTC)
- Comparison window duration (hours)
- Maximum PSI across all features
- Alert status: YES (red) / no (green)

**Feature PSI table** (sorted by PSI descending)

| PSI range | Status label | Colour |
|---|---|---|
| > 0.25 | significant | red |
| 0.10 – 0.25 | moderate | — |
| < 0.10 | stable | green |

> The drift report section only appears after the monitoring flow has run at least once.

---

### `/evaluate` — Model evaluation on FD001 test set

http://localhost:8001/evaluate

Runs inference on `data/cmapss/test_FD001.txt` at request time using the live champion model. Shows:

**Test-set metrics** (last-cycle prediction per engine vs ground-truth RUL labels)
- RMSE (cycles)
- MAE (cycles)
- C-MAPSS Score (asymmetric penalty — see [Evaluation metrics](#evaluation-metrics))
- Number of test engines (100 for FD001)

**Scatter plot** — Predicted RUL vs Actual RUL
- One point per test engine at its last observed cycle
- Dashed diagonal = perfect prediction line
- Useful for spotting systematic over/under-prediction

**Degradation curves** (interactive dropdown per engine)
- X-axis: cycle number
- Y-axis: RUL (cycles remaining)
- Blue solid line: actual RUL
- Orange dashed line: model's prediction at each cycle
- Shows how well the model tracks degradation over an engine's full life

---

## ML model

### Dataset

[NASA C-MAPSS FD001](https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository) — 100 training engines, 100 test engines, single operating condition, single fault mode. Each engine runs from new until failure; 21 sensor channels recorded per cycle.

### Feature engineering

For each of the 21 sensors, the following features are computed **per engine** (grouped so features don't leak across engines):

| Feature type | Windows / lags | Count per sensor |
|---|---|---|
| Rolling mean | 5, 10, 20 cycles | 3 |
| Rolling std | 5, 10, 20 cycles | 3 |
| Lag values | 1, 2, 5 cycles back | 3 |

Plus one global feature: `time_since_start` = current cycle − min cycle for that engine.

**Total features: 21 × 9 + 1 = 190** (rows with NaN from early cycles are dropped before training).

### Target variable

`RUL = max_cycle(engine) − current_cycle`, capped at **125 cycles** (standard C-MAPSS practice to suppress the long flat region early in engine life).

### Model

XGBoost Regressor with the following hyperparameters:

| Parameter | Value |
|---|---|
| `objective` | `reg:squarederror` |
| `n_estimators` | 400 |
| `max_depth` | 6 |
| `learning_rate` | 0.05 |
| `subsample` | 0.8 |
| `colsample_bytree` | 0.8 |
| `tree_method` | `hist` |

**Train/val split**: GroupShuffleSplit at engine level (80/20), seed=42. Engines do not leak between splits.

### Evaluation metrics

| Metric | Formula | Notes |
|---|---|---|
| RMSE | `√mean((ŷ − y)²)` | Standard regression error in cycles |
| MAE | `mean(\|ŷ − y\|)` | Mean absolute error in cycles |
| C-MAPSS Score | `Σ (exp(−d/13)−1)` if `d<0`, else `Σ (exp(d/10)−1)` where `d = ŷ − y` | Asymmetric; **late predictions (positive d) are penalised ~10× harder than early ones** |

### Model registry

Models are registered in MLflow as `pdm-rul`. The current best model holds the **`champion` alias**. The prediction API always loads the champion at startup and on `/reload-model`.

---

## Automated workflows

Both workflows are served as Prefect deployments by `prefect-worker` and run on schedule. Trigger manually via the Prefect UI or CLI.

### Training flow (`pdm-training/training-default`)

**Schedule**: every 6 hours (configurable via `TRAINING_INTERVAL_SECONDS`)

| Step | Task | What happens |
|---|---|---|
| 1 | Fetch readings | Load last 24 h of raw sensor rows from `raw_sensor.readings` |
| 2 | Build features | Compute rolling/lag features, add RUL target (cap 125), drop NaN rows |
| 3 | Snapshot to MinIO | Write feature DataFrame to parquet at `raw-data/training-snapshots/<run_id>.parquet` |
| 4 | Train model | XGBoost with GroupShuffleSplit; log params, RMSE/MAE/score, feature importance to MLflow |
| 5 | Auto-promote | If new RMSE improves on champion by ≥ `PROMOTE_RMSE_IMPROVEMENT_PCT` (default 2%), set `champion` alias and call `/reload-model` |

### Monitoring flow (`pdm-monitoring/monitoring-default`)

**Schedule**: every 24 hours (configurable via `MONITORING_INTERVAL_SECONDS`)

| Step | Task | What happens |
|---|---|---|
| 1 | Get champion info | Look up current champion version + training parquet URI from MLflow |
| 2 | Fetch recent inputs | Get raw sensor rows for engines that received predictions in the last 24 h |
| 3 | Compute PSI | Per-sensor PSI between training baseline (parquet) and recent serving data (10 quantile bins) |
| 4 | Write drift report | Save results to `predictions.drift_reports`; set `alert=true` if `max_psi > 0.25` |

Flow returns `{status: "skipped_no_inputs"}` if no predictions were served in the window.

---

## Drift monitoring

Drift is measured using **Population Stability Index (PSI)** per sensor feature:

```
PSI = Σ (p_compare − p_baseline) × ln(p_compare / p_baseline)
```

- **Baseline**: training set (loaded from MinIO parquet linked to the champion model)
- **Compare**: raw sensor readings for engines served in the last 24 h
- **Bins**: 10 quantile-based histogram bins derived from the baseline
- **Alert threshold**: `max_psi > 0.25`

| PSI | Interpretation |
|---|---|
| < 0.10 | No significant change |
| 0.10 – 0.25 | Moderate shift — monitor |
| > 0.25 | Significant shift — **alert triggered** |

Drift reports are stored in `predictions.drift_reports` and visible on the `/metrics` dashboard.

---

## Database schema

Four schemas in PostgreSQL:

### `raw_sensor.readings`

Raw sensor readings from the ingestion API.

| Column | Type | Notes |
|---|---|---|
| `id` | bigint PK | Auto-increment |
| `engine_id` | int | Indexed |
| `cycle` | int | — |
| `op_setting_1/2/3` | float | Operational parameters |
| `sensor_1` … `sensor_21` | float ×21 | Sensor channels |
| `ts` | timestamptz | Client-supplied; defaults to `NOW()` |
| `ingested_at` | timestamptz | Server write time; defaults to `NOW()` |

Unique constraint on `(engine_id, cycle)`.

### `features.engine_window`

Metadata index for training feature snapshots.

| Column | Type | Notes |
|---|---|---|
| `id` | bigint PK | — |
| `training_run_id` | varchar(64) | Indexed |
| `engine_id` | int | — |
| `cycle` | int | — |
| `rul` | int | Target value for this row |
| `parquet_uri` | varchar(512) | MinIO path to full feature parquet |
| `created_at` | timestamptz | — |

### `predictions.served`

Log of every prediction request.

| Column | Type | Notes |
|---|---|---|
| `id` | bigint PK | — |
| `engine_id` | int | Indexed |
| `predicted_rul` | float | — |
| `model_name` | varchar(64) | e.g. `pdm-rul` |
| `model_version` | varchar(32) | Indexed |
| `input_fingerprint` | varchar(64) | SHA-256 of input rows |
| `n_input_rows` | int | — |
| `latency_ms` | float | — |
| `served_at` | timestamptz | Indexed; defaults to `NOW()` |

### `predictions.drift_reports`

One row per monitoring flow run.

| Column | Type | Notes |
|---|---|---|
| `id` | bigint PK | — |
| `model_name` | varchar(64) | — |
| `model_version` | varchar(32) | Indexed |
| `window_start` | timestamptz | Start of comparison window |
| `window_end` | timestamptz | End of comparison window |
| `n_baseline_rows` | int | Training set size |
| `n_compare_rows` | int | Recent serving data size |
| `psi_per_feature` | JSON | `{"sensor_3": 0.12, "sensor_7": 0.31, ...}` |
| `max_psi` | float | Indexed |
| `alert` | boolean | `true` if `max_psi > 0.25` |
| `created_at` | timestamptz | Indexed |

---

## Configuration

All settings are loaded from `.env` (copy from `.env.example`). Sensitive values have safe defaults for local dev.

| Variable | Default | Description |
|---|---|---|
| `POSTGRES_HOST` | `postgres` | DB hostname (Docker service name) |
| `POSTGRES_PORT` | `5432` | DB port inside Docker (host mapped to 5433) |
| `POSTGRES_USER` | `pdm` | — |
| `POSTGRES_PASSWORD` | `pdm` | — |
| `POSTGRES_DB` | `pdm` | — |
| `MINIO_ENDPOINT_URL` | `http://minio:9000` | S3-compatible endpoint |
| `MINIO_ACCESS_KEY` | `minioadmin` | — |
| `MINIO_SECRET_KEY` | `minioadmin` | — |
| `MINIO_BUCKET_RAW` | `raw-data` | Bucket for training parquet snapshots |
| `MINIO_BUCKET_ARTIFACTS` | `mlflow-artifacts` | Bucket for MLflow artifacts |
| `LOG_LEVEL` | `INFO` | structlog level |
| `INGEST_INTERVAL_SEC` | `5` | Simulator sleep between posts (seconds) |
| `PROMOTE_RMSE_IMPROVEMENT_PCT` | `2.0` | Minimum RMSE improvement % to auto-promote |
| `TRAINING_INTERVAL_SECONDS` | `21600` | Training flow schedule (6 h) |
| `MONITORING_INTERVAL_SECONDS` | `86400` | Monitoring flow schedule (24 h) |
| `MLFLOW_TRACKING_URI` | `http://mlflow:5000` | MLflow server |
| `PREDICTION_API_URL` | `http://prediction-api:8001` | Used by training flow to trigger reload |

---

## Running tests

Unit tests have no external dependencies:

```bash
pip install -e ".[dev]"
python -m pytest tests/unit/ -v
```

Integration tests require the Docker stack:

```bash
docker compose up -d postgres minio mlflow
python -m pytest tests/ -v
```

The test suite has 56 unit tests and integration tests covering the ingestion API, prediction API, training flow, and monitoring flow.

---

## Useful commands

```bash
# Rebuild and restart a single service after a code change
docker compose up -d --build prediction-api

# Stream all logs
docker compose logs -f

# Stream specific services
docker compose logs -f ingestion-api simulator prefect-worker

# Open a postgres shell
docker compose exec postgres psql -U pdm

# Count predictions served
docker compose exec postgres psql -U pdm -c "SELECT count(*) FROM predictions.served;"

# View latest drift report
docker compose exec postgres psql -U pdm -c \
  "SELECT model_version, max_psi, alert, created_at FROM predictions.drift_reports ORDER BY created_at DESC LIMIT 5;"

# Force-promote a specific MLflow model version to champion
docker compose exec prefect-worker python - <<'PY'
import os
from pdm.models.registry import promote_to_production
promote_to_production(version="1", tracking_uri=os.environ["MLFLOW_TRACKING_URI"])
PY

# Reload the prediction model after manually promoting a version
curl -X POST http://localhost:8001/reload-model

# Trigger training flow immediately via CLI
docker compose exec prefect-worker prefect deployment run pdm-training/training-default

# Stop everything (preserves volumes)
docker compose down

# Full reset — deletes all data
docker compose down -v
```

---

## Project structure

```
pdm/                       # Main Python package
  apis/
    ingestion_api.py       # POST /sensor-readings, GET /health
    prediction_api.py      # POST /predict, /reload-model, GET /health, /metrics, /evaluate
  features/
    rul.py                 # compute_rul() — adds capped RUL target column
    windows.py             # compute_windows() — rolling means/stds + lags per sensor
  flows/
    training_flow.py       # Prefect flow: fetch → features → snapshot → train → promote
    monitoring_flow.py     # Prefect flow: PSI drift vs training baseline → drift report
    _serve.py              # Single worker entrypoint: serves both deployments
  models/
    train.py               # train_and_log(), compare_and_promote_decision(), promote()
    evaluate.py            # rmse(), mae(), cmapss_score()
    registry.py            # load_production(), promote_to_production() — MLflow alias API
  monitoring/
    drift.py               # compute_psi(), compute_psi_per_column() — PSI implementation
  orm/
    raw_sensor.py          # SensorReading → raw_sensor.readings
    features.py            # EngineWindow → features.engine_window
    predictions.py         # ServedPrediction + DriftReport → predictions.*
  simulator/
    run.py                 # C-MAPSS CSV loader + HTTP posting loop
  templates/
    metrics.html           # Jinja2 — /metrics dashboard
  config.py                # pydantic-settings; get_settings() with lru_cache
  db.py                    # Engine + session factory
  predict.py               # PredictService — loads champion model, runs inference
  schemas.py               # Pydantic request/response schemas
migrations/                # Alembic versions (4 migrations: raw_sensor → features → predictions)
data/cmapss/               # NASA C-MAPSS FD001–FD004 (train/test/RUL files — committed)
docker/
  mlflow.Dockerfile        # MLflow server image (pins mlflow>=3,<4)
scripts/postgres-init/     # DB init: pg_hba trust auth, create mlflow + prefect databases
tests/
  unit/                    # 56 tests — no Docker required
  integration/             # Requires full Docker stack
```
