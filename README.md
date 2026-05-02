# Predictive Maintenance ML Pipeline

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi)
![XGBoost](https://img.shields.io/badge/XGBoost-2.x-orange)
![MLflow](https://img.shields.io/badge/MLflow-3.x-0194E2?logo=mlflow)
![Prefect](https://img.shields.io/badge/Prefect-3.x-7C3AED?logo=prefect)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-336791?logo=postgresql)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker)

End-to-end **MLOps pipeline** that predicts the Remaining Useful Life (RUL) of jet engines from raw sensor telemetry. Built on the [NASA C-MAPSS turbofan dataset](https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository).

> **What makes this a full MLOps project:** sensor ingestion → feature engineering → model training with experiment tracking → automated promotion → real-time serving → drift monitoring → live dashboards. Every layer is containerised and wired together with production patterns (health checks, structured logging, schema migrations, model registry aliases, PSI-based alerting).

---

## Pipeline overview

```
Simulator (C-MAPSS FD001, every 5 s)
    │  POST /sensor-readings
    ▼
Ingestion API ──────────────► PostgreSQL (raw_sensor.readings)
                                    │
                    Prefect training flow (every 6 h)
                                    │  feature engineering (190 features)
                                    │  XGBoost train + MLflow log
                                    │  auto-promote if RMSE improves ≥ 2%
                                    ▼
                           MLflow Model Registry
                           (champion alias) ◄─── POST /reload-model
                                    │
Client ──► POST /predict ──► Prediction API ──► PostgreSQL (predictions.served)
                                    │
                    Prefect monitoring flow (every 24 h)
                                    │  PSI drift vs training baseline
                                    ▼
                           PostgreSQL (drift_reports)
                                    │
                            GET /metrics  ◄── live dashboard
```

---

## Features at a glance

| Capability | Details |
|---|---|
| **Sensor ingestion** | FastAPI endpoint, idempotent (409 on duplicate), structured JSON logging |
| **Feature engineering** | 190 features: rolling mean/std (windows 5/10/20) + lags (1/2/5) per sensor, grouped by engine to prevent leakage |
| **Model training** | XGBoost regressor, engine-level GroupShuffleSplit (80/20), logged to MLflow (params, metrics, feature importance) |
| **Auto-promotion** | New model promoted to `champion` alias only if RMSE improves ≥ 2% over current champion |
| **RUL serving** | FastAPI prediction API, hot-reload without restart, every call logged with latency + input fingerprint |
| **Drift monitoring** | Per-feature PSI (Population Stability Index) vs training baseline; alert when `max_psi > 0.25` |
| **Dashboards** | `/metrics` (volume, latency, drift) and `/evaluate` (scatter plot + degradation curves, Chart.js) |
| **Orchestration** | Two Prefect deployments (training every 6 h, monitoring every 24 h), single worker process |
| **Observability** | Structured JSON logs (structlog), health endpoints on every API, p50/p95 latency tracking |
| **Testing** | 56 unit tests (no Docker) + integration tests covering both APIs and both flows |

---

## Tech stack

| Layer | Technology |
|---|---|
| Language | Python 3.12 |
| APIs | FastAPI + Uvicorn |
| Database | PostgreSQL 16, SQLAlchemy 2, Alembic migrations |
| Object storage | MinIO (S3-compatible), parquet via PyArrow |
| ML | XGBoost 2, scikit-learn, pandas |
| Experiment tracking | MLflow 3 (tracking + model registry) |
| Orchestration | Prefect 3 |
| Logging | structlog (JSON) |
| Infra | Docker + docker-compose v2 |

---

## Quick start

**Prerequisites:** Docker Desktop, Git. Python 3.12+ only needed to run tests locally.

```bash
# 1. Clone
git clone https://github.com/alfauzan003/C-MAPSS-turbofan.git
cd C-MAPSS-turbofan

# 2. Configure (defaults work out of the box)
cp .env.example .env

# 3. Start everything
docker compose up -d --build
```

> **Windows note:** Postgres is mapped to host port **5433** (not 5432) to avoid conflicts with native installations. The `.env.example` default already reflects this.

Once containers are healthy (~30 s), **trigger the first training run** before making predictions:

1. Open **http://localhost:4200** (Prefect UI)
2. Deployments → `pdm-training/training-default` → **Quick run**
3. After ~1 min the champion model is promoted and the prediction API is ready

---

## Services

| Service | URL | Purpose |
|---|---|---|
| Ingestion API | http://localhost:8000/docs | Accept sensor readings (Swagger UI) |
| Prediction API | http://localhost:8001/docs | Serve RUL predictions (Swagger UI) |
| `/metrics` dashboard | http://localhost:8001/metrics | Volume, latency, drift status |
| `/evaluate` dashboard | http://localhost:8001/evaluate | Test-set scatter + degradation curves |
| MLflow UI | http://localhost:5000 | Experiment runs, model registry |
| Prefect UI | http://localhost:4200 | Flow runs, deployments |
| MinIO console | http://localhost:9001 | Object storage (`minioadmin` / `minioadmin`) |
| Postgres | `localhost:5433` | `psql postgres://pdm:pdm@localhost:5433/pdm` |

---

## API

### `POST /sensor-readings` — Ingestion API (port 8000)

Accepts one sensor reading per call. Body: `engine_id`, `cycle`, `op_setting_1/2/3`, `sensor_1`–`sensor_21`, optional `ts`. Returns `201` on success, `409` on duplicate `(engine_id, cycle)`.

### `POST /predict` — Prediction API (port 8001)

Send a window of 1–500 readings for a single engine (ascending cycle order). Returns the predicted RUL for the last cycle, model name/version, and latency.

```bash
curl -s -X POST http://localhost:8001/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "readings": [{
      "engine_id": 1, "cycle": 50,
      "op_setting_1": 0.0, "op_setting_2": 0.0, "op_setting_3": 100.0,
      "sensor_1": 518.67, "sensor_2": 641.82, "sensor_3": 1589.7,
      "sensor_4": 1400.6, "sensor_5": 14.62, "sensor_6": 21.61,
      "sensor_7": 554.36, "sensor_8": 2388.02, "sensor_9": 9046.19,
      "sensor_10": 1.3, "sensor_11": 47.47, "sensor_12": 521.66,
      "sensor_13": 2388.02, "sensor_14": 8138.62, "sensor_15": 8.4195,
      "sensor_16": 0.03, "sensor_17": 392, "sensor_18": 2388,
      "sensor_19": 100.0, "sensor_20": 38.86, "sensor_21": 23.3619
    }]
  }'
```

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

### Other endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/reload-model` | POST | Hot-reload champion from MLflow (no restart needed) |
| `/health` | GET | Both APIs — DB + model state check |
| `/metrics` | GET | Live HTML dashboard (prediction API) |
| `/evaluate` | GET | Test-set evaluation with interactive charts (prediction API) |

---

## ML model

### Dataset

NASA C-MAPSS **FD001** — 100 training engines, 100 test engines, single operating condition. 21 sensor channels per cycle, recorded until engine failure.

### Features (190 total)

For each of 21 sensors, computed **per engine** (no cross-engine leakage):

- Rolling mean + std at windows **5, 10, 20** cycles → 6 features/sensor
- Lag values at **1, 2, 5** cycles back → 3 features/sensor
- `time_since_start` (global) → 1 feature

### Target

`RUL = max_cycle(engine) − current_cycle`, **capped at 125** (standard C-MAPSS practice).

### XGBoost hyperparameters

`n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, tree_method=hist`

Train/val split: **GroupShuffleSplit 80/20** at engine level, seed=42.

### Evaluation metrics

| Metric | Notes |
|---|---|
| RMSE | Standard regression error (cycles) |
| MAE | Mean absolute error (cycles) |
| C-MAPSS Score | Asymmetric penalty: late predictions (`d > 0`) penalised ~10× harder than early ones via `exp(d/10) − 1` |

---

## Automated workflows

### Training flow — every 6 hours

`fetch readings (24 h) → build 190 features → snapshot to MinIO (parquet) → XGBoost train → log to MLflow → auto-promote if RMSE improves ≥ 2% → reload prediction API`

### Monitoring flow — every 24 hours

`load champion parquet from MinIO → fetch recent serving inputs → compute PSI per sensor (10 quantile bins) → write drift report → alert if max PSI > 0.25`

Trigger either flow immediately from the **Prefect UI** or:

```bash
docker compose exec prefect-worker prefect deployment run pdm-training/training-default
```

---

## Drift monitoring

**Population Stability Index (PSI)** measures distribution shift per sensor between the training baseline and recent serving data:

```
PSI = Σ (p_current − p_baseline) × ln(p_current / p_baseline)
```

| PSI | Status |
|---|---|
| < 0.10 | Stable |
| 0.10 – 0.25 | Moderate shift |
| > 0.25 | **Alert** |

Results are persisted to `predictions.drift_reports` and surfaced on the `/metrics` dashboard with per-feature colour coding.

---

## Testing

```bash
# Unit tests — no Docker needed (56 tests)
pip install -e ".[dev]"
python -m pytest tests/unit/ -v

# Full suite — requires Docker stack
python -m pytest tests/ -v
```

Coverage: ingestion API, prediction API, feature engineering, training flow, monitoring flow, PSI computation.

---

## Configuration

Copy `.env.example` → `.env`. All defaults work locally.

| Variable | Default | Notes |
|---|---|---|
| `POSTGRES_*` | `pdm/pdm/pdm` | Host/port/user/password/db |
| `MINIO_*` | `minioadmin` | Endpoint, keys, bucket names |
| `PROMOTE_RMSE_IMPROVEMENT_PCT` | `2.0` | Auto-promotion threshold (%) |
| `INGEST_INTERVAL_SEC` | `5` | Simulator posting cadence |
| `TRAINING_INTERVAL_SECONDS` | `21600` | 6 h; override to retrain faster |
| `MONITORING_INTERVAL_SECONDS` | `86400` | 24 h |
| `LOG_LEVEL` | `INFO` | structlog level |

---

## Project structure

```
pdm/
├── apis/
│   ├── ingestion_api.py      # POST /sensor-readings, GET /health
│   └── prediction_api.py     # POST /predict, /reload-model, /metrics, /evaluate
├── features/
│   ├── rul.py                # RUL target computation (capped at 125)
│   └── windows.py            # Rolling/lag feature engineering
├── flows/
│   ├── training_flow.py      # Prefect: fetch → features → train → promote
│   ├── monitoring_flow.py    # Prefect: PSI drift detection → drift report
│   └── _serve.py             # Single worker serving both deployments
├── models/
│   ├── train.py              # XGBoost training + MLflow logging
│   ├── evaluate.py           # RMSE, MAE, C-MAPSS score
│   └── registry.py           # MLflow alias API (champion promotion)
├── monitoring/
│   └── drift.py              # PSI computation
├── orm/                      # SQLAlchemy models (4 tables across 3 schemas)
├── simulator/
│   └── run.py                # C-MAPSS CSV loader + HTTP posting loop
├── templates/
│   └── metrics.html          # Jinja2 dashboard template
├── config.py                 # pydantic-settings (.env loader)
├── db.py                     # Engine + session factory
├── predict.py                # PredictService (champion model loader)
└── schemas.py                # Pydantic request/response schemas
migrations/                   # 4 Alembic versions
data/cmapss/                  # NASA C-MAPSS FD001–FD004 (committed)
tests/
├── unit/                     # 56 tests, no Docker
└── integration/              # Full stack tests
```
