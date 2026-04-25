# Predictive Maintenance ML Pipeline

End-to-end predictive maintenance system on the NASA C-MAPSS turbofan dataset.
Predicts Remaining Useful Life (RUL) with XGBoost; orchestrated by Prefect;
served via FastAPI; tracked in MLflow; deployed via docker-compose.

## Status

| Phase | Description | Status |
|-------|-------------|--------|
| Phase 0 | Foundations (config, DB, logging, schemas, ORM) | ✅ Complete |
| Phase 1 | Ingestion Path (API, simulator, migrations) | ✅ Complete |
| Phase 2 | Feature Engineering + Training Path (Prefect + XGBoost + MLflow) | ✅ Complete |
| Phase 3 | Prediction path (RUL serving API) | 🔜 Next |
| Phase 4 | Monitoring + auto-promotion | 🔜 Planned |
| Phase 5 | Polish + VPS deploy | 🔜 Planned |

## Quick start

### Prerequisites

- Docker Desktop running
- NASA C-MAPSS data files in `data/cmapss/` (train_FD001.txt … RUL_FD004.txt)
- Copy `.env.example` → `.env` (defaults work for local Docker)

> **Windows note:** Docker postgres maps to port **5433** (not 5432) to avoid conflict
> with any native Windows postgres. The `.env` default already reflects this.

### Run the full stack

```bash
docker compose up -d --build
```

Services started:
- **postgres** (:5433 on host) — `pdm` DB + all schema migrations applied
- **minio** (:9000 S3 API, :9001 web console)
- **ingestion-api** (:8000) — `POST /sensor-readings`, `GET /health`
- **simulator** — posts FD001 rows to ingestion-api every 5 s
- **mlflow** (:5000) — experiment tracking + model registry UI
- **prefect-server** (:4200) — workflow orchestration UI
- **prefect-worker** — runs the `pdm-training` deployment every 6 hours

### Watch data accumulate

```bash
# Stream ingestion logs
docker compose logs -f ingestion-api simulator

# Count rows in DB
docker compose exec postgres psql -U pdm -c "SELECT count(*) FROM raw_sensor.readings;"
```

### Trigger training

The Prefect worker registers a `pdm-training` deployment that runs every 6 hours.
To trigger immediately:

1. Open http://localhost:4200 → Deployments → `pdm-training/training-default` → Quick run
2. Watch the run complete in the UI
3. Open http://localhost:5000 → Models → `pdm-rul` to see the new version

To promote a model version to Production (required before Phase 3 serving):

```bash
docker compose exec prefect-worker python - <<'PY'
import os
from pdm.models.registry import promote_to_production
promote_to_production(version="1", tracking_uri=os.environ["MLFLOW_TRACKING_URI"])
PY
```

### Run tests

```bash
# Unit tests only (no Docker required)
python -m pytest tests/unit/ -v

# Full suite (requires Docker postgres + minio running)
python -m pytest tests/ -v
```

## Architecture

See [docs/superpowers/specs/2026-04-22-predictive-maintenance-pipeline-design.md](docs/superpowers/specs/2026-04-22-predictive-maintenance-pipeline-design.md).

```
simulator → POST /sensor-readings → ingestion-api → raw_sensor.readings (postgres)
                                                         ↓
prefect-worker (training_flow, every 6h):
  raw_sensor.readings → features → MinIO parquet snapshot → XGBoost → MLflow registry
                                                                              ↓
                                                             (Phase 3: prediction-api)
```
