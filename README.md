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
| Phase 3 | Prediction path (RUL serving API) | ✅ Complete |
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

| Service | Port | URL |
|---|---|---|
| ingestion-api | 8000 | http://localhost:8000/docs |
| prediction-api | 8001 | http://localhost:8001/docs |
| MLflow UI | 5000 | http://localhost:5000 |
| Prefect UI | 4200 | http://localhost:4200 |
| MinIO console | 9001 | http://localhost:9001 |
| postgres | 5433 | `psql postgres://pdm:pdm@localhost:5433/pdm` |

- **simulator** — posts FD001 rows to ingestion-api every 5 s
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

To promote a model version to the `champion` alias (required before Phase 3 serving):

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

## Calling /predict

After a successful Phase 2 training + champion alias promotion, the prediction-api endpoint is ready to serve RUL predictions.

```bash
curl -s -X POST http://localhost:8001/predict \
  -H 'Content-Type: application/json' \
  -d '{"readings": [{"engine_id": 1, "cycle": 1, "op_setting_1": 0.0, "op_setting_2": 0.0, "op_setting_3": 100.0, "sensor_1": 518.67, "sensor_2": 641.82, "sensor_3": 1589.7, "sensor_4": 1400.6, "sensor_5": 14.62, "sensor_6": 21.61, "sensor_7": 554.36, "sensor_8": 2388.02, "sensor_9": 9046.19, "sensor_10": 1.3, "sensor_11": 47.47, "sensor_12": 521.66, "sensor_13": 2388.02, "sensor_14": 8138.62, "sensor_15": 8.4195, "sensor_16": 0.03, "sensor_17": 392, "sensor_18": 2388, "sensor_19": 100.0, "sensor_20": 38.86, "sensor_21": 23.3619}]}'
```

This returns the predicted RUL for the last cycle in the window. Every call is logged to `predictions.served` for Phase 4 drift monitoring.

To reload the model after promoting a newer version in MLflow:

```bash
curl -X POST http://localhost:8001/reload-model
```
