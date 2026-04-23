# Predictive Maintenance ML Pipeline

End-to-end predictive maintenance system on the NASA C-MAPSS turbofan dataset.
Predicts Remaining Useful Life (RUL) with XGBoost; orchestrated by Prefect;
served via FastAPI; tracked in MLflow; deployed via docker-compose.

## Status

| Phase | Description | Status |
|-------|-------------|--------|
| Phase 0 | Foundations (config, DB, logging, schemas, ORM) | ✅ Complete |
| Phase 1 | Ingestion Path (API, simulator, migrations) | ✅ Complete |
| Phase 2 | Feature Engineering + Prefect | 🔜 Next |
| Phase 3 | ML training + MLflow | 🔜 Planned |
| Phase 4 | RUL serving endpoint | 🔜 Planned |

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
- **postgres** (:5433 on host) — `pdm` DB + `raw_sensor.readings` table (migrated on startup)
- **minio** (:9000 S3 API, :9001 web console)
- **ingestion-api** (:8000) — `POST /sensor-readings`, `GET /health`
- **simulator** — posts FD001 rows to ingestion-api every 5 s

### Watch data accumulate

```bash
# Stream ingestion logs
docker compose logs -f ingestion-api simulator

# Count rows in DB
docker compose exec postgres psql -U pdm -c "SELECT count(*) FROM raw_sensor.readings;"
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
                                                  ↗
                                     Alembic migration (runs at startup)
```
