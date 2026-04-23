# Predictive Maintenance ML Pipeline

End-to-end predictive maintenance system on the NASA C-MAPSS turbofan dataset.
Predicts Remaining Useful Life (RUL) with XGBoost; orchestrated by Prefect;
served via FastAPI; tracked in MLflow; deployed via docker-compose.

**Status:** Phase 0 (Foundations) — see `docs/superpowers/specs/` for the design.

## Quick start (local)

```bash
docker compose up -d
pytest
```

## Architecture

See [docs/superpowers/specs/2026-04-22-predictive-maintenance-pipeline-design.md](docs/superpowers/specs/2026-04-22-predictive-maintenance-pipeline-design.md).
