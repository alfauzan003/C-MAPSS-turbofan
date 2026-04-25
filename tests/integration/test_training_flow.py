"""Integration test: training flow against real postgres + minio + mlflow.

Requires: docker compose up -d postgres minio minio-init mlflow.
Seeds raw_sensor.readings with a small synthetic dataset, runs the flow
in-process, and verifies a model was registered.
"""

import os

import mlflow
import pandas as pd
import pytest
from mlflow.tracking import MlflowClient
from sqlalchemy import Engine, text

from pdm.flows.training_flow import training_flow
from pdm.models.registry import REGISTERED_MODEL_NAME

SENSOR_COLS = [f"sensor_{i}" for i in range(1, 22)]


def _seed_synthetic_runs(db_engine: Engine, n_engines: int = 8, max_cycle: int = 60) -> int:
    """Insert run-to-failure traces for `n_engines` synthetic engines."""
    rows = []
    for engine in range(1, n_engines + 1):
        for cycle in range(1, max_cycle + 1):
            row = {
                "engine_id": engine,
                "cycle": cycle,
                "op_setting_1": 0.0,
                "op_setting_2": 0.0,
                "op_setting_3": 100.0,
            }
            for i in range(1, 22):
                # A noisy linear degradation signal so XGBoost has something to learn
                row[f"sensor_{i}"] = float(i) + 0.01 * cycle + (engine % 3) * 0.1
            rows.append(row)
    with db_engine.begin() as c:
        c.execute(text("TRUNCATE TABLE raw_sensor.readings RESTART IDENTITY"))
        # Insert via SQLAlchemy bulk; ts will default to now()
        cols = ", ".join(rows[0].keys())
        placeholders = ", ".join(f":{k}" for k in rows[0].keys())
        c.execute(text(f"INSERT INTO raw_sensor.readings ({cols}) VALUES ({placeholders})"), rows)
    return len(rows)


@pytest.mark.integration
def test_training_flow_registers_model(db_engine: Engine):
    # Make sure mlflow is reachable (skip with clear message otherwise)
    tracking_uri = os.environ.setdefault("MLFLOW_TRACKING_URI", "http://localhost:5000")
    os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")
    os.environ.setdefault("AWS_ACCESS_KEY_ID", "minioadmin")
    os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "minioadmin")

    try:
        mlflow.set_tracking_uri(tracking_uri)
        MlflowClient().search_experiments()
    except Exception as e:
        pytest.skip(f"MLflow not reachable at {tracking_uri}: {e}")

    n = _seed_synthetic_runs(db_engine, n_engines=10, max_cycle=80)
    assert n > 0

    # Run the flow in-process. NOTE: training_flow uses NOW()-interval, so we pass
    # a generous hours window. The just-inserted rows will have ts=now().
    result = training_flow(hours=24, max_rul=50)
    assert result["status"] == "ok", result
    assert "run_id" in result
    assert "model_version" in result

    # Verify the model exists in the registry
    client = MlflowClient()
    versions = client.search_model_versions(f"name = '{REGISTERED_MODEL_NAME}'")
    assert len(versions) >= 1
    # Newest version corresponds to the run we just made
    found = [v for v in versions if v.run_id == result["run_id"]]
    assert len(found) == 1
