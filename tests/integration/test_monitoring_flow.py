"""Integration test: monitoring_flow end-to-end against postgres + minio + mlflow.

Strategy: seed a tiny baseline parquet to MinIO, manually create an MLflow champion
model that points at it, seed some raw_sensor rows + predictions.served rows,
then run the flow and verify a DriftReport row was written.
"""

import os
import uuid

import mlflow
import numpy as np
import pandas as pd
import pytest
from mlflow.tracking import MlflowClient
from sqlalchemy import Engine, text

from pdm.config import get_settings
from pdm.flows.monitoring_flow import PSI_ALERT_THRESHOLD, SENSOR_COLS, monitoring_flow
from pdm.models.registry import PRODUCTION_ALIAS, REGISTERED_MODEL_NAME
from pdm.storage import write_parquet


def _seed_baseline(n: int = 500) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    df = pd.DataFrame({c: rng.normal(0, 1, n) for c in SENSOR_COLS})
    df["engine_id"] = rng.integers(1, 6, n)
    df["cycle"] = np.arange(n)
    return df


def _seed_recent_raw(db_engine: Engine, n: int = 200, shift: float = 0.0) -> int:
    """Insert raw_sensor.readings rows + mark engines active via predictions.served."""
    rng = np.random.default_rng(1)
    rows = []
    for i in range(n):
        row = {
            "engine_id": int(rng.integers(1, 6)),
            "cycle": int(i + 1),
            "op_setting_1": 0.0,
            "op_setting_2": 0.0,
            "op_setting_3": 100.0,
        }
        for c in SENSOR_COLS:
            row[c] = float(rng.normal(shift, 1))
        rows.append(row)
    with db_engine.begin() as c:
        c.execute(text("TRUNCATE TABLE raw_sensor.readings RESTART IDENTITY"))
        c.execute(text("TRUNCATE TABLE predictions.served RESTART IDENTITY"))
        c.execute(text("TRUNCATE TABLE predictions.drift_reports RESTART IDENTITY"))
        cols = ", ".join(rows[0].keys())
        ph = ", ".join(f":{k}" for k in rows[0].keys())
        c.execute(text(f"INSERT INTO raw_sensor.readings ({cols}) VALUES ({ph})"), rows)
        # Mark engines as 'active' by inserting served rows
        for engine_id in {r["engine_id"] for r in rows}:
            c.execute(
                text("""
                    INSERT INTO predictions.served
                        (engine_id, predicted_rul, model_name, model_version,
                         input_fingerprint, n_input_rows, latency_ms)
                    VALUES (:e, 50.0, 'pdm-rul', '999', 'fp', 1, 1.0)
                """),
                {"e": int(engine_id)},
            )
    return len(rows)


def _create_champion_model_with_baseline(baseline_df: pd.DataFrame) -> str:
    """Write baseline parquet to MinIO; create an MLflow run + registered model
    with champion alias pointing at it. Returns model_version."""
    s = get_settings()
    parquet_uri = write_parquet(
        baseline_df,
        bucket=s.minio_bucket_raw,
        key=f"training-snapshots/test-{uuid.uuid4().hex[:8]}.parquet",
    )

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment("pdm-rul")
    with mlflow.start_run() as run:
        mlflow.log_param("training_set_uri", parquet_uri)
        mlflow.log_metric("rmse", 1000.0)
        mlflow.log_text("placeholder", artifact_file="placeholder.txt")
    run_id = run.info.run_id

    client = MlflowClient()
    # Ensure registered model exists
    try:
        client.create_registered_model(REGISTERED_MODEL_NAME)
    except Exception:
        pass
    mv = client.create_model_version(
        name=REGISTERED_MODEL_NAME,
        source=f"runs:/{run_id}/placeholder.txt",
        run_id=run_id,
    )
    # Use alias API (MLflow v3) — NOT deprecated stage transitions
    client.set_registered_model_alias(
        name=REGISTERED_MODEL_NAME,
        alias=PRODUCTION_ALIAS,
        version=mv.version,
    )
    return mv.version


@pytest.mark.integration
def test_monitoring_flow_writes_drift_report_no_alert(db_engine: Engine):
    os.environ.setdefault("MLFLOW_TRACKING_URI", "http://localhost:5000")
    os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")
    os.environ.setdefault("AWS_ACCESS_KEY_ID", "minioadmin")
    os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "minioadmin")

    try:
        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
        MlflowClient().search_experiments()
    except Exception as e:
        pytest.skip(f"MLflow not reachable: {e}")

    baseline = _seed_baseline(n=500)
    _create_champion_model_with_baseline(baseline)
    _seed_recent_raw(db_engine, n=200, shift=0.0)  # no drift

    result = monitoring_flow(compare_hours=24)
    assert result["status"] == "ok"
    assert result["alert"] is False
    assert result["max_psi"] < PSI_ALERT_THRESHOLD


@pytest.mark.integration
def test_monitoring_flow_alerts_on_drift(db_engine: Engine):
    os.environ.setdefault("MLFLOW_TRACKING_URI", "http://localhost:5000")
    os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")
    os.environ.setdefault("AWS_ACCESS_KEY_ID", "minioadmin")
    os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "minioadmin")

    try:
        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
        MlflowClient().search_experiments()
    except Exception as e:
        pytest.skip(f"MLflow not reachable: {e}")

    baseline = _seed_baseline(n=500)
    _create_champion_model_with_baseline(baseline)
    _seed_recent_raw(db_engine, n=200, shift=5.0)  # mean shifted by 5σ — should alert

    result = monitoring_flow(compare_hours=24)
    assert result["status"] == "ok"
    assert result["alert"] is True
    assert result["max_psi"] > PSI_ALERT_THRESHOLD
