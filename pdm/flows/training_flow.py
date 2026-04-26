"""Training flow — pulls raw rows, builds features, trains, logs to MLflow.

Run modes:
    1. As a script:  `python -m pdm.flows.training_flow`
       Calls flow.serve(...) which (a) registers the deployment with the Prefect
       server it's connected to, and (b) polls for and executes scheduled runs.

    2. Manually:     `python -c "from pdm.flows.training_flow import training_flow; training_flow()"`
       Runs the flow once locally (useful for first-time validation).
"""

from __future__ import annotations

import os
import uuid
from datetime import timedelta

import pandas as pd
from prefect import flow, task
from sqlalchemy import text

from pdm.config import get_settings
from pdm.db import get_engine
from pdm.features import compute_rul, compute_windows
from pdm.logging import configure_logging, get_logger
from pdm.models.train import (
    PromoteDecision,
    compare_and_promote_decision,
    get_current_production_rmse,
    promote,
    train_and_log,
    trigger_reload,
)
from pdm.storage import write_parquet

SENSOR_COLS = [f"sensor_{i}" for i in range(1, 22)]


@task
def fetch_recent_readings(hours: int) -> pd.DataFrame:
    """Read the last `hours` hours of raw readings from postgres."""
    log = get_logger("training_flow")
    sql = text(
        """
        SELECT engine_id, cycle,
               op_setting_1, op_setting_2, op_setting_3,
               """ + ", ".join(SENSOR_COLS) + """
        FROM raw_sensor.readings
        WHERE ts >= NOW() - (:hours || ' hours')::interval
        ORDER BY engine_id, cycle
        """
    )
    with get_engine().connect() as c:
        df = pd.read_sql(sql, c, params={"hours": str(hours)})
    log.info("fetched_readings", rows=len(df), engines=df["engine_id"].nunique())
    return df


@task
def build_features(df: pd.DataFrame, max_rul: int = 125) -> pd.DataFrame:
    df = compute_windows(df, sensor_cols=SENSOR_COLS)
    df = compute_rul(df, max_rul=max_rul)
    # Drop rows where lag/rolling features are still NaN at the beginning of each engine
    feature_cols = [c for c in df.columns if c not in ("engine_id", "cycle", "rul")]
    df = df.dropna(subset=feature_cols).reset_index(drop=True)
    return df


@task
def snapshot_to_minio(df: pd.DataFrame, training_run_id: str) -> str:
    bucket = get_settings().minio_bucket_raw
    key = f"training-snapshots/{training_run_id}.parquet"
    return write_parquet(df, bucket=bucket, key=key)


@task
def train_model(df: pd.DataFrame, parquet_uri: str, training_run_id: str) -> dict:
    result = train_and_log(
        df,
        parquet_uri=parquet_uri,
        training_run_id=training_run_id,
        mlflow_tracking_uri=os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000"),
    )
    return {
        "run_id": result.run_id,
        "model_version": result.model_version,
        **result.metrics,
    }


@task
def maybe_promote_and_reload(model_version: str, new_rmse: float) -> dict:
    """Compare to current champion; promote + reload if new is better."""
    log = get_logger("training_flow")
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    prediction_api_url = os.environ.get("PREDICTION_API_URL", "http://prediction-api:8001")
    threshold = float(os.environ.get("PROMOTE_RMSE_IMPROVEMENT_PCT", "2.0"))

    current_rmse = get_current_production_rmse(tracking_uri)
    decision = compare_and_promote_decision(
        new_rmse=new_rmse,
        current_production_rmse=current_rmse,
        improvement_threshold_pct=threshold,
    )
    log.info(
        "promote_decision",
        decision=decision.value,
        new_rmse=new_rmse,
        current_rmse=current_rmse,
        threshold_pct=threshold,
    )
    if decision is PromoteDecision.HOLD:
        return {"promoted": False, "reason": "below_threshold_or_worse"}

    promote(version=model_version, tracking_uri=tracking_uri)
    reloaded = trigger_reload(prediction_api_url)
    return {"promoted": True, "reloaded": reloaded, "version": model_version}


@flow(name="pdm-training", log_prints=True)
def training_flow(hours: int = 24, max_rul: int = 125) -> dict:
    """End-to-end: read raw → features → snapshot → train → log."""
    configure_logging(level=os.environ.get("LOG_LEVEL", "INFO"), service="training_flow")
    log = get_logger("training_flow")
    training_run_id = uuid.uuid4().hex[:12]
    log.info("training_run_started", training_run_id=training_run_id)

    raw = fetch_recent_readings(hours=hours)
    if raw.empty:
        log.warning("no_data_skipping")
        return {"status": "skipped_no_data"}

    feats = build_features(raw, max_rul=max_rul)
    if feats.empty:
        log.warning("no_features_after_dropna_skipping")
        return {"status": "skipped_no_features"}

    parquet_uri = snapshot_to_minio(feats, training_run_id)
    summary = train_model(feats, parquet_uri, training_run_id)
    promo = maybe_promote_and_reload(
        model_version=summary["model_version"],
        new_rmse=summary["rmse"],
    )
    log.info("training_run_complete", **summary, **promo)
    return {"status": "ok", "training_run_id": training_run_id, **summary, **promo}


if __name__ == "__main__":
    # Serve mode: register the deployment + poll for scheduled runs.
    # Schedule = every 6 hours (configurable).
    interval = int(os.environ.get("TRAINING_INTERVAL_SECONDS", "21600"))  # 6h default
    training_flow.serve(
        name="training-default",
        interval=timedelta(seconds=interval),
        tags=["training"],
    )
