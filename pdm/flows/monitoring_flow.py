"""monitoring_flow — compute PSI between recent prediction inputs and training baseline."""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta

import pandas as pd
from prefect import flow, task
from sqlalchemy import text

from pdm.config import get_settings
from pdm.db import get_engine
from pdm.logging import configure_logging, get_logger
from pdm.models.registry import PRODUCTION_ALIAS, REGISTERED_MODEL_NAME
from pdm.monitoring import compute_psi_per_column
from pdm.orm import DriftReport
from pdm.storage import read_parquet

SENSOR_COLS = [f"sensor_{i}" for i in range(1, 22)]
PSI_ALERT_THRESHOLD = 0.25


@task
def get_active_model_info(tracking_uri: str) -> dict:
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    try:
        v = client.get_model_version_by_alias(name=REGISTERED_MODEL_NAME, alias=PRODUCTION_ALIAS)
    except Exception as exc:
        raise LookupError(
            f"No model with alias '{PRODUCTION_ALIAS}' for {REGISTERED_MODEL_NAME!r}"
        ) from exc
    run = client.get_run(v.run_id)
    parquet_uri = run.data.params.get("training_set_uri")
    if not parquet_uri:
        raise LookupError(f"Run {v.run_id} has no `training_set_uri` param")
    return {"model_version": v.version, "run_id": v.run_id, "training_set_uri": parquet_uri}


@task
def fetch_recent_inputs(compare_hours: int) -> pd.DataFrame:
    """Read raw_sensor.readings for engines that were served in the last compare_hours hours."""
    log = get_logger("monitoring_flow")
    sql = text(
        """
        WITH active_engines AS (
            SELECT DISTINCT engine_id FROM predictions.served
            WHERE served_at >= NOW() - (:hours || ' hours')::interval
        )
        SELECT r.engine_id, r.cycle,
               r.op_setting_1, r.op_setting_2, r.op_setting_3,
               """
        + ", ".join(f"r.{c}" for c in SENSOR_COLS)
        + """
        FROM raw_sensor.readings r
        JOIN active_engines e USING (engine_id)
        WHERE r.ts >= NOW() - (:hours || ' hours')::interval
        """
    )
    with get_engine().connect() as c:
        df = pd.read_sql(sql, c, params={"hours": str(compare_hours)})
    log.info("fetched_recent_inputs", rows=len(df), engines=df["engine_id"].nunique())
    return df


@task
def write_drift_report(
    model_name: str,
    model_version: str,
    window_start: datetime,
    window_end: datetime,
    n_baseline: int,
    n_compare: int,
    psi: dict[str, float],
) -> int:
    from pdm.db import get_sessionmaker

    Session = get_sessionmaker()
    max_psi = max(psi.values()) if psi else 0.0
    alert = max_psi > PSI_ALERT_THRESHOLD
    with Session() as session:
        row = DriftReport(
            model_name=model_name,
            model_version=model_version,
            window_start=window_start,
            window_end=window_end,
            n_baseline_rows=n_baseline,
            n_compare_rows=n_compare,
            psi_per_feature=psi,
            max_psi=max_psi,
            alert=alert,
        )
        session.add(row)
        session.commit()
        session.refresh(row)
        return row.id


@flow(name="pdm-monitoring", log_prints=True)
def monitoring_flow(compare_hours: int = 24) -> dict:
    configure_logging(level=os.environ.get("LOG_LEVEL", "INFO"), service="monitoring_flow")
    log = get_logger("monitoring_flow")
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")

    info = get_active_model_info(tracking_uri)
    baseline = read_parquet(info["training_set_uri"])
    compare = fetch_recent_inputs(compare_hours=compare_hours)

    if compare.empty:
        log.warning("no_recent_inputs_skipping")
        return {"status": "skipped_no_inputs"}

    psi = compute_psi_per_column(baseline, compare, columns=SENSOR_COLS)
    window_end = datetime.now(UTC)
    window_start = window_end - timedelta(hours=compare_hours)

    report_id = write_drift_report(
        model_name=REGISTERED_MODEL_NAME,
        model_version=info["model_version"],
        window_start=window_start,
        window_end=window_end,
        n_baseline=len(baseline),
        n_compare=len(compare),
        psi=psi,
    )
    max_psi = max(psi.values()) if psi else 0.0
    alert = max_psi > PSI_ALERT_THRESHOLD
    log.info(
        "drift_report_written",
        report_id=report_id,
        model_version=info["model_version"],
        max_psi=max_psi,
        alert=alert,
    )
    return {
        "status": "ok",
        "report_id": report_id,
        "max_psi": max_psi,
        "alert": alert,
    }


if __name__ == "__main__":
    interval = int(os.environ.get("MONITORING_INTERVAL_SECONDS", "86400"))  # 24h default
    monitoring_flow.serve(
        name="monitoring-default",
        interval=timedelta(seconds=interval),
        tags=["monitoring"],
    )
