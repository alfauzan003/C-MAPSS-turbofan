"""Train an XGBoost RUL regressor and log everything to MLflow.

Caller passes a feature DataFrame with a `rul` column; this module:
  1. Splits into train/validation (engine-level split — engines do not leak).
  2. Fits XGBoost.
  3. Computes RMSE/MAE/C-MAPSS on validation.
  4. Logs params, metrics, the model, and a reference to the training-set parquet.
  5. Registers the model under the `pdm-rul` registered-model name.

Returns (mlflow_run_id, model_version, metrics).
"""

from __future__ import annotations

import enum
import time
from dataclasses import dataclass, field

import httpx
import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import xgboost as xgb
from mlflow.tracking import MlflowClient
from sklearn.model_selection import GroupShuffleSplit

from pdm.logging import get_logger
from pdm.models.evaluate import cmapss_score, mae, rmse
from pdm.models.registry import PRODUCTION_ALIAS

REGISTERED_MODEL_NAME = "pdm-rul"


@dataclass
class TrainingResult:
    run_id: str
    model_version: str
    metrics: dict[str, float] = field(default_factory=dict)


def _split_engine_groupwise(
    X: pd.DataFrame, y: pd.Series, engines: pd.Series, val_fraction: float = 0.2, seed: int = 42
):
    splitter = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
    train_idx, val_idx = next(splitter.split(X, y, groups=engines))
    return (
        X.iloc[train_idx], y.iloc[train_idx],
        X.iloc[val_idx], y.iloc[val_idx],
    )


def train_and_log(
    features_df: pd.DataFrame,
    *,
    parquet_uri: str,
    training_run_id: str,
    mlflow_tracking_uri: str,
    mlflow_experiment: str = "pdm-rul",
    xgb_params: dict | None = None,
) -> TrainingResult:
    log = get_logger("train")

    feature_cols = [
        c for c in features_df.columns
        if c not in ("engine_id", "cycle", "rul")
    ]
    X = features_df[feature_cols].astype("float64")
    y = features_df["rul"].astype("float64")
    engines = features_df["engine_id"]

    X_tr, y_tr, X_val, y_val = _split_engine_groupwise(X, y, engines)

    params = {
        "objective": "reg:squarederror",
        "n_estimators": 400,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "tree_method": "hist",
    }
    if xgb_params:
        params.update(xgb_params)

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(mlflow_experiment)

    with mlflow.start_run() as run:
        mlflow.log_params(params)
        mlflow.log_param("training_run_id", training_run_id)
        mlflow.log_param("training_set_uri", parquet_uri)
        mlflow.log_param("n_features", len(feature_cols))
        mlflow.log_param("n_train_rows", len(X_tr))
        mlflow.log_param("n_val_rows", len(X_val))

        model = xgb.XGBRegressor(**params)
        t0 = time.perf_counter()
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        train_seconds = time.perf_counter() - t0

        y_pred = model.predict(X_val)
        metrics = {
            "rmse": rmse(y_val.to_numpy(), y_pred),
            "mae": mae(y_val.to_numpy(), y_pred),
            "cmapss_score": cmapss_score(y_val.to_numpy(), y_pred),
            "train_seconds": float(train_seconds),
        }
        mlflow.log_metrics(metrics)

        importance = sorted(
            zip(feature_cols, model.feature_importances_), key=lambda kv: -kv[1]
        )[:20]
        mlflow.log_text(
            "\n".join(f"{name}\t{score:.6f}" for name, score in importance),
            artifact_file="feature_importance_top20.txt",
        )

        mlflow.xgboost.log_model(
            xgb_model=model,
            artifact_path="model",
            registered_model_name=REGISTERED_MODEL_NAME,
            input_example=X_val.head(2),
        )
        run_id = run.info.run_id

    client = mlflow.tracking.MlflowClient()
    versions = client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
    version = max((v.version for v in versions), key=int) if versions else "?"

    log.info(
        "training_complete",
        run_id=run_id,
        model_version=version,
        rmse=metrics["rmse"],
        mae=metrics["mae"],
    )
    return TrainingResult(run_id=run_id, model_version=str(version), metrics=metrics)


# ---------------------------------------------------------------------------
# Auto-promotion helpers
# ---------------------------------------------------------------------------


class PromoteDecision(enum.Enum):
    PROMOTE = "promote"
    HOLD = "hold"


def compare_and_promote_decision(
    new_rmse: float,
    current_production_rmse: float | None,
    improvement_threshold_pct: float = 2.0,
) -> PromoteDecision:
    """Pure-function policy: promote iff new beats current by >= threshold percent.

    If there is no current Production model, always promote.
    """
    if current_production_rmse is None:
        return PromoteDecision.PROMOTE
    improvement_pct = ((current_production_rmse - new_rmse) / current_production_rmse) * 100.0
    return PromoteDecision.PROMOTE if improvement_pct >= improvement_threshold_pct else PromoteDecision.HOLD


def get_current_production_rmse(
    tracking_uri: str, name: str = REGISTERED_MODEL_NAME
) -> float | None:
    """Return the `rmse` metric of the current champion model, or None if there isn't one."""
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    try:
        v = client.get_model_version_by_alias(name=name, alias=PRODUCTION_ALIAS)
    except Exception as e:
        log = get_logger("train")
        log.warning("get_champion_rmse_failed", error=str(e), name=name)
        return None
    run = client.get_run(v.run_id)
    rmse_metric = run.data.metrics.get("rmse")
    return float(rmse_metric) if rmse_metric is not None else None


def promote(version: str, tracking_uri: str, name: str = REGISTERED_MODEL_NAME) -> None:
    """Set the champion alias to `version`."""
    mlflow.set_tracking_uri(tracking_uri)
    MlflowClient().set_registered_model_alias(name=name, alias=PRODUCTION_ALIAS, version=version)


def trigger_reload(prediction_api_url: str) -> bool:
    """POST /reload-model. Returns True on success, False on failure (logged)."""
    log = get_logger("train")
    try:
        r = httpx.post(f"{prediction_api_url.rstrip('/')}/reload-model", timeout=10.0)
        if r.status_code == 200:
            log.info("reload_triggered", response=r.json())
            return True
        log.warning("reload_unexpected_status", status=r.status_code, body=r.text[:200])
        return False
    except httpx.HTTPError as e:
        log.warning("reload_http_error", error=str(e))
        return False
