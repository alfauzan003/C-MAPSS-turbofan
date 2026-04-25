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

import time
from dataclasses import dataclass, field

import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GroupShuffleSplit

from pdm.logging import get_logger
from pdm.models.evaluate import cmapss_score, mae, rmse

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
    latest = client.get_latest_versions(name=REGISTERED_MODEL_NAME, stages=["None"])
    version = latest[0].version if latest else "?"

    log.info(
        "training_complete",
        run_id=run_id,
        model_version=version,
        rmse=metrics["rmse"],
        mae=metrics["mae"],
    )
    return TrainingResult(run_id=run_id, model_version=str(version), metrics=metrics)
