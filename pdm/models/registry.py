"""Helpers around the MLflow Model Registry.

`load_production` returns the model object currently in the "Production"
stage for the registered name (raises if none exists).

`promote_to_production` transitions the given version to "Production" and
archives any previous Production version.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlflow
from mlflow.tracking import MlflowClient

from pdm.logging import get_logger

REGISTERED_MODEL_NAME = "pdm-rul"


@dataclass
class LoadedModel:
    model: object
    version: str
    run_id: str


def load_production(tracking_uri: str, name: str = REGISTERED_MODEL_NAME) -> LoadedModel:
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    versions = client.get_latest_versions(name=name, stages=["Production"])
    if not versions:
        raise LookupError(f"No model in Production for {name!r}")
    v = versions[0]
    model = mlflow.pyfunc.load_model(model_uri=f"models:/{name}/{v.version}")
    return LoadedModel(model=model, version=v.version, run_id=v.run_id)


def promote_to_production(
    version: str,
    tracking_uri: str,
    name: str = REGISTERED_MODEL_NAME,
) -> None:
    log = get_logger("registry")
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    client.transition_model_version_stage(
        name=name,
        version=version,
        stage="Production",
        archive_existing_versions=True,
    )
    log.info("promoted_to_production", name=name, version=version)
