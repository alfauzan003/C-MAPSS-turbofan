"""Helpers around the MLflow Model Registry.

`load_production` loads the model version aliased as "champion".
`promote_to_production` sets the "champion" alias on the given version.

MLflow v3 removed stage-based transitions; this module uses aliases instead.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlflow
from mlflow.tracking import MlflowClient

from pdm.logging import get_logger

REGISTERED_MODEL_NAME = "pdm-rul"
PRODUCTION_ALIAS = "champion"


@dataclass
class LoadedModel:
    model: object
    version: str
    run_id: str


def load_production(tracking_uri: str, name: str = REGISTERED_MODEL_NAME) -> LoadedModel:
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    try:
        v = client.get_model_version_by_alias(name=name, alias=PRODUCTION_ALIAS)
    except mlflow.exceptions.MlflowException as exc:
        raise LookupError(f"No model with alias '{PRODUCTION_ALIAS}' for {name!r}") from exc
    model = mlflow.pyfunc.load_model(model_uri=f"models:/{name}@{PRODUCTION_ALIAS}")
    return LoadedModel(model=model, version=v.version, run_id=v.run_id)


def promote_to_production(
    version: str,
    tracking_uri: str,
    name: str = REGISTERED_MODEL_NAME,
) -> None:
    log = get_logger("registry")
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    client.set_registered_model_alias(name=name, alias=PRODUCTION_ALIAS, version=version)
    log.info("promoted_to_production", name=name, version=version, alias=PRODUCTION_ALIAS)
