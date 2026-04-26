"""Serve all PDM flows from a single worker process.

Replaces individual `python -m pdm.flows.training_flow` / `monitoring_flow`
entrypoints. Same effect, but only one container needed.
"""

from __future__ import annotations

import os
from datetime import timedelta

from prefect import serve

from pdm.flows.monitoring_flow import monitoring_flow
from pdm.flows.training_flow import training_flow

if __name__ == "__main__":
    training_interval = int(os.environ.get("TRAINING_INTERVAL_SECONDS", "21600"))    # 6h
    monitoring_interval = int(os.environ.get("MONITORING_INTERVAL_SECONDS", "86400"))  # 24h

    serve(
        training_flow.to_deployment(
            name="training-default",
            interval=timedelta(seconds=training_interval),
            tags=["training"],
        ),
        monitoring_flow.to_deployment(
            name="monitoring-default",
            interval=timedelta(seconds=monitoring_interval),
            tags=["monitoring"],
        ),
    )
