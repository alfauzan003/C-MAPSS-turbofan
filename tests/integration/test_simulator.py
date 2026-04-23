"""Integration test: post_one talks to the real ingestion-api app via TestClient."""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine, text

from pdm.apis.ingestion_api import app
from pdm.simulator.run import CMAPSS_COLUMNS, post_one


class _ClientAdapter:
    """Adapter so post_one (which calls .post(url, json=)) works with TestClient.

    TestClient.post takes a path, not a full URL; we strip the host portion.
    """

    def __init__(self, tc: TestClient):
        self.tc = tc

    def post(self, url: str, json: dict, timeout: float = 5.0):
        path = url.split("://", 1)[-1].split("/", 1)[-1]
        return self.tc.post("/" + path, json=json)


@pytest.mark.integration
def test_simulator_post_writes_row(db_engine: Engine):
    with db_engine.begin() as c:
        c.execute(text("TRUNCATE TABLE raw_sensor.readings RESTART IDENTITY"))

    payload = {col: 0.0 for col in CMAPSS_COLUMNS}
    payload["engine_id"] = 7
    payload["cycle"] = 3
    payload["op_setting_3"] = 100.0

    import structlog
    log = structlog.get_logger("test")
    with TestClient(app) as tc:
        post_one(_ClientAdapter(tc), "http://x/sensor-readings", payload, log)

    with db_engine.connect() as c:
        n = c.execute(
            text("SELECT count(*) FROM raw_sensor.readings WHERE engine_id = 7")
        ).scalar_one()
    assert n == 1
