"""Integration tests for ingestion-api against a real Postgres (pdm_test)."""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine, text

from pdm.apis.ingestion_api import app


def _payload(**overrides):
    base = {
        "engine_id": 1,
        "cycle": 1,
        "op_setting_1": -0.0007,
        "op_setting_2": -0.0004,
        "op_setting_3": 100.0,
    }
    for i in range(1, 22):
        base[f"sensor_{i}"] = float(i)
    base.update(overrides)
    return base


@pytest.fixture
def client(db_engine: Engine) -> TestClient:
    """TestClient that talks to the FastAPI app, which itself talks to pdm_test."""
    # db_engine fixture has already pointed Settings at pdm_test
    return TestClient(app)


@pytest.mark.integration
def test_health_returns_ok(client: TestClient):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


@pytest.mark.integration
def test_post_sensor_reading_persists(client: TestClient, db_engine: Engine):
    # Clean slate
    with db_engine.begin() as c:
        c.execute(text("TRUNCATE TABLE raw_sensor.readings RESTART IDENTITY"))

    r = client.post("/sensor-readings", json=_payload())
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["engine_id"] == 1
    assert body["cycle"] == 1
    assert body["id"] >= 1

    with db_engine.connect() as c:
        n = c.execute(text("SELECT count(*) FROM raw_sensor.readings")).scalar_one()
    assert n == 1


@pytest.mark.integration
def test_post_sensor_reading_duplicate_returns_409(client: TestClient, db_engine: Engine):
    with db_engine.begin() as c:
        c.execute(text("TRUNCATE TABLE raw_sensor.readings RESTART IDENTITY"))

    r1 = client.post("/sensor-readings", json=_payload())
    assert r1.status_code == 201
    r2 = client.post("/sensor-readings", json=_payload())
    assert r2.status_code == 409
    assert "engine_id=1" in r2.json()["detail"]


@pytest.mark.integration
def test_post_sensor_reading_invalid_payload_returns_422(client: TestClient):
    r = client.post("/sensor-readings", json={"engine_id": 1})  # missing fields
    assert r.status_code == 422
    body = r.json()
    assert any("cycle" in str(err["loc"]) for err in body["detail"])
