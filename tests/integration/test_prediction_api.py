"""Integration tests for prediction-api.

Strategy: bypass MLflow by injecting a fake PredictService onto app.state.
This isolates the API logic from MLflow's load path.
"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine, text

from pdm.apis.prediction_api import app
from pdm.predict import PredictService

SENSOR_COLS = [f"sensor_{i}" for i in range(1, 22)]


class _FakeModel:
    def predict(self, X):
        import numpy as np
        return np.full(len(X), 42.0)


def _row(cycle: int = 1) -> dict:
    base = {
        "engine_id": 7,
        "cycle": cycle,
        "op_setting_1": 0.0,
        "op_setting_2": 0.0,
        "op_setting_3": 100.0,
    }
    for i in range(1, 22):
        base[f"sensor_{i}"] = float(i)
    return base


@pytest.fixture
def client(db_engine: Engine) -> TestClient:
    """TestClient with a fake PredictService preinstalled (skips MLflow load)."""
    fake_svc = PredictService(model=_FakeModel(), model_name="pdm-rul", model_version="test")
    with TestClient(app) as c:
        c.app.state.predict_service = fake_svc
        c.app.state.startup_error = None
        yield c


@pytest.mark.integration
def test_predict_returns_response_and_logs_to_db(client: TestClient, db_engine: Engine):
    with db_engine.begin() as c:
        c.execute(text("TRUNCATE TABLE predictions.served RESTART IDENTITY"))

    payload = {"readings": [_row(cycle=c) for c in range(1, 6)]}
    r = client.post("/predict", json=payload)
    assert r.status_code == 200, r.text

    body = r.json()
    assert body["engine_id"] == 7
    assert body["predicted_rul"] == 42.0
    assert body["model_name"] == "pdm-rul"
    assert body["model_version"] == "test"
    assert body["n_input_rows"] == 5
    assert body["latency_ms"] >= 0

    with db_engine.connect() as c:
        row = c.execute(
            text(
                "SELECT engine_id, predicted_rul, model_version, n_input_rows "
                "FROM predictions.served"
            )
        ).fetchone()
    assert row.engine_id == 7
    assert row.predicted_rul == 42.0
    assert row.model_version == "test"
    assert row.n_input_rows == 5


@pytest.mark.integration
def test_predict_rejects_mixed_engines(client: TestClient):
    payload = {
        "readings": [
            _row(cycle=1),
            {**_row(cycle=2), "engine_id": 99},
        ]
    }
    r = client.post("/predict", json=payload)
    # After the fix in Task 4: ValueError from mixed engine IDs → HTTP 422
    assert r.status_code == 422


@pytest.mark.integration
def test_predict_rejects_empty_readings(client: TestClient):
    r = client.post("/predict", json={"readings": []})
    assert r.status_code == 422


@pytest.mark.integration
def test_predict_returns_503_when_no_model_loaded(db_engine: Engine):
    with TestClient(app) as c:
        c.app.state.predict_service = None
        c.app.state.startup_error = "test: no model"
        r = c.post("/predict", json={"readings": [_row()]})
    assert r.status_code == 503
    assert "no model" in r.json()["detail"]


@pytest.mark.integration
def test_health_returns_ok_when_model_loaded(client: TestClient):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


@pytest.mark.integration
def test_health_returns_degraded_when_no_model(db_engine: Engine):
    with TestClient(app) as c:
        c.app.state.predict_service = None
        c.app.state.startup_error = "x"
        r = c.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "degraded"
    assert "model" in body["detail"]
