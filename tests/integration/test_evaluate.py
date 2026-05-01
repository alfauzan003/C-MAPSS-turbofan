"""Integration test for GET /evaluate."""
import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine

from pdm.apis.prediction_api import app
from pdm.predict import PredictService


class _FakeModel:
    def predict(self, X):
        return np.full(len(X), 50.0)


@pytest.fixture
def client(db_engine: Engine) -> TestClient:
    fake_svc = PredictService(model=_FakeModel(), model_name="pdm-rul", model_version="test")
    with TestClient(app) as c:
        c.app.state.predict_service = fake_svc
        c.app.state.startup_error = None
        yield c


@pytest.mark.integration
def test_evaluate_returns_html(client: TestClient):
    r = client.get("/evaluate")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]


@pytest.mark.integration
def test_evaluate_contains_metric_cards(client: TestClient):
    r = client.get("/evaluate")
    body = r.text
    assert "RMSE" in body
    assert "MAE" in body
    assert "C-MAPSS" in body


@pytest.mark.integration
def test_evaluate_contains_chart_canvases(client: TestClient):
    r = client.get("/evaluate")
    body = r.text
    assert "scatterChart" in body
    assert "degradationChart" in body


@pytest.mark.integration
def test_evaluate_degraded_when_no_model(db_engine: Engine):
    with TestClient(app) as c:
        c.app.state.predict_service = None
        c.app.state.startup_error = "no model"
        r = c.get("/evaluate")
    assert r.status_code == 200
    assert "not available" in r.text.lower() or "no model" in r.text.lower()
