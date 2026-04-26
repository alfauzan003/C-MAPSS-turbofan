"""Integration test: /metrics page renders without error."""

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine, text

from pdm.apis.prediction_api import app
from pdm.predict import PredictService


class _FakeModel:
    def predict(self, X):
        return np.full(len(X), 50.0)


@pytest.fixture
def client(db_engine: Engine) -> TestClient:
    """TestClient with a fake PredictService preinstalled (skips MLflow load)."""
    svc = PredictService(model=_FakeModel(), model_name="pdm-rul", model_version="test")
    with TestClient(app) as c:
        c.app.state.predict_service = svc
        c.app.state.startup_error = None
        yield c


@pytest.mark.integration
def test_metrics_page_renders_when_no_data(client: TestClient, db_engine: Engine):
    with db_engine.begin() as c:
        c.execute(text("TRUNCATE TABLE predictions.served RESTART IDENTITY"))
        c.execute(text("TRUNCATE TABLE predictions.drift_reports RESTART IDENTITY"))
    r = client.get("/metrics")
    assert r.status_code == 200
    body = r.text
    assert "PDM Prediction API" in body
    assert "No drift report yet" in body


@pytest.mark.integration
def test_metrics_page_renders_with_data(client: TestClient, db_engine: Engine):
    with db_engine.begin() as c:
        c.execute(text("TRUNCATE TABLE predictions.served RESTART IDENTITY"))
        c.execute(text("TRUNCATE TABLE predictions.drift_reports RESTART IDENTITY"))
        # 3 served rows
        for _ in range(3):
            c.execute(text("""
                INSERT INTO predictions.served
                    (engine_id, predicted_rul, model_name, model_version,
                     input_fingerprint, n_input_rows, latency_ms)
                VALUES (1, 50.0, 'pdm-rul', 'test', 'fp', 5, 12.5)
            """))
        # 1 drift report
        c.execute(text("""
            INSERT INTO predictions.drift_reports
                (model_name, model_version, window_start, window_end,
                 n_baseline_rows, n_compare_rows, psi_per_feature, max_psi, alert)
            VALUES (
                'pdm-rul', 'test',
                NOW() - INTERVAL '24 hours', NOW(),
                500, 200,
                '{"sensor_3": 0.05, "sensor_7": 0.31}'::json,
                0.31, true
            )
        """))

    r = client.get("/metrics")
    assert r.status_code == 200
    body = r.text
    assert "sensor_7" in body
    assert "0.310" in body
    assert "YES" in body
