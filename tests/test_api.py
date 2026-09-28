"""FastAPI contract tests using a deterministic in-memory estimator."""

import numpy as np
from fastapi.testclient import TestClient

from api import main as api_main


class FixedProbabilityModel:
    """Small estimator double that makes API response assertions reproducible."""

    def predict_proba(self, features):
        assert len(features) == 1
        return np.array([[0.22, 0.78]])


class BatchProbabilityModel:
    """Deterministic multi-row estimator for the batch endpoint contract."""

    def predict_proba(self, features):
        return np.tile(np.array([[0.22, 0.78]]), (len(features), 1))


def test_health_endpoint_reports_available_model(monkeypatch):
    """Health confirms model availability without exposing customer data."""
    monkeypatch.setattr(api_main, "get_pipeline", lambda: FixedProbabilityModel())
    response = TestClient(api_main.app).get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_model_info_includes_versioned_operating_point(monkeypatch):
    """Model metadata endpoint adds CV and threshold details without breaking identity fields."""
    monkeypatch.setattr(
        api_main,
        "read_metadata",
        lambda: {
            "model_name": "test",
            "model_version": "test-version",
            "model_type": "TestEstimator",
            "features_version": "feature-version",
            "metrics": {"PR_AUC": 0.6},
            "training_date": "2026-01-01",
            "threshold": {"value": 0.5},
            "cross_validation": {"folds": 5},
        },
    )
    response = TestClient(api_main.app).get("/model-info")
    assert response.status_code == 200
    assert response.json()["cross_validation"]["folds"] == 5
    assert response.json()["threshold"]["value"] == 0.5


def test_predict_endpoint_returns_probability_and_risk(monkeypatch):
    """Valid requests return the documented prediction contract."""
    monkeypatch.setattr(api_main, "get_pipeline", lambda: FixedProbabilityModel())
    monkeypatch.setattr(api_main, "log_prediction", lambda *args, **kwargs: None)
    response = TestClient(api_main.app).post(
        "/predict",
        json={"tenure": 12, "monthly_charges": 75.5, "contract": "Month-to-month"},
    )
    assert response.status_code == 200
    assert response.json()["prediction"] == 1
    assert response.json()["churn_probability"] == 0.78
    assert response.json()["risk_level"] == "High Risk"
    assert response.json()["model_version"]


def test_predict_endpoint_rejects_invalid_input(monkeypatch):
    """Pydantic validation returns 422 for out-of-range input."""
    monkeypatch.setattr(api_main, "get_pipeline", lambda: FixedProbabilityModel())
    response = TestClient(api_main.app).post("/predict", json={"tenure": -1})
    assert response.status_code == 422


def test_predict_endpoint_defaults_services_for_no_internet(monkeypatch):
    """The API applies compatible service defaults when the customer has no internet."""
    monkeypatch.setattr(api_main, "get_pipeline", lambda: FixedProbabilityModel())
    monkeypatch.setattr(api_main, "log_prediction", lambda *args, **kwargs: None)
    response = TestClient(api_main.app).post("/predict", json={"internet_service": "No"})
    assert response.status_code == 200


def test_batch_endpoint_returns_minimal_rows_and_model_version(monkeypatch):
    """Batch API preserves optional IDs while hiding submitted feature payloads."""
    monkeypatch.setattr(api_main, "get_pipeline", lambda: BatchProbabilityModel())
    monkeypatch.setattr(api_main, "log_prediction", lambda *args, **kwargs: None)
    response = TestClient(api_main.app).post(
        "/predict/batch",
        json={
            "customers": [
                {"customer_id": "account-1", "tenure": 12},
                {"customer_id": "account-2", "tenure": 24},
            ]
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["predictions"][0]["customer_id"] == "account-1"
    assert payload["predictions"][0]["prediction"] == 1
    assert "tenure" not in payload["predictions"][0]
    assert payload["model_version"]


def test_batch_endpoint_limits_customer_count():
    """Batch size is bounded at request validation to protect API resources."""
    response = TestClient(api_main.app).post("/predict/batch", json={"customers": []})
    assert response.status_code == 422
