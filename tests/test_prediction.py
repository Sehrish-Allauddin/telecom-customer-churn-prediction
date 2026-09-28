"""Tests for prediction formats and raw input validation."""

import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from config import FEATURES_VERSION, MODEL_VERSION
from src.predict import load_model, predict_batch, predict_customer
from src.train import build_model_pipeline


class FixedProbabilityModel:
    """Small deterministic estimator double for checking the inference contract."""

    def predict_proba(self, features):
        assert len(features) == 1
        return np.array([[0.2, 0.8]])


def test_valid_customer_returns_binary_label_and_probability(raw_customer):
    """A valid customer gets the expected prediction tuple."""
    prediction, probability = predict_customer(raw_customer, FixedProbabilityModel())
    assert prediction == 1
    assert 0 <= probability <= 1


def test_batch_prediction_returns_minimal_predictions(raw_customer):
    """Batch inference keeps optional IDs and returns only output fields."""
    import pandas as pd

    frame = pd.DataFrame([{**raw_customer, "customerID": "sample-1"}])
    result = predict_batch(frame, FixedProbabilityModel())
    assert result.columns.tolist() == [
        "customer_id",
        "churn_probability",
        "risk_level",
        "predicted_churn",
    ]
    assert result.loc[0, "customer_id"] == "sample-1"
    assert result.loc[0, "predicted_churn"] == 1


def test_batch_prediction_rejects_missing_schema(raw_customer):
    """Batch inference reports missing required fields before model invocation."""
    import pandas as pd

    raw_customer.pop("tenure")
    with pytest.raises(ValueError, match="missing required columns"):
        predict_batch(pd.DataFrame([raw_customer]), FixedProbabilityModel())


def test_customer_identifier_is_ignored_and_not_a_model_feature(raw_customer):
    """Existing callers may include customerID; inference continues to ignore it."""
    raw_customer["customerID"] = "example-id"
    prediction, probability = predict_customer(raw_customer, FixedProbabilityModel())
    assert prediction == 1
    assert probability == 0.8


def test_missing_customer_fields_are_rejected(raw_customer):
    """Inference rejects an incomplete payload before calling the estimator."""
    raw_customer.pop("tenure")
    with pytest.raises(ValueError, match="Missing required"):
        predict_customer(raw_customer, FixedProbabilityModel())


def test_out_of_range_numeric_input_is_rejected(raw_customer):
    """Inference rejects impossible tenure values."""
    raw_customer["tenure"] = -1
    with pytest.raises(ValueError, match="between 0 and 72"):
        predict_customer(raw_customer, FixedProbabilityModel())


def test_missing_total_charges_is_allowed_for_pipeline_imputation(raw_customer):
    """The source dataset's blank TotalCharges rows remain valid inference inputs."""
    raw_customer["TotalCharges"] = None
    _, probability = predict_customer(raw_customer, FixedProbabilityModel())
    assert probability == 0.8


def test_unsupported_category_is_rejected(raw_customer):
    """Inference rejects categories outside the dataset vocabulary."""
    raw_customer["Contract"] = "Lifetime"
    with pytest.raises(ValueError, match="Unsupported category"):
        predict_customer(raw_customer, FixedProbabilityModel())


def test_incompatible_model_metadata_fails_before_deserialization(tmp_path):
    """A model with a mismatched version is rejected rather than used."""
    model_path = tmp_path / "model.pkl"
    metadata_path = tmp_path / "metadata.json"
    model_path.write_bytes(b"not deserialized because compatibility fails first")
    metadata_path.write_text(
        json.dumps(
            {
                "model_version": "999.0.0",
                "features_version": FEATURES_VERSION,
                "model_name": "Old Model",
                "model_type": "LogisticRegression",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="Model version is incompatible"):
        load_model(model_path, metadata_path)


def test_compatible_model_metadata_loads_artifact(tmp_path):
    """A valid pipeline and matching metadata deserialize successfully."""
    features = pd.DataFrame(
        {
            "tenure": list(range(1, 21)),
            "Contract": ["Month-to-month", "One year"] * 10,
        }
    )
    labels = np.array([0, 1] * 10)
    model = build_model_pipeline(features, LogisticRegression(max_iter=500))
    model.fit(features, labels)
    model_path = tmp_path / "model.pkl"
    metadata_path = tmp_path / "metadata.json"
    joblib.dump(model, model_path)
    metadata_path.write_text(
        json.dumps(
            {
                "model_version": MODEL_VERSION,
                "features_version": FEATURES_VERSION,
                "model_name": "Logistic Regression",
                "model_type": "LogisticRegression",
                "features": list(features.columns),
            }
        ),
        encoding="utf-8",
    )
    loaded = load_model(model_path, metadata_path)
    np.testing.assert_allclose(loaded.predict_proba(features), model.predict_proba(features))
