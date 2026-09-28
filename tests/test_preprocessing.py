"""Tests for deterministic feature engineering and sklearn preprocessing."""

import numpy as np
import pandas as pd

from src.preprocessing import engineer_features, make_preprocessor


def test_engineered_features_and_identifier_removal(source_frame):
    """Feature engineering adds documented values and removes the identifier."""
    cleaned = source_frame.copy()
    cleaned["TotalCharges"] = pd.to_numeric(cleaned["TotalCharges"], errors="coerce")
    engineered = engineer_features(cleaned)
    assert "customerID" not in engineered
    assert engineered.loc[0, "tenure_group"] == "0-12"
    assert engineered.loc[0, "service_count"] == 2
    assert pd.isna(engineered.loc[0, "average_monthly_spend"])


def test_preprocessor_encodes_and_imputes_without_nan(source_frame):
    """One-hot output is numeric, includes expanded categories, and has no NaNs."""
    features = engineer_features(source_frame.drop(columns="Churn"))
    preprocessor = make_preprocessor(features)
    transformed = preprocessor.fit_transform(features)
    values = transformed.toarray() if hasattr(transformed, "toarray") else transformed
    assert values.shape[0] == len(features)
    assert values.shape[1] > features.shape[1]
    assert np.isfinite(values).all()
    assert any("Contract" in name for name in preprocessor.get_feature_names_out())
