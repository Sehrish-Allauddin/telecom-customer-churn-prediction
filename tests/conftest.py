"""Shared, small synthetic data fixtures used by unit tests."""

import pandas as pd
import pytest


@pytest.fixture
def source_frame() -> pd.DataFrame:
    """Return a small telco-like source frame with a blank total charge."""
    return pd.DataFrame(
        {
            "customerID": [f"T{i}" for i in range(12)],
            "gender": ["Female", "Male"] * 6,
            "SeniorCitizen": [0, 1] * 6,
            "Partner": ["Yes", "No"] * 6,
            "Dependents": ["No", "Yes"] * 6,
            "tenure": list(range(1, 13)),
            "PhoneService": ["Yes"] * 12,
            "MultipleLines": ["No", "Yes"] * 6,
            "InternetService": ["DSL", "Fiber optic"] * 6,
            "OnlineSecurity": ["Yes", "No"] * 6,
            "OnlineBackup": ["No", "Yes"] * 6,
            "DeviceProtection": ["No", "Yes"] * 6,
            "TechSupport": ["Yes", "No"] * 6,
            "StreamingTV": ["No", "Yes"] * 6,
            "StreamingMovies": ["No", "Yes"] * 6,
            "Contract": ["Month-to-month", "One year"] * 6,
            "PaperlessBilling": ["Yes", "No"] * 6,
            "PaymentMethod": ["Electronic check", "Mailed check"] * 6,
            "MonthlyCharges": [30.0 + i for i in range(12)],
            "TotalCharges": [" "] + [str((30 + i) * (i + 1)) for i in range(1, 12)],
            "Churn": ["No", "Yes"] * 6,
        }
    )


@pytest.fixture
def raw_customer() -> dict:
    """Return one complete valid customer payload for inference tests."""
    return {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "Yes",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.0,
        "TotalCharges": 840.0,
    }
