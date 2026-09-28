"""Load the fitted pipeline and predict churn from raw customer fields."""

import json
import logging
import math
import pickle
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from config import CHURN_THRESHOLD, FEATURES_VERSION, METADATA_PATH, MODEL_PATH, MODEL_VERSION
from src.logging_config import configure_logging
from src.preprocessing import engineer_features

REQUIRED_CUSTOMER_FIELDS = {
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
}
CUSTOMER_FEATURE_ORDER = (
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
)
ALLOWED_CATEGORIES = {
    "gender": {"Female", "Male"},
    "Partner": {"Yes", "No"},
    "Dependents": {"Yes", "No"},
    "PhoneService": {"Yes", "No"},
    "MultipleLines": {"Yes", "No", "No phone service"},
    "InternetService": {"DSL", "Fiber optic", "No"},
    "OnlineSecurity": {"Yes", "No", "No internet service"},
    "OnlineBackup": {"Yes", "No", "No internet service"},
    "DeviceProtection": {"Yes", "No", "No internet service"},
    "TechSupport": {"Yes", "No", "No internet service"},
    "StreamingTV": {"Yes", "No", "No internet service"},
    "StreamingMovies": {"Yes", "No", "No internet service"},
    "Contract": {"Month-to-month", "One year", "Two year"},
    "PaperlessBilling": {"Yes", "No"},
    "PaymentMethod": {
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    },
}
configure_logging()
logger = logging.getLogger(__name__)


def validate_customer_input(customer: Mapping[str, Any]) -> None:
    """Reject incomplete or out-of-domain customer values before inference.

    Args:
        customer: Raw feature values supplied to the model.

    Raises:
        ValueError: If required fields, categories, or numeric ranges are invalid.
    """
    missing = sorted(REQUIRED_CUSTOMER_FIELDS - customer.keys())
    if missing:
        raise ValueError(f"Missing required customer fields: {', '.join(missing)}")
    unexpected = sorted(customer.keys() - REQUIRED_CUSTOMER_FIELDS - {"customerID"})
    if unexpected:
        raise ValueError(f"Unexpected customer fields: {', '.join(unexpected)}")
    invalid_categories = [
        f"{column}={customer[column]!r}"
        for column, allowed in ALLOWED_CATEGORIES.items()
        if customer[column] not in allowed
    ]
    if invalid_categories:
        raise ValueError("Unsupported category values: " + ", ".join(invalid_categories))
    service_fields = (
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    )
    if customer["InternetService"] == "No" and any(
        customer[field] != "No internet service" for field in service_fields
    ):
        raise ValueError(
            "Internet-dependent services must be 'No internet service' "
            "when InternetService is No."
        )
    if customer["InternetService"] != "No" and any(
        customer[field] == "No internet service" for field in service_fields
    ):
        raise ValueError(
            "Use Yes or No for internet-dependent services when InternetService is active."
        )
    if customer["PhoneService"] == "No" and customer["MultipleLines"] != "No phone service":
        raise ValueError("MultipleLines must be 'No phone service' when PhoneService is No.")
    if customer["PhoneService"] == "Yes" and customer["MultipleLines"] == "No phone service":
        raise ValueError("MultipleLines cannot be 'No phone service' when PhoneService is Yes.")
    total_charges = customer["TotalCharges"]
    if total_charges is None or (
        isinstance(total_charges, (float, int)) and math.isnan(float(total_charges))
    ):
        total_charges = None  # The fitted numeric imputer handles this known source-data case.
    for column in ("tenure", "MonthlyCharges", "SeniorCitizen"):
        try:
            number = float(customer[column])
        except (TypeError, ValueError) as error:
            raise ValueError(f"{column} must be numeric.") from error
        if not math.isfinite(number):
            raise ValueError(f"{column} must be a finite number.")
        if column == "tenure" and not 0 <= number <= 72:
            raise ValueError("tenure must be between 0 and 72 months.")
        if column == "tenure" and not number.is_integer():
            raise ValueError("tenure must be a whole number of months.")
        if column == "MonthlyCharges" and not 0 <= number <= 1000:
            raise ValueError("MonthlyCharges must be between 0 and 1000.")
        if column == "TotalCharges" and not 0 <= number <= 100000:
            raise ValueError("TotalCharges must be between 0 and 100000.")
        if column == "SeniorCitizen" and number not in {0, 1}:
            raise ValueError("SeniorCitizen must be 0 or 1.")
    if total_charges is not None:
        try:
            total_number = float(total_charges)
        except (TypeError, ValueError) as error:
            raise ValueError("TotalCharges must be numeric or missing.") from error
        if not math.isfinite(total_number) or not 0 <= total_number <= 100000:
            raise ValueError("TotalCharges must be between 0 and 100000, or missing.")


def load_model(path: Path = MODEL_PATH, metadata_path: Path = METADATA_PATH) -> Pipeline:
    """Load the complete preprocessing and estimator pipeline.

    Args:
        path: Pickle file containing a fitted imbalanced-learn pipeline.
        metadata_path: JSON metadata paired with the fitted model.

    Returns:
        The deserialized fitted pipeline.

    Raises:
        FileNotFoundError: If the requested model file does not exist.
        RuntimeError: If metadata is missing, invalid, or incompatible.
    """
    if not path.exists():
        raise FileNotFoundError(f"Model not found at {path}. Run `python -m src.train` first.")
    if not metadata_path.exists():
        raise RuntimeError(f"Model metadata not found at {metadata_path}; retrain the model.")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError("Model metadata is unreadable; retrain the model.") from error
    if not isinstance(metadata, dict):
        raise RuntimeError("Model metadata is invalid; retrain the model.")
    if metadata.get("model_version") != MODEL_VERSION:
        raise RuntimeError(
            "Model version is incompatible with this application; retrain the model."
        )
    if metadata.get("features_version") != FEATURES_VERSION:
        raise RuntimeError(
            "Model feature version is incompatible with this application; retrain the model."
        )
    if not metadata.get("model_name") or not metadata.get("model_type"):
        raise RuntimeError("Model metadata is incomplete; retrain the model.")
    try:
        model = joblib.load(path)
    except (
        OSError,
        EOFError,
        ImportError,
        AttributeError,
        ValueError,
        TypeError,
        pickle.UnpicklingError,
    ) as error:
        logger.exception("Saved model artifact could not be loaded.")
        raise RuntimeError(
            "Saved model artifact is unreadable or incompatible; retrain the model."
        ) from error
    actual_type = (
        model.named_steps.get("model").__class__.__name__ if hasattr(model, "named_steps") else None
    )
    if actual_type != metadata["model_type"]:
        raise RuntimeError("Saved model type does not match metadata; retrain the model.")
    model_features = list(getattr(model, "feature_names_in_", []))
    if not metadata.get("features") or model_features != metadata["features"]:
        raise RuntimeError("Saved model feature schema does not match metadata; retrain the model.")
    logger.info("Loaded model version %s (%s)", MODEL_VERSION, actual_type)
    return model


def predict_customer(
    customer: Mapping[str, Any], model: Pipeline | None = None
) -> tuple[int, float]:
    """Return a binary churn prediction and probability for one raw customer.

    Args:
        customer: Raw feature names and values for one customer.
        model: Optional fitted pipeline, useful for repeated predictions and tests.

    Returns:
        A tuple containing the 0/1 prediction and churn probability.
    """
    validate_customer_input(customer)
    fitted_model = model or load_model()
    row = engineer_features(pd.DataFrame([customer]))
    probability = float(fitted_model.predict_proba(row)[:, 1][0])
    return int(probability >= CHURN_THRESHOLD), probability


def predict_batch(frame: pd.DataFrame, model: Pipeline | Any | None = None) -> pd.DataFrame:
    """Validate and score raw customer rows, returning only prediction fields.

    An optional ``customerID`` column is echoed as the result key. The source target
    column is ignored so it cannot become an inference feature.
    """
    from config import HIGH_RISK_THRESHOLD, MEDIUM_RISK_THRESHOLD

    reserved = {"customerID", "Churn"}
    missing = sorted(REQUIRED_CUSTOMER_FIELDS - set(frame.columns))
    unexpected = sorted(set(frame.columns) - REQUIRED_CUSTOMER_FIELDS - reserved)
    if missing:
        raise ValueError("Batch file is missing required columns: " + ", ".join(missing))
    if unexpected:
        raise ValueError("Batch file has unsupported columns: " + ", ".join(unexpected))
    records = frame[list(CUSTOMER_FEATURE_ORDER)].to_dict(orient="records")
    for row_number, customer in enumerate(records, start=1):
        try:
            validate_customer_input(customer)
        except ValueError as error:
            raise ValueError(f"Invalid customer row {row_number}: {error}") from error
    fitted_model = model or load_model()
    feature_frame = engineer_features(frame[list(CUSTOMER_FEATURE_ORDER)])
    probabilities = fitted_model.predict_proba(feature_frame)[:, 1]
    identifiers = (
        frame["customerID"].astype("string").where(frame["customerID"].notna(), None).tolist()
        if "customerID" in frame
        else [f"row_{number:06d}" for number in range(1, len(frame) + 1)]
    )
    predictions = (probabilities >= CHURN_THRESHOLD).astype(int)
    risks = [
        (
            "High Risk"
            if probability >= HIGH_RISK_THRESHOLD
            else "Medium Risk" if probability >= MEDIUM_RISK_THRESHOLD else "Low Risk"
        )
        for probability in probabilities
    ]
    return pd.DataFrame(
        {
            "customer_id": identifiers,
            "churn_probability": probabilities.astype(float),
            "risk_level": risks,
            "predicted_churn": predictions.astype(int),
        }
    )
