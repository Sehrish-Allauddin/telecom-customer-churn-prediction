"""DataFrame validation for the IBM Telco source and cleaned training data."""

import pandas as pd
import pandera.pandas as pa
from pandera.errors import SchemaErrors

REQUIRED_DATA_COLUMNS = {
    "Churn",
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
CATEGORY_VALUES = {
    "gender": ["Female", "Male"],
    "Partner": ["Yes", "No"],
    "Dependents": ["Yes", "No"],
    "PhoneService": ["Yes", "No"],
    "MultipleLines": ["Yes", "No", "No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["Yes", "No", "No internet service"],
    "OnlineBackup": ["Yes", "No", "No internet service"],
    "DeviceProtection": ["Yes", "No", "No internet service"],
    "TechSupport": ["Yes", "No", "No internet service"],
    "StreamingTV": ["Yes", "No", "No internet service"],
    "StreamingMovies": ["Yes", "No", "No internet service"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["Yes", "No"],
    "PaymentMethod": [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ],
}


def _make_schema(columns: set[str]) -> pa.DataFrameSchema:
    """Build a non-strict schema for known columns present in this dataset."""
    schema_columns: dict[str, pa.Column] = {
        "Churn": pa.Column(int, checks=pa.Check.isin([0, 1]), nullable=False, coerce=True),
        "tenure": pa.Column(
            float, checks=[pa.Check.ge(0), pa.Check.le(72)], nullable=False, coerce=True
        ),
        "MonthlyCharges": pa.Column(
            float, checks=[pa.Check.ge(0), pa.Check.le(1000)], nullable=False, coerce=True
        ),
        "TotalCharges": pa.Column(
            float, checks=[pa.Check.ge(0), pa.Check.le(100000)], nullable=True, coerce=True
        ),
    }
    if "SeniorCitizen" in columns:
        schema_columns["SeniorCitizen"] = pa.Column(
            int, checks=pa.Check.isin([0, 1]), nullable=False, coerce=True
        )
    for column, allowed in CATEGORY_VALUES.items():
        if column in columns:
            schema_columns[column] = pa.Column(
                str, checks=pa.Check.isin(allowed), nullable=True, coerce=True
            )
    if "customerID" in columns:
        schema_columns["customerID"] = pa.Column(str, nullable=False, coerce=True)
    return pa.DataFrameSchema(schema_columns, strict=False, coerce=False)


def validate_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate required fields, values, types, and numeric ranges.

    Missing category values and TotalCharges are allowed because the existing
    sklearn pipeline imputes them. Missing target or core model fields are rejected.

    Args:
        frame: Cleaned frame with binary Churn target.

    Returns:
        A validated copy of the frame.

    Raises:
        ValueError: If the input is missing required columns or violates the schema.
    """
    missing = sorted(REQUIRED_DATA_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError("Data validation failed: missing required columns: " + ", ".join(missing))
    try:
        return _make_schema(set(frame.columns)).validate(frame, lazy=True)
    except SchemaErrors as error:
        failures = error.failure_cases
        details: list[str] = []
        if {"column", "check"}.issubset(failures.columns):
            unique_failures = failures[["column", "check"]].drop_duplicates()
            details = [
                f"{row.column}: {row.check}" for row in unique_failures.itertuples(index=False)
            ]
        summary = "; ".join(details) if details else "schema or value checks failed"
        raise ValueError(f"Data validation failed: {summary}") from error
