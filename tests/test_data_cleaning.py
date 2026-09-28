"""Tests for conservative input cleaning and target validation."""

import pandas as pd
import pytest

from src.data_cleaning import load_and_clean_data, normalize_columns
from src.validation import validate_dataset


def test_cleaning_converts_target_and_blank_total_charges(tmp_path, source_frame):
    """Blank totals become missing while target values become binary."""
    path = tmp_path / "source.csv"
    source_frame.to_csv(path, index=False)
    cleaned = load_and_clean_data(path, save_cleaned=False)
    assert cleaned.shape == source_frame.shape
    assert set(cleaned["Churn"].unique()) == {0, 1}
    assert pd.isna(cleaned.loc[0, "TotalCharges"])


def test_cleaning_drops_only_exact_duplicates(tmp_path, source_frame):
    """Exact duplicate rows are removed without dropping other records."""
    path = tmp_path / "duplicated.csv"
    pd.concat([source_frame, source_frame.iloc[[0]]], ignore_index=True).to_csv(path, index=False)
    cleaned = load_and_clean_data(path, save_cleaned=False)
    assert len(cleaned) == len(source_frame)


def test_invalid_target_is_rejected(tmp_path, source_frame):
    """Unsupported target labels fail clearly instead of becoming silent classes."""
    source_frame.loc[0, "Churn"] = "Maybe"
    path = tmp_path / "bad_target.csv"
    source_frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="unsupported values"):
        load_and_clean_data(path, save_cleaned=False)


def test_missing_target_column_is_rejected(source_frame):
    """A missing target is an invalid training frame."""
    with pytest.raises(ValueError, match="target column"):
        normalize_columns(source_frame.drop(columns="Churn"))


def test_pandera_rejects_invalid_category_and_numeric_range(source_frame):
    """Unexpected service categories and negative tenure do not enter training."""
    frame = source_frame.copy()
    frame["Churn"] = frame["Churn"].map({"No": 0, "Yes": 1})
    frame["TotalCharges"] = pd.to_numeric(frame["TotalCharges"], errors="coerce")
    frame.loc[0, "Contract"] = "Lifetime"
    frame.loc[1, "tenure"] = -4
    with pytest.raises(ValueError, match="Data validation failed"):
        validate_dataset(frame)


def test_pandera_reports_missing_required_fields(source_frame):
    """Core modeling fields are required by the project data schema."""
    with pytest.raises(ValueError, match="missing required columns"):
        validate_dataset(source_frame.drop(columns=["Churn", "Contract"]))
