"""Dataset loading, schema normalization, and cleaning utilities."""

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from config import DATA_DIR, DATA_PATH, TARGET
from src.logging_config import configure_logging
from src.validation import validate_dataset

configure_logging()
logger = logging.getLogger(__name__)


def find_dataset(path: Path = DATA_PATH) -> Path:
    """Return the configured dataset or identify a CSV in data/."""
    if path.exists():
        return path
    candidates = sorted(p for p in DATA_DIR.glob("*.csv") if p.is_file())
    for candidate in candidates:
        try:
            if TARGET.lower() in {
                str(c).strip().lower() for c in pd.read_csv(candidate, nrows=0).columns
            }:
                return candidate
        except (OSError, pd.errors.ParserError, UnicodeDecodeError):
            continue
    raise FileNotFoundError(
        f"Dataset not found. Expected {path}; place the IBM Telco Customer Churn CSV in {DATA_DIR}."
    )


def normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize whitespace and match the churn target case-insensitively."""
    result = frame.copy()
    result.columns = [str(column).strip() for column in result.columns]
    target = next((c for c in result.columns if c.lower() == TARGET.lower()), None)
    if target is None:
        raise ValueError(
            f"Required target column {TARGET!r} is missing. Found: {list(result.columns)}"
        )
    if target != TARGET:
        result = result.rename(columns={target: TARGET})
    return result


def load_and_clean_data(path: Path | None = None, save_cleaned: bool = True) -> pd.DataFrame:
    """Load the actual CSV and apply conservative cleaning without discarding rows."""
    dataset_path = find_dataset(path or DATA_PATH)
    frame = normalize_columns(pd.read_csv(dataset_path))
    logger.info(
        "Loaded dataset with %s rows and %s columns from %s",
        frame.shape[0],
        frame.shape[1],
        dataset_path,
    )
    logger.info("Dataset columns: %s", list(frame.columns))
    logger.info("Dataset dtypes:\n%s", frame.dtypes.to_string())
    logger.info("Missing values by column:\n%s", frame.isna().sum().to_string())
    duplicates = int(frame.duplicated().sum())
    logger.info("Duplicated rows: %s", duplicates)
    # Exact duplicate records add no independent information; remove them only when present.
    if duplicates:
        frame = frame.drop_duplicates().copy()
    if "TotalCharges" in frame:
        # Telco source stores some blank totals as whitespace; coercion marks only those as missing.
        frame["TotalCharges"] = pd.to_numeric(
            frame["TotalCharges"].replace(r"^\s*$", pd.NA, regex=True), errors="coerce"
        )
    for column in frame.select_dtypes(include="object").columns:
        frame[column] = frame[column].map(
            lambda value: value.strip() if isinstance(value, str) else value
        )
    frame[TARGET] = frame[TARGET].astype("string").str.strip().str.lower().map({"yes": 1, "no": 0})
    if frame[TARGET].isna().any():
        invalid = frame.loc[frame[TARGET].isna(), TARGET].unique().tolist()
        raise ValueError(f"Target contains missing or unsupported values: {invalid}")
    frame = validate_dataset(frame)
    if save_cleaned:
        output = DATA_DIR / "cleaned_telco_churn.csv"
        frame.to_csv(output, index=False)
    return frame


def categorical_value_summary(frame: pd.DataFrame) -> dict[str, Any]:
    """Return categorical unique values for notebook and training diagnostics."""
    return {
        column: frame[column].dropna().unique().tolist()
        for column in frame.select_dtypes(include=["object", "category", "string"]).columns
    }
