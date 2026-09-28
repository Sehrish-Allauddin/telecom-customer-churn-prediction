"""Privacy-conscious JSONL logging for prediction outcome monitoring."""

import json
import logging
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import MODEL_VERSION, PREDICTION_LOG_PATH

logger = logging.getLogger(__name__)
MONITORING_FEATURES = (
    "tenure",
    "Contract",
    "InternetService",
    "PaymentMethod",
    "TechSupport",
    "MonthlyCharges",
    "TotalCharges",
)


def log_prediction(
    customer: Mapping[str, Any],
    prediction: int,
    probability: float,
    risk_level: str,
    destination: Path = PREDICTION_LOG_PATH,
) -> None:
    """Append a prediction event without IDs, contact details, or demographics.

    Args:
        customer: Input values; only the allow-listed non-PII monitoring fields are logged.
        prediction: Binary model result.
        probability: Estimated churn probability.
        risk_level: Configured risk band.
        destination: JSON Lines file path for future drift analysis.
    """
    safe_features = {
        feature: customer[feature] for feature in MONITORING_FEATURES if feature in customer
    }
    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_version": MODEL_VERSION,
        "prediction": int(prediction),
        "churn_probability": float(probability),
        "risk_level": risk_level,
        "features": safe_features,
    }
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(event, separators=(",", ":"), default=str) + "\n")
    except OSError:
        logger.exception("Could not append a prediction monitoring event.")


def read_prediction_events(path: Path = PREDICTION_LOG_PATH) -> list[dict[str, Any]]:
    """Read valid JSONL prediction events; malformed lines are skipped safely."""
    if not path.exists():
        return []
    events = []
    try:
        with path.open(encoding="utf-8") as log_file:
            for line in log_file:
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(value, dict) and isinstance(value.get("features"), dict):
                    events.append(value)
    except OSError:
        logger.exception("Could not read prediction monitoring events.")
    return events


def prediction_summary(events: list[dict[str, Any]]) -> dict[str, float | int]:
    """Summarize predictions only; these are not ground-truth performance metrics."""
    probabilities = [
        float(event["churn_probability"])
        for event in events
        if isinstance(event.get("churn_probability"), (int, float))
        and np.isfinite(event["churn_probability"])
    ]
    return {
        "prediction_count": len(events),
        "predicted_churn_rate": (
            float(np.mean([int(event.get("prediction", 0)) for event in events])) if events else 0.0
        ),
        "mean_churn_probability": float(np.mean(probabilities)) if probabilities else 0.0,
        "median_churn_probability": float(np.median(probabilities)) if probabilities else 0.0,
    }


def _numeric_psi(expected: pd.Series, actual: pd.Series) -> float:
    """Estimate numeric PSI using quantile bins defined by the baseline data."""
    baseline = pd.to_numeric(expected, errors="coerce").dropna().to_numpy(dtype=float)
    current = pd.to_numeric(actual, errors="coerce").dropna().to_numpy(dtype=float)
    if not len(baseline) or not len(current):
        return float("nan")
    edges = np.unique(np.quantile(baseline, np.linspace(0, 1, 11)))
    if len(edges) < 2:
        return 0.0 if np.all(current == baseline[0]) else float("inf")
    edges[0], edges[-1] = -np.inf, np.inf
    expected_share = np.histogram(baseline, bins=edges)[0] / len(baseline)
    current_share = np.histogram(current, bins=edges)[0] / len(current)
    expected_share = np.clip(expected_share, 1e-6, None)
    current_share = np.clip(current_share, 1e-6, None)
    return float(np.sum((current_share - expected_share) * np.log(current_share / expected_share)))


def _categorical_psi(expected: pd.Series, actual: pd.Series) -> float:
    """Estimate categorical PSI over the combined baseline and current categories."""
    baseline = expected.dropna().astype(str)
    current = actual.dropna().astype(str)
    if not len(baseline) or not len(current):
        return float("nan")
    categories = sorted(set(baseline) | set(current))
    expected_share = (
        baseline.value_counts(normalize=True).reindex(categories, fill_value=0).to_numpy()
    )
    current_share = (
        current.value_counts(normalize=True).reindex(categories, fill_value=0).to_numpy()
    )
    expected_share = np.clip(expected_share, 1e-6, None)
    current_share = np.clip(current_share, 1e-6, None)
    return float(np.sum((current_share - expected_share) * np.log(current_share / expected_share)))


def feature_drift_report(
    baseline: pd.DataFrame,
    events: list[dict[str, Any]],
    minimum_events: int = 30,
) -> pd.DataFrame:
    """Compare allow-listed logged feature distributions with training baseline PSI.

    PSI bands (<0.10, 0.10-0.25, >0.25) are heuristic review flags, not tests of
    statistical significance or proof that model performance has degraded.
    """
    if len(events) < minimum_events:
        return pd.DataFrame(
            [
                {
                    "status": "insufficient_data",
                    "events": len(events),
                    "minimum_events": minimum_events,
                }
            ]
        )
    current = pd.DataFrame([event.get("features", {}) for event in events])
    rows = []
    for feature in MONITORING_FEATURES:
        if feature not in baseline or feature not in current:
            continue
        if pd.api.types.is_numeric_dtype(baseline[feature]):
            psi = _numeric_psi(baseline[feature], current[feature])
        else:
            psi = _categorical_psi(baseline[feature], current[feature])
        rows.append(
            {
                "feature": feature,
                "psi": psi,
                "status": "high" if psi > 0.25 else "moderate" if psi >= 0.10 else "low",
                "events": len(events),
            }
        )
    return pd.DataFrame(rows)
