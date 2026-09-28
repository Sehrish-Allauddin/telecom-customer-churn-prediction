"""Tests for privacy-conscious prediction event logs."""

import json

import pandas as pd

from src.monitoring import feature_drift_report, log_prediction, prediction_summary


def test_monitoring_logs_only_allowlisted_non_pii_fields(tmp_path, raw_customer):
    """A structured event is written without demographic or identifier data."""
    path = tmp_path / "logs" / "predictions.jsonl"
    raw_customer["customerID"] = "private-id"
    log_prediction(raw_customer, 1, 0.78, "High Risk", path)
    event = json.loads(path.read_text(encoding="utf-8").strip())
    assert event["prediction"] == 1
    assert event["features"]["Contract"] == "Month-to-month"
    assert "customerID" not in path.read_text(encoding="utf-8")
    assert "gender" not in event["features"]


def test_prediction_summary_is_prediction_only(raw_customer, tmp_path):
    """Summary reports volume and scores without inventing observed outcomes."""
    path = tmp_path / "predictions.jsonl"
    log_prediction(raw_customer, 1, 0.78, "High Risk", path)
    summary = prediction_summary([json.loads(path.read_text())])
    assert summary["prediction_count"] == 1
    assert summary["predicted_churn_rate"] == 1.0
    assert "recall" not in summary


def test_feature_drift_report_requires_minimum_sample(raw_customer):
    """A tiny stream is marked insufficient instead of over-interpreting PSI."""
    baseline = pd.DataFrame([raw_customer])
    events = [{"features": {"Contract": "Month-to-month"}}]
    report = feature_drift_report(baseline, events, minimum_events=30)
    assert report.loc[0, "status"] == "insufficient_data"


def test_feature_drift_report_compares_allowlisted_fields(raw_customer):
    """PSI report only covers available monitoring features and is sample-size gated."""
    baseline = pd.DataFrame(
        {
            "tenure": list(range(1, 101)),
            "Contract": ["One year"] * 80 + ["Month-to-month"] * 20,
            "MonthlyCharges": [50.0] * 100,
        }
    )
    events = [
        {
            "features": {
                "tenure": 1,
                "Contract": "Month-to-month",
                "MonthlyCharges": 100.0,
            }
        }
        for _ in range(30)
    ]
    report = feature_drift_report(baseline, events, minimum_events=30)
    assert set(report["feature"]) == {"tenure", "Contract", "MonthlyCharges"}
    assert report["psi"].notna().all()
