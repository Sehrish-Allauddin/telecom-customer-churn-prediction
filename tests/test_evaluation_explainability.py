"""Tests for model diagnostics and feature-space-consistent SHAP output."""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.evaluate import save_business_reports, save_evaluation_artifacts
from src.explainability import explain_customer, save_global_explanation
from src.train import build_model_pipeline


def test_evaluation_artifacts_include_pr_calibration_and_business_reports(tmp_path, monkeypatch):
    """Evaluation outputs include threshold and ranking diagnostics on supplied labels."""
    import src.evaluate as evaluation

    monkeypatch.setattr(evaluation, "FIGURES_DIR", tmp_path / "figures")
    y_true = np.array([0, 1, 0, 1, 1, 0, 0, 1])
    probabilities = np.array([0.1, 0.9, 0.2, 0.7, 0.4, 0.3, 0.5, 0.8])
    metrics, predictions = evaluation.calculate_metrics(y_true, probabilities)
    results = pd.DataFrame([{"Model": "test", **metrics}])
    save_evaluation_artifacts(
        results,
        {"test": {"metrics": metrics, "predictions": predictions, "probabilities": probabilities}},
        y_true,
    )
    thresholds, top_k = save_business_reports(
        y_true,
        probabilities,
        y_true,
        probabilities,
        [0.3, 0.5],
        [0.25, 0.5],
        tmp_path / "reports",
    )
    assert metrics["PR_AUC"] > 0
    assert metrics["Brier_Score"] >= 0
    assert len(thresholds) == 2
    assert len(top_k) == 2
    assert (tmp_path / "figures" / "precision_recall_curve_comparison.png").exists()
    assert (tmp_path / "figures" / "calibration_curve_comparison.png").exists()
    assert (tmp_path / "reports" / "threshold_analysis.csv").exists()
    assert (tmp_path / "reports" / "top_k_lift_analysis.csv").exists()


def test_shap_maps_encoded_columns_and_writes_global_artifacts(tmp_path):
    """Local/global SHAP use the fitted preprocessing space and actual row values."""
    rng = np.random.default_rng(41)
    labels = np.array([0, 1] * 20)
    features = pd.DataFrame(
        {
            "tenure": rng.integers(1, 73, size=len(labels)),
            "MonthlyCharges": rng.uniform(20, 120, size=len(labels)),
            "Contract": np.where(labels == 1, "Month-to-month", "One year"),
        }
    )
    pipeline = build_model_pipeline(
        features,
        RandomForestClassifier(n_estimators=12, max_depth=4, random_state=41),
    )
    pipeline.fit(features, labels)
    contributions = explain_customer(
        pipeline,
        features.iloc[[0]],
        features.iloc[1:16],
        top_n=4,
    )
    assert len(contributions) == 4
    assert contributions["Direction"].isin(["Risk increasing", "Risk reducing"]).all()
    importance = save_global_explanation(
        pipeline, features.iloc[1:16], features.iloc[16:24], tmp_path
    )
    assert importance.iloc[0]["Mean absolute SHAP"] >= 0
    assert (tmp_path / "shap_summary.png").exists()
    assert (tmp_path / "shap_global_importance.csv").exists()
