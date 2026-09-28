"""Small integration tests for train, evaluation, and model serialization."""

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from src.evaluate import calculate_metrics, threshold_analysis, top_k_analysis
from src.train import build_model_pipeline


def test_small_training_pipeline_evaluates_and_round_trips(tmp_path):
    """Fit the preprocessing-model pipeline and reload the saved estimator."""
    rng = np.random.default_rng(42)
    size = 80
    labels = np.array([0] * 48 + [1] * 32)
    rng.shuffle(labels)
    features = pd.DataFrame(
        {
            "tenure": rng.integers(1, 73, size),
            "MonthlyCharges": rng.uniform(20, 120, size),
            "Contract": np.where(labels == 1, "Month-to-month", "One year"),
            "TechSupport": np.where(labels == 1, "No", "Yes"),
        }
    )
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.25, random_state=42, stratify=labels
    )
    pipeline = build_model_pipeline(X_train, LogisticRegression(max_iter=1000, random_state=42))
    pipeline.fit(X_train, y_train)
    probabilities = pipeline.predict_proba(X_test)[:, 1]
    metrics, predictions = calculate_metrics(y_test, probabilities)
    assert {
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "ROC_AUC",
        "PR_AUC",
        "Brier_Score",
        "Predicted_Churn_Volume",
    } == set(metrics)
    assert metrics["ROC_AUC"] == roc_auc_score(y_test, probabilities)
    assert len(predictions) == len(y_test)
    destination = tmp_path / "pipeline.joblib"
    joblib.dump(pipeline, destination)
    reloaded = joblib.load(destination)
    np.testing.assert_allclose(reloaded.predict_proba(X_test), pipeline.predict_proba(X_test))


def test_business_reports_use_probability_ranking():
    """Threshold and Top-K metrics reflect supplied scores and labels."""
    labels = np.array([0, 1, 0, 1, 1, 0, 0, 1])
    probabilities = np.array([0.1, 0.9, 0.2, 0.7, 0.4, 0.3, 0.5, 0.8])
    thresholds = threshold_analysis(labels, probabilities, [0.5])
    assert thresholds.loc[0, "predicted_churn_customers"] == 4
    assert thresholds.loc[0, "actual_churn_captured"] == 3
    top_k = top_k_analysis(labels, probabilities, [0.25])
    assert top_k.loc[0, "targeted_customers"] == 2
    assert top_k.loc[0, "actual_churners_captured"] == 2
