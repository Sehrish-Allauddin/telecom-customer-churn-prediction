"""Metrics, report tables, and diagnostic plots."""

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import (
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from config import FIGURES_DIR


def calculate_metrics(
    y_true: Any, probabilities: np.ndarray, threshold: float = 0.5
) -> tuple[dict[str, float], np.ndarray]:
    """Compute classification metrics from probabilities and a decision threshold.

    Args:
        y_true: Ground-truth binary labels.
        probabilities: Predicted positive-class probabilities.
        threshold: Probability threshold used to create class predictions.

    Returns:
        A metrics mapping and corresponding binary predictions.
    """
    predictions = (np.asarray(probabilities) >= threshold).astype(int)
    return (
        {
            "Accuracy": accuracy_score(y_true, predictions),
            "Precision": precision_score(y_true, predictions, zero_division=0),
            "Recall": recall_score(y_true, predictions, zero_division=0),
            "F1": f1_score(y_true, predictions, zero_division=0),
            "ROC_AUC": roc_auc_score(y_true, probabilities),
            "PR_AUC": average_precision_score(y_true, probabilities),
            "Brier_Score": brier_score_loss(y_true, probabilities),
            "Predicted_Churn_Volume": float(predictions.sum()),
        },
        predictions,
    )


def save_evaluation_artifacts(
    results: pd.DataFrame, evaluations: dict[str, dict[str, Any]], y_test: Any
) -> None:
    """Write metric comparison, ROC and confusion-matrix figures.

    Args:
        results: Per-model metric rows.
        evaluations: Probabilities, predictions, and metric dictionaries by model.
        y_test: Ground-truth held-out labels.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    metric_cols = ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC", "PR_AUC"]
    ax = results.set_index("Model")[metric_cols].plot(
        kind="bar", figsize=(11, 6), ylim=(0, 1), rot=15
    )
    ax.set(title="Churn Model Metric Comparison", ylabel="Score", xlabel="Model")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "model_comparison.png", dpi=160)
    plt.close()
    fig, ax = plt.subplots(figsize=(8, 7))
    for name, item in evaluations.items():
        RocCurveDisplay.from_predictions(
            y_test,
            item["probabilities"],
            name=f"{name} (AUC={item['metrics']['ROC_AUC']:.3f})",
            ax=ax,
        )
    ax.set_title("ROC Curve Comparison")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "roc_curve_comparison.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 7))
    for name, item in evaluations.items():
        PrecisionRecallDisplay.from_predictions(
            y_test,
            item["probabilities"],
            name=f"{name} (AP={item['metrics']['PR_AUC']:.3f})",
            ax=ax,
        )
    ax.set_title("Precision-Recall Curve Comparison")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "precision_recall_curve_comparison.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 7))
    for name, item in evaluations.items():
        CalibrationDisplay.from_predictions(
            y_test,
            item["probabilities"],
            name=f"{name} (Brier={item['metrics']['Brier_Score']:.3f})",
            n_bins=10,
            strategy="quantile",
            ax=ax,
        )
    ax.set_title("Probability Calibration (reliability curve)")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "calibration_curve_comparison.png", dpi=160)
    plt.close(fig)
    for name, item in evaluations.items():
        matrix = confusion_matrix(y_test, item["predictions"], labels=[0, 1])
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        image = ax.imshow(matrix, cmap="Blues")
        ax.set(
            xticks=[0, 1],
            yticks=[0, 1],
            xticklabels=["Predicted stay", "Predicted churn"],
            yticklabels=["Actual stay", "Actual churn"],
            ylabel="Actual",
            xlabel="Predicted",
            title=f"{name} Confusion Matrix",
        )
        labels = [
            [f"TN\n{matrix[0,0]}", f"FP\n{matrix[0,1]}"],
            [f"FN\n{matrix[1,0]}", f"TP\n{matrix[1,1]}"],
        ]
        for i in range(2):
            for j in range(2):
                ax.text(j, i, labels[i][j], ha="center", va="center", color="black", fontsize=12)
        fig.colorbar(image, ax=ax, fraction=0.046)
        plt.tight_layout()
        fig.savefig(FIGURES_DIR / f"confusion_matrix_{name.lower().replace(' ', '_')}.png", dpi=160)
        plt.close(fig)


def threshold_analysis(
    y_true: Any, probabilities: np.ndarray, thresholds: list[float]
) -> pd.DataFrame:
    """Calculate operating metrics for candidate thresholds without choosing one.

    Args:
        y_true: Binary ground truth from an out-of-fold or held-out set.
        probabilities: Corresponding churn scores.
        thresholds: Explicit operating points to compare.

    Returns:
        One row per threshold, including outreach volume and captured churners.
    """
    labels = np.asarray(y_true, dtype=int)
    scores = np.asarray(probabilities, dtype=float)
    rows = []
    for threshold in thresholds:
        predicted = scores >= threshold
        true_positive = int(np.sum(predicted & (labels == 1)))
        rows.append(
            {
                "threshold": float(threshold),
                "precision": precision_score(labels, predicted, zero_division=0),
                "recall": recall_score(labels, predicted, zero_division=0),
                "f1": f1_score(labels, predicted, zero_division=0),
                "predicted_churn_customers": int(predicted.sum()),
                "actual_churn_captured": true_positive,
            }
        )
    return pd.DataFrame(rows)


def top_k_analysis(y_true: Any, probabilities: np.ndarray, fractions: list[float]) -> pd.DataFrame:
    """Measure churn capture when only the highest scoring customers are contacted."""
    labels = np.asarray(y_true, dtype=int)
    scores = np.asarray(probabilities, dtype=float)
    order = np.argsort(-scores)
    total_churners = int(labels.sum())
    rows = []
    for fraction in fractions:
        count = min(len(labels), max(1, int(np.ceil(len(labels) * fraction))))
        captured = int(labels[order[:count]].sum())
        precision_at_k = captured / count if count else 0.0
        recall_at_k = captured / total_churners if total_churners else 0.0
        baseline_rate = total_churners / len(labels) if len(labels) else 0.0
        rows.append(
            {
                "top_fraction": float(fraction),
                "targeted_customers": count,
                "actual_churners_captured": captured,
                "precision_at_k": precision_at_k,
                "recall_at_k": recall_at_k,
                "lift": precision_at_k / baseline_rate if baseline_rate else 0.0,
            }
        )
    return pd.DataFrame(rows)


def save_business_reports(
    y_train: Any,
    train_oof_probabilities: np.ndarray,
    y_test: Any,
    test_probabilities: np.ndarray,
    thresholds: list[float],
    fractions: list[float],
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Write threshold and top-k business operating-point tables and figures."""
    output_dir.mkdir(parents=True, exist_ok=True)
    threshold_rows = threshold_analysis(y_train, train_oof_probabilities, thresholds)
    top_k_rows = top_k_analysis(y_test, test_probabilities, fractions)
    threshold_rows.to_csv(output_dir / "threshold_analysis.csv", index=False)
    top_k_rows.to_csv(output_dir / "top_k_lift_analysis.csv", index=False)

    ax = threshold_rows.set_index("threshold")[["precision", "recall", "f1"]].plot(
        marker="o", figsize=(9, 5), ylim=(0, 1), grid=True
    )
    ax.set(title="Threshold trade-offs (out-of-fold training predictions)", ylabel="Score")
    ax.set_xlabel("Decision threshold")
    plt.tight_layout()
    plt.savefig(output_dir / "threshold_analysis.png", dpi=160)
    plt.close()

    ax = top_k_rows.set_index("top_fraction")[["recall_at_k", "lift"]].plot(
        marker="o", figsize=(9, 5), grid=True
    )
    ax.set(title="Top-K targeting on held-out test data", ylabel="Recall / lift")
    ax.set_xlabel("Fraction of highest-risk customers targeted")
    plt.tight_layout()
    plt.savefig(output_dir / "top_k_lift_analysis.png", dpi=160)
    plt.close()
    return threshold_rows, top_k_rows


def save_feature_importance(
    model: Any, model_name: str, output_dir: Path = FIGURES_DIR, top_n: int = 15
) -> pd.DataFrame:
    """Extract model importances against fitted encoded feature names.

    Args:
        model: Fitted preprocessing and estimator pipeline.
        model_name: Human-readable estimator name for output artifact names.
        output_dir: Destination for the sorted CSV and chart.
        top_n: Number of highest importance features to include in the chart.

    Returns:
        All feature importances, sorted descending.
    """
    fitted_prep = model.named_steps["preprocessor"]
    estimator = model.named_steps["model"]
    values = getattr(estimator, "feature_importances_", None)
    if values is None:
        return pd.DataFrame(columns=["Feature", "Importance"])
    importance = pd.DataFrame(
        {"Feature": fitted_prep.get_feature_names_out(), "Importance": values}
    )
    importance = importance.sort_values("Importance", ascending=False).reset_index(drop=True)
    importance.to_csv(
        output_dir / f"feature_importance_{model_name.lower().replace(' ', '_')}.csv", index=False
    )
    top = importance.head(top_n).sort_values("Importance")
    ax = top.plot.barh(
        x="Feature",
        y="Importance",
        legend=False,
        figsize=(9, 7),
        title=f"{model_name}: Top {min(top_n, len(top))} Features",
    )
    ax.set_ylabel("Encoded feature")
    plt.tight_layout()
    plt.savefig(
        output_dir / f"feature_importance_{model_name.lower().replace(' ', '_')}.png", dpi=160
    )
    plt.close()
    return importance
