"""SHAP explanations for fitted preprocessing-plus-estimator pipelines."""

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _dense(values: Any) -> np.ndarray:
    """Convert sparse or dense transformed features to a numeric array."""
    return values.toarray() if hasattr(values, "toarray") else np.asarray(values)


def _display_names(model: Any) -> list[str]:
    """Map encoded names to source feature/category labels for explanation output."""
    preprocessor = model.named_steps["preprocessor"]
    raw_names = [str(name) for name in preprocessor.get_feature_names_out()]
    source_names = list(getattr(preprocessor, "feature_names_in_", []))
    display = []
    for name in raw_names:
        prefix, _, value = name.partition("__")
        if prefix == "categorical":
            original = next(
                (field for field in source_names if value.startswith(f"{field}_")), None
            )
            if original:
                value = f"{original} = {value[len(original) + 1:]}"
        display.append(value)
    return display


def make_explainer(model: Any, background: pd.DataFrame) -> tuple[Any, list[str]]:
    """Create a SHAP explainer in the model's fitted transformed feature space."""
    import shap

    preprocessor = model.named_steps["preprocessor"]
    estimator = model.named_steps["model"]
    background_values = _dense(preprocessor.transform(background))
    names = _display_names(model)
    explainer = shap.Explainer(estimator, background_values, feature_names=names)
    return explainer, names


def explain_rows(
    model: Any, raw_rows: pd.DataFrame, explainer: Any
) -> tuple[list[str], np.ndarray]:
    """Return positive-class SHAP values for rows in the raw model feature schema."""
    transformed = _dense(model.named_steps["preprocessor"].transform(raw_rows))
    explanation = explainer(transformed)
    values = np.asarray(explanation.values)
    if values.ndim == 3:
        positive_index = 1 if values.shape[-1] > 1 else 0
        values = values[:, :, positive_index]
    elif values.ndim == 2:
        pass
    else:
        raise ValueError(f"Unexpected SHAP value shape: {values.shape}")
    return _display_names(model), values


def explain_customer(
    model: Any, raw_customer: pd.DataFrame, background: pd.DataFrame, top_n: int = 8
) -> pd.DataFrame:
    """Rank a customer's positive and negative contributions to churn score."""
    explainer, names = make_explainer(model, background)
    returned_names, values = explain_rows(model, raw_customer, explainer)
    contribution = values[0]
    result = pd.DataFrame({"Feature": returned_names or names, "SHAP value": contribution})
    result["Direction"] = np.where(result["SHAP value"] >= 0, "Risk increasing", "Risk reducing")
    return result.reindex(result["SHAP value"].abs().sort_values(ascending=False).index).head(top_n)


def save_global_explanation(
    model: Any,
    background: pd.DataFrame,
    evaluation_rows: pd.DataFrame,
    output_dir: Path,
    max_rows: int = 500,
) -> pd.DataFrame:
    """Save SHAP global importance and summary plot on training-independent test rows."""
    import shap

    output_dir.mkdir(parents=True, exist_ok=True)
    explainer, names = make_explainer(model, background)
    raw_sample = evaluation_rows.head(max_rows)
    transformed = _dense(model.named_steps["preprocessor"].transform(raw_sample))
    explanation = explainer(transformed)
    values = np.asarray(explanation.values)
    if values.ndim == 3:
        values = values[:, :, 1 if values.shape[-1] > 1 else 0]
    importance = pd.DataFrame(
        {"Feature": names, "Mean absolute SHAP": np.abs(values).mean(axis=0)}
    ).sort_values("Mean absolute SHAP", ascending=False)
    importance.to_csv(output_dir / "shap_global_importance.csv", index=False)
    shap.summary_plot(values, transformed, feature_names=names, show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(output_dir / "shap_summary.png", dpi=170, bbox_inches="tight")
    plt.close()
    return importance.reset_index(drop=True)
