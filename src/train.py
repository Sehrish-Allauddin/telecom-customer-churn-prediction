"""Reproducible, leakage-safe cross-validated churn model selection."""

import hashlib
import json
import logging
import os
import platform
from datetime import date, datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from config import (
    CHURN_THRESHOLD,
    FEATURES_VERSION,
    FIGURES_DIR,
    METADATA_PATH,
    MODEL_DIR,
    MODEL_PATH,
    MODEL_VERSION,
    RANDOM_STATE,
    RESULTS_PATH,
    TARGET,
    TEST_SIZE,
)
from src.data_cleaning import find_dataset, load_and_clean_data
from src.evaluate import (
    calculate_metrics,
    save_business_reports,
    save_evaluation_artifacts,
    save_feature_importance,
)
from src.explainability import save_global_explanation
from src.logging_config import configure_logging
from src.preprocessing import engineer_features, make_preprocessor

configure_logging()
logger = logging.getLogger(__name__)


def _training_package_versions() -> dict[str, str]:
    """Record relevant library versions available in the training environment."""
    packages = ("scikit-learn", "xgboost", "xgboost-cpu", "pandas", "numpy", "shap")
    result = {"python": platform.python_version()}
    for package in packages:
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            continue
    return result


def build_model_pipeline(features: pd.DataFrame, estimator: Any) -> Pipeline:
    """Build preprocessing and estimator pipeline; no synthetic rows are created.

    Imbalance is handled by estimator class weights or positive-class weights. This keeps
    categorical combinations valid after one-hot encoding and avoids vanilla SMOTE's
    interpolation of meaningless fractional category indicators.
    """
    return Pipeline([("preprocessor", make_preprocessor(features)), ("model", estimator)])


def _candidate_searches(features: pd.DataFrame, y_train: pd.Series) -> dict[str, tuple[Any, dict]]:
    """Create bounded reproducible searches for the three existing model families."""
    positives = int(y_train.sum())
    negatives = int(len(y_train) - positives)
    estimators = {
        "Logistic Regression": (
            LogisticRegression(max_iter=1500, class_weight="balanced", random_state=RANDOM_STATE),
            {"model__C": [0.05, 0.2, 0.5, 1.0, 2.0, 5.0]},
        ),
        "Random Forest": (
            RandomForestClassifier(
                n_estimators=240,
                class_weight="balanced_subsample",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            {
                "model__max_depth": [None, 8, 14],
                "model__min_samples_leaf": [1, 3, 6],
                "model__max_features": ["sqrt", 0.7],
            },
        ),
        "XGBoost": (
            XGBClassifier(
                n_estimators=180,
                learning_rate=0.06,
                max_depth=4,
                min_child_weight=1,
                subsample=0.9,
                colsample_bytree=0.9,
                scale_pos_weight=negatives / max(positives, 1),
                random_state=RANDOM_STATE,
                eval_metric="logloss",
                n_jobs=-1,
            ),
            {
                "model__n_estimators": [120, 220],
                "model__max_depth": [2, 4, 6],
                "model__learning_rate": [0.03, 0.08],
                "model__min_child_weight": [1, 5],
            },
        ),
    }
    return {
        name: (build_model_pipeline(features, estimator), parameters)
        for name, (estimator, parameters) in estimators.items()
    }


def _dataset_sha256() -> str:
    """Hash the source CSV bytes so the training artifact is traceable."""
    digest = hashlib.sha256()
    with find_dataset().open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def train() -> tuple[pd.DataFrame, str]:
    """Tune candidates on training folds and evaluate once on the untouched test split."""
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))
    cleaned = load_and_clean_data()
    y = cleaned[TARGET].astype(int)
    raw_features = cleaned.drop(columns=[TARGET])
    raw_X_train, raw_X_test, y_train, y_test = train_test_split(
        raw_features,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    X_train = engineer_features(raw_X_train)
    X_test = engineer_features(raw_X_test)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = {"PR_AUC": "average_precision", "ROC_AUC": "roc_auc"}
    searches: dict[str, RandomizedSearchCV] = {}
    rows: list[dict[str, Any]] = []
    pipelines: dict[str, Pipeline] = {}
    evaluations: dict[str, dict[str, Any]] = {}

    for name, (pipeline, parameter_space) in _candidate_searches(X_train, y_train).items():
        search = RandomizedSearchCV(
            pipeline,
            parameter_space,
            n_iter=4,
            scoring=scoring,
            refit="PR_AUC",
            cv=cv,
            random_state=RANDOM_STATE,
            n_jobs=1,
            return_train_score=False,
            error_score="raise",
        )
        search.fit(X_train, y_train)
        fitted = search.best_estimator_
        test_probabilities = fitted.predict_proba(X_test)[:, 1]
        metrics, predictions = calculate_metrics(y_test, test_probabilities, CHURN_THRESHOLD)
        pipelines[name] = fitted
        searches[name] = search
        evaluations[name] = {
            "metrics": metrics,
            "predictions": predictions,
            "probabilities": test_probabilities,
        }
        rows.append(
            {
                "Model": name,
                "CV_PR_AUC": float(search.best_score_),
                "CV_PR_AUC_Std": float(search.cv_results_["std_test_PR_AUC"][search.best_index_]),
                "CV_ROC_AUC": float(search.cv_results_["mean_test_ROC_AUC"][search.best_index_]),
                "Best_Parameters": json.dumps(search.best_params_, sort_keys=True),
                **metrics,
            }
        )
        logger.info(
            "%s: CV PR-AUC=%.4f, CV ROC-AUC=%.4f, test PR-AUC=%.4f, test ROC-AUC=%.4f",
            name,
            search.best_score_,
            search.cv_results_["mean_test_ROC_AUC"][search.best_index_],
            metrics["PR_AUC"],
            metrics["ROC_AUC"],
        )

    results = pd.DataFrame(rows).sort_values("CV_PR_AUC", ascending=False).reset_index(drop=True)
    selected = str(results.loc[0, "Model"])
    selected_model = pipelines[selected]

    # Threshold trade-offs use only out-of-fold predictions from the training partition.
    train_oof_probabilities = cross_val_predict(
        selected_model,
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=1,
    )[:, 1]
    threshold_values = sorted({0.30, 0.40, 0.50, 0.60, 0.70, CHURN_THRESHOLD})

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULTS_PATH, index=False)
    save_evaluation_artifacts(results, evaluations, y_test)
    threshold_report, top_k_report = save_business_reports(
        y_train,
        train_oof_probabilities,
        y_test,
        evaluations[selected]["probabilities"],
        threshold_values,
        [0.05, 0.10, 0.20, 0.30],
        RESULTS_PATH.parent,
    )
    for name in ("Random Forest", "XGBoost"):
        save_feature_importance(pipelines[name], name)
    global_importance = save_global_explanation(
        selected_model,
        X_train.sample(n=min(100, len(X_train)), random_state=RANDOM_STATE),
        X_test,
        FIGURES_DIR,
    )

    joblib.dump(selected_model, MODEL_PATH)
    test_exposure = float(
        (
            X_test["MonthlyCharges"].astype(float).to_numpy()
            * evaluations[selected]["probabilities"]
        ).sum()
    )
    selected_row = results.loc[results["Model"] == selected].iloc[0]
    configured_operating_point = (
        threshold_report.loc[np.isclose(threshold_report["threshold"], CHURN_THRESHOLD)]
        .iloc[0]
        .to_dict()
    )
    configured_operating_point = {
        key: value.item() if isinstance(value, np.generic) else value
        for key, value in configured_operating_point.items()
    }
    metadata = {
        "model_name": selected,
        "model_type": selected_model.named_steps["model"].__class__.__name__,
        "model_version": MODEL_VERSION,
        "features_version": FEATURES_VERSION,
        "features": list(X_train.columns),
        "target": TARGET,
        "training_date": date.today().isoformat(),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "dataset_sha256": _dataset_sha256(),
        "dataset_rows": int(len(cleaned)),
        "training_environment": _training_package_versions(),
        "selection_criterion": (
            "Highest mean 5-fold training PR-AUC; test set was not used for selection."
        ),
        "imbalance_strategy": (
            "Class-weighted estimators; no synthetic sampling of one-hot features."
        ),
        "cross_validation": {
            "strategy": "StratifiedKFold",
            "folds": 5,
            "random_state": RANDOM_STATE,
            "selected_mean_pr_auc": float(selected_row["CV_PR_AUC"]),
            "selected_std_pr_auc": float(selected_row["CV_PR_AUC_Std"]),
            "selected_mean_roc_auc": float(selected_row["CV_ROC_AUC"]),
            "best_parameters": json.loads(selected_row["Best_Parameters"]),
        },
        "threshold": {
            "value": CHURN_THRESHOLD,
            "source": (
                "Configured business operating point; not optimized without retention cost data."
            ),
            "cross_validated_operating_point": configured_operating_point,
        },
        "metrics": evaluations[selected]["metrics"],
        "top_shap_features": global_importance.head(10).to_dict(orient="records"),
        "model_comparison": results.set_index("Model").to_dict(orient="index"),
        "business_test": {
            "top_k": top_k_report.to_dict(orient="records"),
            "estimated_revenue_exposure": test_exposure,
            "estimated_revenue_exposure_definition": (
                "sum(MonthlyCharges * predicted churn probability) on held-out test rows; "
                "a descriptive estimate, not causal loss."
            ),
        },
        "training_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "random_state": RANDOM_STATE,
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logger.info("Selected %s by training CV PR-AUC; saved model to %s", selected, MODEL_PATH)
    return results, selected


def main() -> None:
    """Run model training from the console entry point."""
    logger.info("Training started")
    results, selected = train()
    logger.info("Training completed successfully; selected model: %s", selected)
    logger.info("Model comparison results:\n%s", results.to_string(index=False))


if __name__ == "__main__":
    main()
