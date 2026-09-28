"""Feature engineering and reusable sklearn preprocessing."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SERVICE_COLUMNS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]
DROP_COLUMNS = {"customerID", "Churn"}


def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add engineered customer features and remove non-feature identifiers.

    Args:
        frame: Raw customer rows. Existing target columns are removed from output.

    Returns:
        A copy with tenure group, affirmative service count, and safe average spend.
    """
    data = frame.copy()
    if "tenure" in data:
        tenure = pd.to_numeric(data["tenure"], errors="coerce")
        data["tenure_group"] = pd.cut(
            tenure,
            bins=[-1, 12, 24, 48, 72],
            labels=["0-12", "13-24", "25-48", "49-72"],
            include_lowest=True,
        ).astype("object")
    present = [column for column in SERVICE_COLUMNS if column in data]
    if present:
        # Count affirmative subscriptions only; 'No internet service' and 'No' count as zero.
        data["service_count"] = sum(
            data[column].astype("string").str.strip().str.lower().eq("yes").astype(int)
            for column in present
        )
    if {"TotalCharges", "tenure"}.issubset(data.columns):
        total = pd.to_numeric(data["TotalCharges"], errors="coerce")
        tenure = pd.to_numeric(data["tenure"], errors="coerce").replace(0, np.nan)
        data["average_monthly_spend"] = total.div(tenure)
    return data.drop(columns=[column for column in DROP_COLUMNS if column in data], errors="ignore")


def make_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    """Create numeric and categorical preprocessing based on the training dtypes.

    Args:
        features: Training feature frame used to determine its column groups.

    Returns:
        A ColumnTransformer with median imputation/scaling and mode imputation/one-hot encoding.
    """
    numeric = features.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical = [column for column in features.columns if column not in numeric]
    numeric_pipeline = SklearnPipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical_pipeline = SklearnPipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, numeric),
            ("categorical", categorical_pipeline, categorical),
        ],
        remainder="drop",
    )


def feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Return transformed feature names, including encoded categories.

    Args:
        preprocessor: Fitted ColumnTransformer.

    Returns:
        The names of all numeric and expanded one-hot output columns.
    """
    return [str(name) for name in preprocessor.get_feature_names_out()]
