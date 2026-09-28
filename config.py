"""Project-wide paths, environment settings, and prediction thresholds."""

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env", override=False)


def _project_path(variable: str, default: Path) -> Path:
    """Resolve a path environment setting relative to the project root."""
    configured = Path(os.getenv(variable, str(default)))
    return configured if configured.is_absolute() else PROJECT_ROOT / configured


DATA_DIR = _project_path("CHURN_DATA_DIR", PROJECT_ROOT / "data")
DATA_PATH = _project_path("CHURN_DATA_PATH", DATA_DIR / "telco_churn.csv")
MODEL_DIR = _project_path("CHURN_MODEL_DIR", PROJECT_ROOT / "models")
MODEL_PATH = _project_path("CHURN_MODEL_PATH", MODEL_DIR / "churn_model.pkl")
METADATA_PATH = _project_path("CHURN_METADATA_PATH", MODEL_DIR / "model_metadata.json")
REPORTS_DIR = _project_path("CHURN_REPORTS_DIR", PROJECT_ROOT / "reports")
FIGURES_DIR = _project_path("CHURN_FIGURES_DIR", REPORTS_DIR / "figures")
RESULTS_PATH = _project_path("CHURN_RESULTS_PATH", REPORTS_DIR / "model_results.csv")
LOG_DIR = _project_path("CHURN_LOG_DIR", PROJECT_ROOT / "logs")
PREDICTION_LOG_PATH = _project_path("CHURN_PREDICTION_LOG_PATH", LOG_DIR / "predictions.jsonl")
API_HOST = os.getenv("CHURN_API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("CHURN_API_PORT", "8000"))
MODEL_VERSION = os.getenv("CHURN_MODEL_VERSION", "2.0.0")
FEATURES_VERSION = os.getenv("CHURN_FEATURES_VERSION", "1.0.0")
TARGET = "Churn"
RANDOM_STATE = int(os.getenv("CHURN_RANDOM_STATE", "42"))
TEST_SIZE = float(os.getenv("CHURN_TEST_SIZE", "0.20"))
CHURN_THRESHOLD = float(os.getenv("CHURN_THRESHOLD", "0.50"))
HIGH_RISK_THRESHOLD = float(os.getenv("CHURN_HIGH_RISK_THRESHOLD", "0.70"))
MEDIUM_RISK_THRESHOLD = float(os.getenv("CHURN_MEDIUM_RISK_THRESHOLD", "0.40"))

if not 0 < TEST_SIZE < 1:
    raise ValueError("CHURN_TEST_SIZE must be between 0 and 1.")
if not 0 <= CHURN_THRESHOLD <= 1:
    raise ValueError("CHURN_THRESHOLD must be between 0 and 1.")
if not 0 <= MEDIUM_RISK_THRESHOLD <= HIGH_RISK_THRESHOLD <= 1:
    raise ValueError(
        "Risk thresholds must satisfy 0 <= CHURN_MEDIUM_RISK_THRESHOLD "
        "<= CHURN_HIGH_RISK_THRESHOLD <= 1."
    )
