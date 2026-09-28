"""Run the explicit first-time Docker model bootstrap only when artifacts are absent/stale."""

import argparse
import json
import logging

from config import FEATURES_VERSION, METADATA_PATH, MODEL_PATH, MODEL_VERSION
from src.logging_config import configure_logging
from src.train import train

configure_logging()
logger = logging.getLogger(__name__)


def artifacts_are_current() -> bool:
    """Check that the persisted image-volume model matches application versions."""
    if not MODEL_PATH.exists() or not METADATA_PATH.exists():
        return False
    try:
        metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        metadata.get("model_version") == MODEL_VERSION
        and metadata.get("features_version") == FEATURES_VERSION
    )


def main() -> None:
    """Train once for an empty or stale model volume; keep startup repeatable."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Retrain even if artifacts exist.")
    force = parser.parse_args().force
    if artifacts_are_current() and not force:
        logger.info("Compatible model artifacts already exist; skipping training.")
        return
    logger.info("Model artifacts are absent or stale; running the explicit bootstrap training job.")
    train()


if __name__ == "__main__":
    main()
