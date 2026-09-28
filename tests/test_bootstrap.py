"""Tests for idempotent Docker model-volume initialization."""

import json

from config import FEATURES_VERSION, MODEL_VERSION
from src import bootstrap


def test_bootstrap_requires_current_model_and_metadata(tmp_path, monkeypatch):
    """Existing matching artifacts skip redundant model training on container restart."""
    model_path = tmp_path / "churn_model.pkl"
    metadata_path = tmp_path / "model_metadata.json"
    monkeypatch.setattr(bootstrap, "MODEL_PATH", model_path)
    monkeypatch.setattr(bootstrap, "METADATA_PATH", metadata_path)
    assert not bootstrap.artifacts_are_current()
    model_path.write_bytes(b"model-placeholder")
    metadata_path.write_text(
        json.dumps({"model_version": MODEL_VERSION, "features_version": FEATURES_VERSION}),
        encoding="utf-8",
    )
    assert bootstrap.artifacts_are_current()
    metadata_path.write_text(json.dumps({"model_version": "old"}), encoding="utf-8")
    assert not bootstrap.artifacts_are_current()
