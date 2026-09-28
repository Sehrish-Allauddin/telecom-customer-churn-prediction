"""FastAPI endpoints for health, model details, and churn predictions."""

import json
import logging
from functools import lru_cache
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException

from api.schemas import (
    BatchPredictionItem,
    BatchPredictionRequest,
    BatchPredictionResponse,
    CustomerRequest,
    PredictionResponse,
)
from config import (
    HIGH_RISK_THRESHOLD,
    MEDIUM_RISK_THRESHOLD,
    METADATA_PATH,
    MODEL_VERSION,
)
from src.logging_config import configure_logging
from src.monitoring import log_prediction
from src.predict import load_model, predict_batch, predict_customer

configure_logging()
logger = logging.getLogger(__name__)
app = FastAPI(title="Customer Churn Prediction API", version=MODEL_VERSION)


@lru_cache(maxsize=1)
def get_pipeline() -> Any:
    """Load the compatible fitted pipeline once per API process."""
    return load_model()


def read_metadata() -> dict[str, Any]:
    """Read model metadata without returning local filesystem paths."""
    if not METADATA_PATH.exists():
        raise HTTPException(status_code=503, detail="Model metadata is unavailable.")
    try:
        metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        logger.exception("Model metadata is unavailable or invalid.")
        raise HTTPException(status_code=503, detail="Model metadata is unavailable.") from error
    return metadata


@app.get("/health")
def health() -> dict[str, str]:
    """Report service health and whether the version-compatible model can load."""
    try:
        get_pipeline()
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        logger.error("Health check could not load a valid model: %s", error)
        raise HTTPException(status_code=503, detail="Model is unavailable.") from error
    return {"status": "ok"}


@app.get("/model-info")
def model_info() -> dict[str, Any]:
    """Return non-sensitive model identity and evaluation metadata."""
    metadata = read_metadata()
    return {
        "model_name": metadata.get("model_name"),
        "model_version": metadata.get("model_version", MODEL_VERSION),
        "model_type": metadata.get("model_type"),
        "features_version": metadata.get("features_version"),
        "metrics": metadata.get("metrics", {}),
        "training_date": metadata.get("training_date"),
        "selection_criterion": metadata.get("selection_criterion"),
        "threshold": metadata.get("threshold", {}),
        "cross_validation": metadata.get("cross_validation", {}),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: CustomerRequest) -> PredictionResponse:
    """Return class, churn probability, and risk band for a validated payload."""
    try:
        model = get_pipeline()
        prediction, probability = predict_customer(request.to_model_features(), model)
    except (FileNotFoundError, RuntimeError) as error:
        logger.exception("Prediction service has no compatible model.")
        raise HTTPException(status_code=503, detail="Prediction model is unavailable.") from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    risk_level = (
        "High Risk"
        if probability >= HIGH_RISK_THRESHOLD
        else "Medium Risk" if probability >= MEDIUM_RISK_THRESHOLD else "Low Risk"
    )
    logger.info(
        "Prediction completed: model_version=%s prediction=%s risk_level=%s",
        MODEL_VERSION,
        prediction,
        risk_level,
    )
    log_prediction(request.to_model_features(), prediction, probability, risk_level)
    return PredictionResponse(
        prediction=prediction,
        churn_probability=probability,
        risk_level=risk_level,
        model_version=MODEL_VERSION,
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch_route(request: BatchPredictionRequest) -> BatchPredictionResponse:
    """Score up to 1,000 validated customers without returning raw feature values."""
    try:
        model = get_pipeline()
        rows = []
        for customer in request.customers:
            row = customer.to_model_features()
            if customer.customer_id is not None:
                row["customerID"] = customer.customer_id
            rows.append(row)
        results = predict_batch(pd.DataFrame(rows), model)
    except (FileNotFoundError, RuntimeError) as error:
        logger.exception("Batch prediction service has no compatible model.")
        raise HTTPException(status_code=503, detail="Prediction model is unavailable.") from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    response_rows = []
    for source, result in zip(request.customers, results.to_dict(orient="records"), strict=True):
        probability = float(result["churn_probability"])
        prediction = int(result["predicted_churn"])
        risk = str(result["risk_level"])
        log_prediction(source.to_model_features(), prediction, probability, risk)
        response_rows.append(
            BatchPredictionItem(
                customer_id=source.customer_id,
                prediction=prediction,
                churn_probability=probability,
                risk_level=risk,
                model_version=MODEL_VERSION,
            )
        )
    return BatchPredictionResponse(predictions=response_rows, model_version=MODEL_VERSION)
