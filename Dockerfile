# Shared dependencies; training and serving are separate image targets/processes.
FROM python:3.11-slim AS dependencies

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements-runtime.txt ./requirements-runtime.txt
RUN python -m pip install --no-cache-dir --upgrade pip && \
    python -m pip install --no-cache-dir -r requirements-runtime.txt

# One-shot training target used by the Compose model-bootstrap service.
FROM dependencies AS training
COPY . /app
RUN groupadd --system --gid 10001 churn && \
    useradd --system --uid 10001 --gid churn --create-home churn && \
    mkdir -p /models /app/logs /app/reports && \
    chown -R churn:churn /models /app
USER churn
ENV CHURN_MODEL_DIR=/models \
    CHURN_MODEL_PATH=/models/churn_model.pkl \
    CHURN_METADATA_PATH=/models/model_metadata.json \
    CHURN_PREDICTION_LOG_PATH=/app/logs/predictions.jsonl
CMD ["python", "-m", "src.bootstrap"]

# Inference image has no training step. Model artifacts arrive through a named volume.
FROM dependencies AS runtime
COPY . /app
RUN groupadd --system --gid 10001 churn && \
    useradd --system --uid 10001 --gid churn --create-home churn && \
    mkdir -p /app/logs /models && chown -R churn:churn /app /models
USER churn
ENV CHURN_MODEL_DIR=/models \
    CHURN_MODEL_PATH=/models/churn_model.pkl \
    CHURN_METADATA_PATH=/models/model_metadata.json \
    CHURN_PREDICTION_LOG_PATH=/app/logs/predictions.jsonl
EXPOSE 8000 8501
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
