"""FastAPI service for credit risk predictions."""

from __future__ import annotations

import os
from functools import lru_cache

import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException

from src.api.pydantic_models import (
    CustomerPrediction,
    PredictionRequest,
    PredictionResponse,
)
from src.data_processing import process_raw_data
from src.predict import REGISTERED_MODEL_NAME

app = FastAPI(
    title="Credit Risk API",
    description="Predict customer credit risk from transaction history.",
    version="1.0.0",
)


@lru_cache(maxsize=1)
def get_model():
    model_uri = os.getenv("MLFLOW_MODEL_URI", f"models:/{REGISTERED_MODEL_NAME}/latest")
    try:
        return mlflow.sklearn.load_model(model_uri)
    except Exception as exc:
        fallback_uri = os.getenv("MLFLOW_FALLBACK_MODEL_URI")
        if fallback_uri:
            return mlflow.sklearn.load_model(fallback_uri)
        raise exc


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    try:
        raw_df = pd.DataFrame([record.model_dump() for record in request.transactions])
        model = get_model()
        processed = process_raw_data(raw_df)
        feature_cols = [
            col for col in processed.columns if col not in {"CustomerId", "is_high_risk"}
        ]
        probabilities = model.predict_proba(processed[feature_cols])[:, 1]

        predictions = [
            CustomerPrediction(
                CustomerId=customer_id,
                risk_probability=float(probability),
                is_high_risk_predicted=int(probability >= 0.5),
            )
            for customer_id, probability in zip(processed["CustomerId"], probabilities)
        ]
        return PredictionResponse(predictions=predictions)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
