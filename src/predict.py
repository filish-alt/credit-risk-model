"""Batch inference helper for the credit risk model."""

from __future__ import annotations

from pathlib import Path

import mlflow.sklearn
import pandas as pd

from src.data_processing import get_feature_matrix, process_raw_data

REGISTERED_MODEL_NAME = "credit_risk_best_model"


def load_model(model_uri: str | None = None):
    uri = model_uri or f"models:/{REGISTERED_MODEL_NAME}/latest"
    return mlflow.sklearn.load_model(uri)


def predict_risk(
    raw_df: pd.DataFrame,
    model_uri: str | None = None,
) -> pd.DataFrame:
    """Process raw transactions and return customer-level risk probabilities."""
    processed = process_raw_data(raw_df)
    X, _ = get_feature_matrix(processed)
    model = load_model(model_uri)
    probabilities = model.predict_proba(X)[:, 1]

    return pd.DataFrame(
        {
            "CustomerId": processed["CustomerId"],
            "risk_probability": probabilities,
            "is_high_risk_predicted": (probabilities >= 0.5).astype(int),
        }
    )


def predict_from_csv(
    raw_path: str | Path,
    output_path: str | Path | None = None,
    model_uri: str | None = None,
) -> pd.DataFrame:
    raw_df = pd.read_csv(raw_path)
    predictions = predict_risk(raw_df, model_uri=model_uri)

    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(path, index=False)

    return predictions
