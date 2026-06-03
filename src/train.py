"""Model training, hyperparameter tuning, and MLflow experiment tracking."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier

from src.data_processing import (
    DEFAULT_RANDOM_STATE,
    get_feature_matrix,
    load_and_process,
)

EXPERIMENT_NAME = "credit_risk_modeling"
REGISTERED_MODEL_NAME = "credit_risk_best_model"


def _evaluate_model(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
    }
    if len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
    else:
        metrics["roc_auc"] = 0.0
    return metrics


def _get_cv_folds(y: pd.Series, max_cv: int = 3) -> int:
    """Return a safe number of CV folds for hyperparameter search."""
    if y.empty:
        return 0
    min_class = int(y.value_counts().min())
    n_samples = len(y)
    if min_class < 2 or n_samples < 4:
        return 0
    return min(max_cv, min_class, n_samples)


def _fit_with_search(
    estimator,
    param_grid: dict,
    search_type: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int,
):
    cv_folds = _get_cv_folds(y_train)
    if cv_folds == 0:
        estimator.fit(X_train, y_train)
        return estimator, {}

    if search_type == "grid":
        search = GridSearchCV(
            estimator,
            param_grid,
            cv=cv_folds,
            scoring="f1",
        )
    else:
        search = RandomizedSearchCV(
            estimator,
            param_grid,
            n_iter=min(4, max(len(v) for v in param_grid.values())),
            cv=cv_folds,
            scoring="f1",
            random_state=random_state,
        )
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_


def _get_model_candidates(random_state: int) -> dict[str, Any]:
    return {
        "logistic_regression": (
            LogisticRegression(max_iter=1000, random_state=random_state),
            {"C": [0.01, 0.1, 1.0, 10.0]},
            "grid",
        ),
        "decision_tree": (
            DecisionTreeClassifier(random_state=random_state),
            {"max_depth": [2, 3, 5, None], "min_samples_split": [2, 5]},
            "grid",
        ),
        "random_forest": (
            RandomForestClassifier(random_state=random_state),
            {"n_estimators": [50, 100], "max_depth": [2, 3, 5]},
            "random",
        ),
        "gradient_boosting": (
            GradientBoostingClassifier(random_state=random_state),
            {"n_estimators": [50, 100], "learning_rate": [0.05, 0.1], "max_depth": [2, 3]},
            "random",
        ),
    }


def prepare_data(
    raw_path: str | Path,
    test_size: float = 0.2,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    """Load raw data, process features, and split into train/test sets."""
    processed = load_and_process(raw_path)
    X, y = get_feature_matrix(processed)

    stratify = y if y.nunique() > 1 and y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )
    return X_train, X_test, y_train, y_test, processed


def train_and_track(
    raw_path: str | Path = "data/raw/data.csv",
    test_size: float = 0.2,
    random_state: int = DEFAULT_RANDOM_STATE,
    register_model: bool = True,
) -> tuple[str, dict[str, float]]:
    """Train multiple models, log experiments to MLflow, and register the best model."""
    mlflow.set_experiment(EXPERIMENT_NAME)

    X_train, X_test, y_train, y_test, processed = prepare_data(
        raw_path=raw_path,
        test_size=test_size,
        random_state=random_state,
    )

    best_run_name = ""
    best_metrics: dict[str, float] = {}
    best_f1 = -1.0
    best_model = None

    for model_name, (estimator, param_grid, search_type) in _get_model_candidates(
        random_state
    ).items():
        with mlflow.start_run(run_name=model_name):
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("random_state", random_state)
            mlflow.log_param("test_size", test_size)

            model, best_params = _fit_with_search(
                estimator,
                param_grid,
                search_type,
                X_train,
                y_train,
                random_state,
            )
            y_pred = model.predict(X_test)
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test)[:, 1]
            else:
                y_prob = model.decision_function(X_test)

            metrics = _evaluate_model(y_test.to_numpy(), y_pred, y_prob)
            mlflow.log_params(best_params)
            for metric_name, value in metrics.items():
                mlflow.log_metric(metric_name, value)

            signature = infer_signature(X_train, y_pred)
            mlflow.sklearn.log_model(model, artifact_path="model", signature=signature)

            if metrics["f1_score"] > best_f1:
                best_f1 = metrics["f1_score"]
                best_run_name = model_name
                best_metrics = metrics
                best_model = model

    if register_model and best_model is not None:
        with mlflow.start_run(run_name=f"best_{best_run_name}"):
            mlflow.log_param("selected_model", best_run_name)
            for metric_name, value in best_metrics.items():
                mlflow.log_metric(metric_name, value)
            signature = infer_signature(X_train, best_model.predict(X_train))
            mlflow.sklearn.log_model(
                best_model,
                artifact_path="model",
                signature=signature,
                registered_model_name=REGISTERED_MODEL_NAME,
            )

    processed_path = Path("data/processed/processed_data.csv")
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    processed.to_csv(processed_path, index=False)

    return best_run_name, best_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train credit risk models with MLflow tracking.")
    parser.add_argument("--raw-path", default="data/raw/data.csv")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    parser.add_argument("--no-register", action="store_true")
    args = parser.parse_args()

    best_model, metrics = train_and_track(
        raw_path=args.raw_path,
        test_size=args.test_size,
        random_state=args.random_state,
        register_model=not args.no_register,
    )
    print(f"Best model: {best_model}")
    print(f"Metrics: {metrics}")


if __name__ == "__main__":
    main()
