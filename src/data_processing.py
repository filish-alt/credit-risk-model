"""Feature engineering pipeline for credit risk modeling."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler

CUSTOMER_ID_COLUMN = "CustomerId"
DATETIME_COLUMN = "TransactionStartTime"
CATEGORICAL_COLUMNS = ["ProductCategory", "ChannelId", "CurrencyCode"]
AGGREGATE_NUMERIC_COLUMNS = [
    "total_amount",
    "avg_amount",
    "transaction_count",
    "std_amount",
]
DATETIME_FEATURE_COLUMNS = [
    "transaction_hour",
    "transaction_day",
    "transaction_month",
    "transaction_year",
]
DEFAULT_RANDOM_STATE = 42


def _ensure_dataframe(X) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X.copy()
    raise TypeError("Expected a pandas DataFrame.")


class DateTimeFeatureExtractor(BaseEstimator, TransformerMixin):
    """Parse transaction timestamps and extract calendar features."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = _ensure_dataframe(X)
        df[DATETIME_COLUMN] = pd.to_datetime(df[DATETIME_COLUMN], utc=True, errors="coerce")
        df["transaction_hour"] = df[DATETIME_COLUMN].dt.hour
        df["transaction_day"] = df[DATETIME_COLUMN].dt.day
        df["transaction_month"] = df[DATETIME_COLUMN].dt.month
        df["transaction_year"] = df[DATETIME_COLUMN].dt.year
        return df


class CustomerAggregator(BaseEstimator, TransformerMixin):
    """Aggregate transaction-level records to customer-level features."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = _ensure_dataframe(X)

        agg = df.groupby(CUSTOMER_ID_COLUMN).agg(
            total_amount=("Amount", "sum"),
            avg_amount=("Amount", "mean"),
            transaction_count=("Amount", "count"),
            std_amount=("Amount", "std"),
            transaction_hour=("transaction_hour", "mean"),
            transaction_day=("transaction_day", "mean"),
            transaction_month=("transaction_month", "mean"),
            transaction_year=("transaction_year", "mean"),
            ProductCategory=("ProductCategory", lambda s: s.mode().iloc[0]),
            ChannelId=("ChannelId", lambda s: s.mode().iloc[0]),
            CurrencyCode=("CurrencyCode", lambda s: s.mode().iloc[0]),
            last_transaction_time=(DATETIME_COLUMN, "max"),
        )
        agg["std_amount"] = agg["std_amount"].fillna(0.0)
        return agg.reset_index()


class WoEEncoder(BaseEstimator, TransformerMixin):
    """Weight of Evidence encoder for categorical columns against a binary target."""

    def __init__(self, categorical_columns: Iterable[str], target_column: str):
        self.categorical_columns = list(categorical_columns)
        self.target_column = target_column
        self.woe_maps_: dict[str, dict[str, float]] = {}
        self.iv_scores_: dict[str, float] = {}

    def fit(self, X, y=None):
        df = _ensure_dataframe(X)
        target = df[self.target_column] if y is None else y

        for col in self.categorical_columns:
            if col not in df.columns:
                continue
            woe_map, iv = self._fit_column(df[col], target)
            self.woe_maps_[col] = woe_map
            self.iv_scores_[col] = iv
        return self

    def transform(self, X):
        df = _ensure_dataframe(X)
        for col, woe_map in self.woe_maps_.items():
            if col not in df.columns:
                continue
            default_woe = float(np.mean(list(woe_map.values()))) if woe_map else 0.0
            df[f"{col}_woe"] = df[col].astype(str).map(woe_map).fillna(default_woe)
        return df

    @staticmethod
    def _fit_column(series: pd.Series, target: pd.Series) -> tuple[dict[str, float], float]:
        data = pd.DataFrame({"feature": series.astype(str), "target": target})
        total_good = max((data["target"] == 0).sum(), 1)
        total_bad = max((data["target"] == 1).sum(), 1)

        woe_map: dict[str, float] = {}
        iv = 0.0
        for category, group in data.groupby("feature"):
            good = max((group["target"] == 0).sum(), 0.5)
            bad = max((group["target"] == 1).sum(), 0.5)
            dist_good = good / total_good
            dist_bad = bad / total_bad
            woe = np.log(dist_good / dist_bad)
            woe_map[category] = float(woe)
            iv += (dist_good - dist_bad) * woe
        return woe_map, float(iv)


class FeaturePreprocessor(BaseEstimator, TransformerMixin):
    """Impute, encode, scale, and optionally apply WoE on customer-level data."""

    def __init__(
        self,
        scale_method: str = "standard",
        use_woe: bool = False,
        target_column: Optional[str] = None,
    ):
        self.scale_method = scale_method
        self.use_woe = use_woe
        self.target_column = target_column
        self.woe_encoder_: Optional[WoEEncoder] = None
        self.one_hot_encoder_ = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.preprocessor_: Optional[Pipeline] = None
        self.feature_columns_: list[str] = []
        self.one_hot_columns_: list[str] = []

    def fit(self, X, y=None):
        df = _ensure_dataframe(X)
        working = df.copy()

        if self.use_woe and self.target_column:
            self.woe_encoder_ = WoEEncoder(CATEGORICAL_COLUMNS, self.target_column)
            self.woe_encoder_.fit(working)
            working = self.woe_encoder_.transform(working)

        cat_present = [col for col in CATEGORICAL_COLUMNS if col in working.columns]
        if cat_present:
            self.one_hot_encoder_.fit(working[cat_present])
            self.one_hot_columns_ = list(
                self.one_hot_encoder_.get_feature_names_out(cat_present)
            )

        numeric_cols = (
            AGGREGATE_NUMERIC_COLUMNS
            + DATETIME_FEATURE_COLUMNS
            + [
                f"{col}_woe"
                for col in CATEGORICAL_COLUMNS
                if f"{col}_woe" in working.columns
            ]
        )
        numeric_cols = [col for col in numeric_cols if col in working.columns]

        if cat_present and self.one_hot_columns_:
            encoded = self.one_hot_encoder_.transform(working[cat_present])
            encoded_df = pd.DataFrame(
                encoded, columns=self.one_hot_columns_, index=working.index
            )
            working = pd.concat([working, encoded_df], axis=1)

        self.feature_columns_ = numeric_cols + self.one_hot_columns_
        scaler = (
            StandardScaler() if self.scale_method == "standard" else MinMaxScaler()
        )
        self.preprocessor_ = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", scaler),
            ]
        )
        self.preprocessor_.fit(working[self.feature_columns_])
        return self

    def transform(self, X):
        if self.preprocessor_ is None:
            raise RuntimeError("FeaturePreprocessor must be fitted before transform.")

        df = _ensure_dataframe(X)
        working = df.copy()
        if self.use_woe and self.woe_encoder_ is not None:
            working = self.woe_encoder_.transform(working)

        cat_present = [col for col in CATEGORICAL_COLUMNS if col in working.columns]
        if cat_present and self.one_hot_columns_:
            encoded = self.one_hot_encoder_.transform(working[cat_present])
            encoded_df = pd.DataFrame(
                encoded, columns=self.one_hot_columns_, index=working.index
            )
            working = pd.concat([working, encoded_df], axis=1)

        scaled = self.preprocessor_.transform(working[self.feature_columns_])
        scaled_df = pd.DataFrame(scaled, columns=self.feature_columns_, index=working.index)

        return pd.concat(
            [
                working[[CUSTOMER_ID_COLUMN]].reset_index(drop=True),
                scaled_df.reset_index(drop=True),
            ],
            axis=1,
        )


class CreditDataProcessor(BaseEstimator, TransformerMixin):
    """End-to-end processor from raw transactions to model-ready customer data."""

    def __init__(self, scale_method: str = "standard", use_woe: bool = False):
        self.scale_method = scale_method
        self.use_woe = use_woe
        self.pipeline_ = Pipeline(
            [
                ("datetime", DateTimeFeatureExtractor()),
                ("aggregate", CustomerAggregator()),
                (
                    "preprocess",
                    FeaturePreprocessor(scale_method=scale_method, use_woe=use_woe),
                ),
            ]
        )

    def fit(self, X, y=None):
        self.pipeline_.fit(X, y)
        return self

    def transform(self, X):
        return self.pipeline_.transform(X)

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    @property
    def pipeline(self) -> Pipeline:
        return self.pipeline_


def build_processing_pipeline(
    scale_method: str = "standard",
    use_woe: bool = False,
) -> Pipeline:
    """Return the sklearn Pipeline for credit data processing."""
    return CreditDataProcessor(scale_method=scale_method, use_woe=use_woe).pipeline


def process_raw_data(
    raw_df: pd.DataFrame,
    scale_method: str = "standard",
    use_woe: bool = False,
) -> pd.DataFrame:
    """Transform raw transaction data into a model-ready customer-level DataFrame."""
    processor = CreditDataProcessor(scale_method=scale_method, use_woe=use_woe)
    return processor.fit_transform(raw_df)


def load_and_process(
    raw_path: str | Path,
    processed_path: Optional[str | Path] = None,
    **kwargs,
) -> pd.DataFrame:
    """Load raw CSV data, process it, and optionally persist the result."""
    raw_df = pd.read_csv(raw_path)
    processed = process_raw_data(raw_df, **kwargs)

    if processed_path is not None:
        path = Path(processed_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        processed.to_csv(path, index=False)

    return processed
