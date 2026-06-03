"""Unit tests for feature engineering helpers."""

import pandas as pd

from src.data_processing import CustomerAggregator, DateTimeFeatureExtractor, process_raw_data


def _sample_transactions(n_customers: int = 5, rows_per_customer: int = 4) -> pd.DataFrame:
    rows = []
    for customer_idx in range(n_customers):
        customer_id = f"CustomerId_{customer_idx}"
        for tx_idx in range(rows_per_customer):
            rows.append(
                {
                    "TransactionId": f"TransactionId_{customer_idx}_{tx_idx}",
                    "BatchId": f"BatchId_{customer_idx}",
                    "AccountId": f"AccountId_{customer_idx}",
                    "SubscriptionId": f"SubscriptionId_{customer_idx}",
                    "CustomerId": customer_id,
                    "CurrencyCode": "UGX",
                    "CountryCode": 256,
                    "ProductId": f"ProductId_{tx_idx % 2}",
                    "ProductCategory": (
                        ["airtime", "utility_bill", "financial_services"][tx_idx % 3]
                    ),
                    "ChannelId": f"ChannelId_{tx_idx % 2 + 1}",
                    "Amount": float(100 * (tx_idx + 1) + customer_idx * 50),
                    "Value": int(100 * (tx_idx + 1)),
                    "TransactionStartTime": f"2018-11-{10 + tx_idx:02d}T0{tx_idx}:00:00Z",
                    "PricingStrategy": 2,
                }
            )
    return pd.DataFrame(rows)


def test_datetime_feature_extractor_adds_expected_columns():
    raw_df = _sample_transactions(n_customers=1, rows_per_customer=2)
    result = DateTimeFeatureExtractor().transform(raw_df)
    expected = {"transaction_hour", "transaction_day", "transaction_month", "transaction_year"}
    assert expected.issubset(result.columns)


def test_customer_aggregator_returns_expected_aggregate_columns():
    raw_df = _sample_transactions(n_customers=3, rows_per_customer=3)
    enriched = DateTimeFeatureExtractor().transform(raw_df)
    aggregated = CustomerAggregator().transform(enriched)
    expected_columns = {
        "CustomerId",
        "total_amount",
        "avg_amount",
        "transaction_count",
        "std_amount",
    }
    assert expected_columns.issubset(set(aggregated.columns))
    assert len(aggregated) == 3


def test_process_raw_data_returns_model_ready_customer_features():
    raw_df = _sample_transactions(n_customers=4, rows_per_customer=4)
    processed = process_raw_data(raw_df)
    assert "CustomerId" in processed.columns
    assert len(processed) == 4
    assert processed.isnull().sum().sum() == 0
