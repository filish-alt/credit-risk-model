"""Pydantic schemas for the credit risk prediction API."""

from pydantic import BaseModel, Field


class TransactionRecord(BaseModel):
    TransactionId: str
    BatchId: str
    AccountId: str
    SubscriptionId: str
    CustomerId: str
    CurrencyCode: str
    CountryCode: int
    ProductId: str
    ProductCategory: str
    ChannelId: str
    Amount: float
    Value: int
    TransactionStartTime: str
    PricingStrategy: int


class PredictionRequest(BaseModel):
    transactions: list[TransactionRecord] = Field(..., min_length=1)


class CustomerPrediction(BaseModel):
    CustomerId: str
    risk_probability: float = Field(..., ge=0.0, le=1.0)
    is_high_risk_predicted: int = Field(..., ge=0, le=1)


class PredictionResponse(BaseModel):
    predictions: list[CustomerPrediction]
