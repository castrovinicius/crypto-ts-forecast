"""Pydantic schemas for API request/response models."""

import datetime as dt
from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="API status")
    version: str = Field(..., description="API version")
    timestamp: dt.datetime = Field(..., description="Current timestamp")


class PipelineRunRequest(BaseModel):
    """Request to run a pipeline."""

    pipeline_name: str = Field(
        default="__default__",
        description="Name of the pipeline to run",
    )
    params: dict[str, Any] | None = Field(
        default=None,
        description="Optional parameter overrides",
    )


class PipelineRunResponse(BaseModel):
    """Response after running a pipeline."""

    status: str = Field(..., description="Pipeline execution status")
    pipeline_name: str = Field(..., description="Name of the pipeline executed")
    message: str = Field(..., description="Execution message")
    duration_seconds: float | None = Field(
        default=None,
        description="Execution duration in seconds",
    )


class ForecastRequest(BaseModel):
    """Request for generating forecast."""

    days_ahead: int = Field(
        default=30,
        ge=1,
        le=365,
        description="Number of days to forecast",
    )
    retrain: bool = Field(
        default=False,
        description="Whether to retrain the model before forecasting",
    )


class PredictionPoint(BaseModel):
    """Single prediction point."""

    prediction_date: dt.date = Field(..., description="Prediction date")
    predicted_price: float = Field(..., description="Predicted price in USD")
    predicted_price_lower: float = Field(
        ..., description="Lower bound of prediction interval"
    )
    predicted_price_upper: float = Field(
        ..., description="Upper bound of prediction interval"
    )
    trend: float = Field(..., description="Trend component")


class ForecastResponse(BaseModel):
    """Forecast response with predictions."""

    status: str = Field(..., description="Forecast status")
    symbol: str = Field(..., description="Trading pair symbol")
    last_historical_date: dt.date = Field(
        ..., description="Last date of historical data"
    )
    last_historical_price: float = Field(..., description="Last known price")
    forecast_days: int = Field(..., description="Number of days forecasted")
    predictions: list[PredictionPoint] = Field(..., description="List of predictions")
    summary: dict[str, Any] = Field(..., description="Forecast summary statistics")


class ModelInfoResponse(BaseModel):
    """Model information response."""

    model_exists: bool = Field(..., description="Whether a trained model exists")
    model_type: str | None = Field(default=None, description="Type of model")
    training_date: str | None = Field(default=None, description="Date of last training")
    metrics: dict[str, Any] | None = Field(
        default=None, description="Model evaluation metrics"
    )
    training_info: dict[str, Any] | None = Field(
        default=None, description="Training data information"
    )


class ErrorResponse(BaseModel):
    """Error response."""

    status: str = Field(default="error", description="Error status")
    message: str = Field(..., description="Error message")
    detail: str | None = Field(default=None, description="Error details")


class CurrentPriceResponse(BaseModel):
    """Current price response from Binance."""

    symbol: str = Field(..., description="Trading pair symbol")
    price: float = Field(..., description="Current price in USD")
    timestamp: dt.datetime = Field(..., description="Price timestamp")
