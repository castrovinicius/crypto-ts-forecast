"""Nodes for model training pipeline.

This module contains functions to train and evaluate Prophet models.
"""

import logging
from typing import Any

import pandas as pd
from prophet import Prophet

logger = logging.getLogger(__name__)


def train_prophet_model(
    train_data: pd.DataFrame,
    seasonality_mode: str,
    yearly_seasonality: bool,
    weekly_seasonality: bool,
    daily_seasonality: bool,
    changepoint_prior_scale: float,
    seasonality_prior_scale: float,
    add_volume_regressor: bool,
    changepoint_range: float,
) -> Prophet:
    """Train a Prophet model on the training data.

    Args:
        train_data: Training dataset with ds and y columns.
        seasonality_mode: 'additive' or 'multiplicative'.
        yearly_seasonality: Whether to include yearly seasonality.
        weekly_seasonality: Whether to include weekly seasonality.
        daily_seasonality: Whether to include daily seasonality.
        changepoint_prior_scale: Flexibility of trend changes.
        seasonality_prior_scale: Flexibility of seasonality.
        add_volume_regressor: Whether to add volume as regressor.
        changepoint_range: Proportion of data to consider for changepoints.

    Returns:
        Trained Prophet model.
    """
    logger.info("Initializing Prophet model...")

    model = Prophet(
        seasonality_mode=seasonality_mode,
        yearly_seasonality=yearly_seasonality,
        weekly_seasonality=weekly_seasonality,
        daily_seasonality=daily_seasonality,
        changepoint_prior_scale=changepoint_prior_scale,
        seasonality_prior_scale=seasonality_prior_scale,
        changepoint_range=changepoint_range,
    )

    # Add volume as regressor if available
    if add_volume_regressor and "volume" in train_data.columns:
        model.add_regressor("volume")
        logger.info("Added volume as regressor")

    # Add crypto-specific seasonality (4-year cycle related to Bitcoin halving)
    model.add_seasonality(
        name="halving_cycle",
        period=365.25 * 4,  # ~4 years
        fourier_order=3,
    )
    logger.info("Added Bitcoin halving cycle seasonality (4 years)")

    logger.info(f"Training Prophet model on {len(train_data)} samples...")
    model.fit(train_data)

    logger.info("Prophet model training completed")

    return model


def evaluate_model(
    model: Prophet,
    test_data: pd.DataFrame,
    add_volume_regressor: bool,
) -> dict[str, Any]:
    """Evaluate the trained model on test data.

    Args:
        model: Trained Prophet model.
        test_data: Test dataset with ds and y columns.
        add_volume_regressor: Whether volume was used as regressor.

    Returns:
        Dictionary with evaluation metrics.
    """
    logger.info("Evaluating model on test data...")

    # Prepare future dataframe for test period
    future = test_data[["ds"]].copy()

    if add_volume_regressor and "volume" in test_data.columns:
        future["volume"] = test_data["volume"].values

    # Make predictions
    forecast = model.predict(future)

    # Calculate metrics
    y_true = test_data["y"].values
    y_pred = forecast["yhat"].values

    # Mean Absolute Error
    mae = float(abs(y_true - y_pred).mean())

    # Mean Absolute Percentage Error
    # Use epsilon to avoid division by zero
    epsilon = 1e-10
    mape = float((abs(y_true - y_pred) / (abs(y_true) + epsilon)).mean() * 100)

    # Root Mean Squared Error
    rmse = float(((y_true - y_pred) ** 2).mean() ** 0.5)

    # R-squared
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    r2 = float(1 - (ss_res / ss_tot))

    metrics = {
        "mae": mae,
        "mape": mape,
        "rmse": rmse,
        "r2": r2,
        "test_samples": len(test_data),
        "test_start_date": str(test_data["ds"].min()),
        "test_end_date": str(test_data["ds"].max()),
    }

    logger.info(f"Model evaluation results:")
    logger.info(f"  MAE: ${mae:,.2f}")
    logger.info(f"  MAPE: {mape:.2f}%")
    logger.info(f"  RMSE: ${rmse:,.2f}")
    logger.info(f"  R²: {r2:.4f}")

    return metrics


def create_model_report(
    metrics: dict[str, Any],
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
) -> dict[str, Any]:
    """Create a comprehensive model report.

    Args:
        metrics: Evaluation metrics.
        train_data: Training data for additional stats.
        test_data: Test data for additional stats.

    Returns:
        Comprehensive report dictionary.
    """
    report = {
        "model_type": "Prophet",
        "training_info": {
            "samples": len(train_data),
            "start_date": str(train_data["ds"].min()),
            "end_date": str(train_data["ds"].max()),
            "price_range": {
                "min": float(train_data["y"].min()),
                "max": float(train_data["y"].max()),
                "mean": float(train_data["y"].mean()),
            },
        },
        "test_info": {
            "samples": len(test_data),
            "start_date": str(test_data["ds"].min()),
            "end_date": str(test_data["ds"].max()),
        },
        "metrics": metrics,
    }

    logger.info("Model report created successfully")

    return report
