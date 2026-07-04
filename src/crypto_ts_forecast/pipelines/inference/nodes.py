"""Nodes for inference pipeline.

This module contains functions to generate forecasts using trained Prophet models.
"""

import logging
from typing import Any

import pandas as pd
from prophet import Prophet

from ..model_training.nodes import prepare_future_regressors

logger = logging.getLogger(__name__)


def create_future_dataframe(
    model: Prophet,
    prophet_data: pd.DataFrame,
    forecast_days: int,
    ma_window: int = 21,
    strategy: str = "ma",
) -> pd.DataFrame:
    """Create a future dataframe for making predictions.

    Args:
        model: Trained Prophet model.
        prophet_data: Historical data used for training.
        forecast_days: Number of days to forecast into the future.
        ma_window: Window size for moving averages (default: 21).
        strategy: Strategy to estimate future regressor values.
            'ma' uses the MA-{ma_window} of the last known values (default).
            'last_value' repeats the last known value.
            'trend' projects a linear trend forward.

    Returns:
        DataFrame ready for predictions with future regressor values.
    """
    logger.info(f"Creating future dataframe for {forecast_days} days ahead...")

    # Create future dates — hourly frequency, forecast_days × 24 periods
    last_date = prophet_data["ds"].max()
    freq = pd.infer_freq(prophet_data["ds"].tail(10)) or "h"
    is_hourly = "h" in freq.lower() or "H" in freq
    if is_hourly:
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(hours=1),
            periods=forecast_days * 24,
            freq="h",
        )
    else:
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=forecast_days,
            freq="D",
        )

    # Prepare future regressors using the chosen strategy
    future_df = prepare_future_regressors(
        train_data=prophet_data,
        future_dates=future_dates,
        ma_window=ma_window,
        strategy=strategy,
    )

    logger.info(f"Future dataframe created with {len(future_df)} rows")
    logger.info(f"Columns: {list(future_df.columns)}")

    return future_df


def generate_forecast(
    model: Prophet,
    future_df: pd.DataFrame,
) -> pd.DataFrame:
    """Generate forecast using the trained model.

    Args:
        model: Trained Prophet model.
        future_df: Future dataframe with dates to predict.

    Returns:
        DataFrame with predictions including uncertainty intervals.
    """
    logger.info("Generating forecast...")

    forecast = model.predict(future_df)

    # Select relevant columns
    forecast_output = forecast[
        [
            "ds",
            "yhat",
            "yhat_lower",
            "yhat_upper",
            "trend",
            "trend_lower",
            "trend_upper",
        ]
    ].copy()

    # Rename columns for clarity
    forecast_output = forecast_output.rename(
        columns={
            "yhat": "predicted_price",
            "yhat_lower": "predicted_price_lower",
            "yhat_upper": "predicted_price_upper",
        }
    )

    logger.info(
        f"Forecast generated from {forecast_output['ds'].min()} "
        f"to {forecast_output['ds'].max()}"
    )

    return forecast_output


def extract_future_predictions(
    forecast: pd.DataFrame,
    prophet_data: pd.DataFrame,
) -> pd.DataFrame:
    """Extract only the future predictions (dates beyond historical data).

    Args:
        forecast: Full forecast including historical fitted values.
        prophet_data: Historical data to determine cutoff date.

    Returns:
        DataFrame with only future predictions.
    """
    last_historical_date = prophet_data["ds"].max()

    future_only = forecast[forecast["ds"] > last_historical_date].copy()
    future_only = future_only.reset_index(drop=True)

    logger.info(
        f"Extracted {len(future_only)} future predictions "
        f"from {future_only['ds'].min()} to {future_only['ds'].max()}"
    )

    return future_only


def create_forecast_summary(
    future_predictions: pd.DataFrame,
    prophet_data: pd.DataFrame,
) -> dict[str, Any]:
    """Create a summary of the forecast.

    Args:
        future_predictions: Future prediction data.
        prophet_data: Historical data for context.

    Returns:
        Summary dictionary with key statistics.
    """
    last_price = float(prophet_data["y"].iloc[-1])
    last_date = str(prophet_data["ds"].iloc[-1].date())

    # Handle empty predictions
    if future_predictions.empty:
        logger.warning("No future predictions available")
        return {
            "last_historical_price": last_price,
            "last_historical_date": last_date,
            "forecast_start_date": None,
            "forecast_end_date": None,
            "forecast_days": 0,
            "predictions": None,
            "expected_changes": None,
            "uncertainty": None,
        }

    # Price predictions
    first_pred = float(future_predictions["predicted_price"].iloc[0])
    last_pred = float(future_predictions["predicted_price"].iloc[-1])
    max_pred = float(future_predictions["predicted_price"].max())
    min_pred = float(future_predictions["predicted_price"].min())

    # Calculate expected changes
    change_7d = None
    change_30d = None
    change_end = float((last_pred - last_price) / last_price * 100)

    if len(future_predictions) >= 7:
        pred_7d = float(future_predictions["predicted_price"].iloc[6])
        change_7d = float((pred_7d - last_price) / last_price * 100)

    if len(future_predictions) >= 30:
        pred_30d = float(future_predictions["predicted_price"].iloc[29])
        change_30d = float((pred_30d - last_price) / last_price * 100)

    summary = {
        "last_historical_price": last_price,
        "last_historical_date": last_date,
        "forecast_start_date": str(future_predictions["ds"].iloc[0].date()),
        "forecast_end_date": str(future_predictions["ds"].iloc[-1].date()),
        "forecast_days": len(future_predictions),
        "predictions": {
            "first_day": first_pred,
            "last_day": last_pred,
            "max_predicted": max_pred,
            "min_predicted": min_pred,
        },
        "expected_changes": {
            "7_day_change_pct": change_7d,
            "30_day_change_pct": change_30d,
            "end_of_forecast_change_pct": change_end,
        },
        "uncertainty": {
            "avg_interval_width": float(
                (
                    future_predictions["predicted_price_upper"]
                    - future_predictions["predicted_price_lower"]
                ).mean()
            ),
        },
    }

    logger.info("Forecast summary created:")
    logger.info(f"  Last price: ${last_price:,.2f}")
    logger.info(f"  Forecast days: {len(future_predictions)}")
    logger.info(f"  End prediction: ${last_pred:,.2f} ({change_end:+.2f}%)")

    return summary
