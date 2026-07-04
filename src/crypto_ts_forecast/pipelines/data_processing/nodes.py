"""Nodes for data processing pipeline.

This module contains functions to transform raw data into Prophet format.
"""

import logging

import pandas as pd
import requests

logger = logging.getLogger(__name__)

FEAR_GREED_API_URL = "https://api.alternative.me/fng/?limit=0&format=json&date_format=us"


def fetch_fear_greed_data() -> pd.DataFrame:
    """Fetch the full historical Fear & Greed Index from Alternative.me.

    The index ranges from 0 (Extreme Fear) to 100 (Extreme Greed) and is
    published daily. Data is available from 2018-02-01 onward.

    Returns:
        DataFrame with columns ['timestamp', 'fear_greed_index'].
    """
    logger.info("Fetching Fear & Greed Index from Alternative.me...")

    response = requests.get(FEAR_GREED_API_URL, timeout=30)
    response.raise_for_status()

    records = response.json()["data"]
    df = pd.DataFrame(records)[["timestamp", "value"]].rename(
        columns={"value": "fear_greed_index"}
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["fear_greed_index"] = df["fear_greed_index"].astype(float)
    df = df.sort_values("timestamp").reset_index(drop=True)

    logger.info(
        f"Fetched {len(df)} Fear & Greed records "
        f"({df['timestamp'].min().date()} → {df['timestamp'].max().date()})"
    )
    return df


def merge_fear_greed_data(
    validated_data: pd.DataFrame,
    fear_greed_data: pd.DataFrame,
) -> pd.DataFrame:
    """Merge Fear & Greed Index into the validated OHLCV dataset.

    Joins on date (day-level). Rows before 2018-02-01 (when the index started)
    are filled with 50 (neutral) to avoid dropping historical BTC data.

    Args:
        validated_data: Validated OHLCV data with a 'timestamp' column.
        fear_greed_data: Fear & Greed data from fetch_fear_greed_data.

    Returns:
        validated_data with an additional 'fear_greed_index' column.
    """
    logger.info("Merging Fear & Greed Index into validated data...")

    validated = validated_data.copy()
    fear_greed = fear_greed_data.copy()

    # Normalise both sides to date-only for the join key
    validated["_date"] = pd.to_datetime(validated["timestamp"]).dt.normalize()
    fear_greed["_date"] = pd.to_datetime(fear_greed["timestamp"]).dt.normalize()

    merged = validated.merge(
        fear_greed[["_date", "fear_greed_index"]],
        on="_date",
        how="left",
    ).drop(columns=["_date"])

    # Fill missing values with neutral (50) — covers pre-2018 BTC history
    missing = merged["fear_greed_index"].isna().sum()
    if missing:
        logger.warning(
            f"{missing} rows without Fear & Greed data — filling with neutral value (50)"
        )
        merged["fear_greed_index"] = merged["fear_greed_index"].fillna(50.0)

    logger.info("Fear & Greed Index merged successfully")
    return merged


def create_prophet_dataset(
    validated_data: pd.DataFrame,
    price_column: str,
    add_regressors: bool = True,
    lag_days: int = 1,
    regressor_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Transform validated data into Prophet-ready format.

    Prophet requires a DataFrame with columns 'ds' (datetime) and 'y' (value).
    Optionally adds lagged OHLCV features as regressors. Using lag avoids the
    circular dependency of predicting close(t) with same-day OHLCV(t).

    Args:
        validated_data: Validated Bitcoin data with timestamp and price columns.
        price_column: Name of the price column to use for forecasting.
        add_regressors: Whether to add lagged regressor columns.
        lag_days: Number of days to lag the regressor columns (default: 1).
        regressor_columns: Columns to use as regressors. Defaults to all OHLCV
            columns except price_column. Configure via params:prophet.regressor_columns.

    Returns:
        DataFrame with 'ds', 'y' and optional lagged regressor columns ready for Prophet.
    """
    logger.info(f"Creating Prophet dataset using '{price_column}' as target variable")

    validated_data = validated_data.sort_values("timestamp").reset_index(drop=True)

    # Base columns
    prophet_df = pd.DataFrame(
        {
            "ds": validated_data["timestamp"],
            "y": validated_data[price_column],
        }
    )

    # Add lagged regressors (exclude price_column since it equals y)
    if add_regressors:
        candidate_columns = regressor_columns if regressor_columns else ["open", "high", "low", "close", "volume"]
        available_regressors = [
            col for col in candidate_columns
            if col in validated_data.columns and col != price_column
        ]

        for col in available_regressors:
            prophet_df[f"{col}_lag{lag_days}"] = validated_data[col].shift(lag_days).values

        # Drop first lag_days rows which have NaN in all lag columns
        prophet_df = prophet_df.iloc[lag_days:].reset_index(drop=True)

        lag_names = [f"{c}_lag{lag_days}" for c in available_regressors]
        logger.info(f"Added lag-{lag_days} regressors: {lag_names}")

    # Ensure ds is datetime
    prophet_df["ds"] = pd.to_datetime(prophet_df["ds"])

    # Remove timezone info if present (Prophet doesn't handle it well)
    if prophet_df["ds"].dt.tz is not None:
        prophet_df["ds"] = prophet_df["ds"].dt.tz_localize(None)

    # Sort by date
    prophet_df = prophet_df.sort_values("ds").reset_index(drop=True)

    # Remove any duplicates
    prophet_df = prophet_df.drop_duplicates(subset=["ds"])

    logger.info(
        f"Prophet dataset created with {len(prophet_df)} records "
        f"from {prophet_df['ds'].min()} to {prophet_df['ds'].max()}"
    )

    return prophet_df


def add_features(
    prophet_df: pd.DataFrame,
    add_moving_averages: bool = True,
    ma_window: int = 21,
) -> pd.DataFrame:
    """Add additional features/regressors to the Prophet dataset.

    This function adds moving averages for each regressor column.
    Moving averages will be used to fill future values of regressors during prediction.

    Args:
        prophet_df: Base Prophet dataset with ds, y and regressor columns.
        add_moving_averages: Whether to add moving averages for regressors.
        ma_window: Window size for moving averages (default: 21 days).

    Returns:
        Enhanced Prophet dataset with moving averages.
    """
    if add_moving_averages:
        # Identify regressor columns (everything except ds and y)
        regressor_columns = [col for col in prophet_df.columns if col not in ["ds", "y"]]

        if regressor_columns:
            logger.info(f"Calculating {ma_window}-day moving averages for regressors: {regressor_columns}")

            for col in regressor_columns:
                ma_col_name = f"{col}_ma{ma_window}"
                prophet_df[ma_col_name] = (
                    prophet_df[col]
                    .rolling(window=ma_window, min_periods=1)
                    .mean()
                )

            logger.info(f"Added {len(regressor_columns)} moving average features")
        else:
            logger.info("No regressors found in dataset, skipping moving averages")

    return prophet_df


def split_train_test(
    prophet_df: pd.DataFrame,
    test_size_days: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the data into training and test sets.

    Args:
        prophet_df: Full Prophet dataset.
        test_size_days: Number of days to use for testing.

    Returns:
        Tuple of (train_df, test_df).
    """
    # Sort by date to ensure proper split
    prophet_df = prophet_df.sort_values("ds").reset_index(drop=True)

    # Calculate split point
    split_date = prophet_df["ds"].max() - pd.Timedelta(days=test_size_days)

    train_df = prophet_df[prophet_df["ds"] <= split_date].copy()
    test_df = prophet_df[prophet_df["ds"] > split_date].copy()

    logger.info(
        f"Data split: {len(train_df)} training samples, {len(test_df)} test samples"
    )
    logger.info(f"Training period: {train_df['ds'].min()} to {train_df['ds'].max()}")
    logger.info(f"Test period: {test_df['ds'].min()} to {test_df['ds'].max()}")

    return train_df, test_df
