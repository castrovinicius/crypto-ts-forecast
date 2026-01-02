"""Nodes for data processing pipeline.

This module contains functions to transform raw data into Prophet format.
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def create_prophet_dataset(
    validated_data: pd.DataFrame,
    price_column: str,
) -> pd.DataFrame:
    """Transform validated data into Prophet-ready format.

    Prophet requires a DataFrame with columns 'ds' (datetime) and 'y' (value).

    Args:
        validated_data: Validated Bitcoin data with timestamp and price columns.
        price_column: Name of the price column to use for forecasting.

    Returns:
        DataFrame with 'ds' and 'y' columns ready for Prophet.
    """
    logger.info(f"Creating Prophet dataset using '{price_column}' as target variable")

    prophet_df = pd.DataFrame(
        {
            "ds": validated_data["timestamp"],
            "y": validated_data[price_column],
        }
    )

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
    add_volume: bool,
    validated_data: pd.DataFrame,
) -> pd.DataFrame:
    """Add additional features/regressors to the Prophet dataset.

    This function can add extra features that Prophet can use as regressors.

    Args:
        prophet_df: Base Prophet dataset with ds and y columns.
        add_volume: Whether to add volume as a regressor.
        validated_data: Original validated data with additional columns.

    Returns:
        Enhanced Prophet dataset with additional features.
    """
    # Merge with original data to get additional columns
    if add_volume and "volume" in validated_data.columns:
        volume_data = validated_data[["timestamp", "volume"]].copy()
        volume_data = volume_data.rename(columns={"timestamp": "ds"})
        volume_data["ds"] = pd.to_datetime(volume_data["ds"])

        if volume_data["ds"].dt.tz is not None:
            volume_data["ds"] = volume_data["ds"].dt.tz_localize(None)

        prophet_df = prophet_df.merge(volume_data, on="ds", how="left")

        # Log transform volume for better scaling
        prophet_df["volume"] = prophet_df["volume"].apply(
            lambda x: x if x > 0 else 1
        )  # Avoid log(0)

        logger.info("Added volume as additional feature")

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
