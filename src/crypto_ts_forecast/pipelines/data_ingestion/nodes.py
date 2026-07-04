"""Nodes for data ingestion pipeline.

This module contains functions to fetch Bitcoin data from Binance API.
"""

import logging
from datetime import datetime, timedelta
from typing import Any

import pandas as pd
import requests

logger = logging.getLogger(__name__)

BINANCE_BASE_URL = "https://api.binance.com"


def fetch_bitcoin_klines(
    symbol: str,
    interval: str,
    years_of_data: int,
) -> pd.DataFrame:
    """Fetch Bitcoin klines (candlestick) data from Binance API.

    The Binance API endpoint GET /api/v3/klines returns OHLCV data.
    We need to paginate because the API has a limit of 1000 records per request.

    Args:
        symbol: Trading pair symbol (e.g., "BTCUSDT").
        interval: Kline interval (e.g., "1d" for daily).
        years_of_data: Number of years of historical data to fetch.

    Returns:
        DataFrame with columns: timestamp, open, high, low, close, volume,
        close_time, quote_volume, trades, taker_buy_base, taker_buy_quote.
    """
    endpoint = f"{BINANCE_BASE_URL}/api/v3/klines"

    # Calculate start time (years_of_data years ago)
    end_time = datetime.now()
    start_time = end_time - timedelta(days=years_of_data * 365)

    # Convert to milliseconds timestamp
    start_ms = int(start_time.timestamp() * 1000)
    end_ms = int(end_time.timestamp() * 1000)

    all_klines: list[list[Any]] = []
    current_start = start_ms

    logger.info(f"Fetching {symbol} data from {start_time.date()} to {end_time.date()}")

    while current_start < end_ms:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": current_start,
            "endTime": end_ms,
            "limit": 1000,  # Maximum allowed by Binance
        }

        try:
            response = requests.get(endpoint, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            if not data:
                break

            all_klines.extend(data)

            # Update start time for next iteration
            # Use the close time of the last kline + 1ms
            last_close_time = data[-1][6]
            current_start = last_close_time + 1

            logger.info(f"Fetched {len(all_klines)} klines so far...")

        except requests.RequestException as e:
            logger.error(f"Error fetching data from Binance: {e}")
            raise

    # Convert to DataFrame
    columns = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_volume",
        "trades",
        "taker_buy_base",
        "taker_buy_quote",
        "ignore",
    ]

    df = pd.DataFrame(all_klines, columns=columns)

    # Convert timestamp to datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")

    # Convert numeric columns
    numeric_cols = ["open", "high", "low", "close", "volume", "quote_volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop the 'ignore' column
    df = df.drop(columns=["ignore"])

    # Remove duplicates if any
    df = df.drop_duplicates(subset=["timestamp"])

    # Sort by timestamp
    df = df.sort_values("timestamp").reset_index(drop=True)

    logger.info(
        f"Successfully fetched {len(df)} records from "
        f"{df['timestamp'].min()} to {df['timestamp'].max()}"
    )

    return df


def validate_raw_data(raw_data: pd.DataFrame) -> pd.DataFrame:
    """Validate the raw data from Binance.

    Performs basic validation checks on the fetched data.

    Args:
        raw_data: Raw kline data from Binance.

    Returns:
        Validated DataFrame.

    Raises:
        ValueError: If validation fails.
    """
    if raw_data.empty:
        raise ValueError("Raw data is empty!")

    required_columns = ["timestamp", "open", "high", "low", "close", "volume"]
    missing_cols = set(required_columns) - set(raw_data.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Check for null values in critical columns
    null_counts = raw_data[required_columns].isnull().sum()
    if null_counts.any():
        logger.warning(f"Null values found:\n{null_counts[null_counts > 0]}")
        # Fill nulls with forward fill
        raw_data[required_columns] = raw_data[required_columns].ffill()

    # Check for negative prices
    price_cols = ["open", "high", "low", "close"]
    for col in price_cols:
        if (raw_data[col] < 0).any():
            raise ValueError(f"Negative prices found in column {col}")

    logger.info("Data validation passed successfully")

    return raw_data
