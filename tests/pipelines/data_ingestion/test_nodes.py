"""
This module contains unit tests for the data ingestion pipeline nodes.
"""

from unittest.mock import Mock, patch

import pandas as pd
import pytest

from crypto_ts_forecast.pipelines.data_ingestion.nodes import (
    fetch_bitcoin_klines,
    validate_raw_data,
)


@pytest.fixture
def sample_raw_data():
    return pd.DataFrame(
        {
            "timestamp": [1609459200000, 1609545600000],
            "open": [29000.0, 29500.0],
            "high": [29500.0, 30000.0],
            "low": [28500.0, 29000.0],
            "close": [29300.0, 29800.0],
            "volume": [1000.0, 1200.0],
            "close_time": [1609545599999, 1609631999999],
            "quote_volume": [29000000.0, 35400000.0],
            "trades": [10000, 12000],
            "taker_buy_base": [500.0, 600.0],
            "taker_buy_quote": [14500000.0, 17700000.0],
            "ignore": [0, 0],
        }
    )


class TestDataIngestionNodes:
    @patch("crypto_ts_forecast.pipelines.data_ingestion.nodes.requests.get")
    def test_fetch_bitcoin_klines(self, mock_get):
        # Mock response data
        mock_response = Mock()
        mock_response.json.side_effect = [
            [
                [
                    1609459200000,
                    "29000.0",
                    "29500.0",
                    "28500.0",
                    "29300.0",
                    "1000.0",
                    1609545599999,
                    "29000000.0",
                    10000,
                    "500.0",
                    "14500000.0",
                    "0",
                ],
            ],
            [],  # Second call returns empty list to stop loop
        ]
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        df = fetch_bitcoin_klines(symbol="BTCUSDT", interval="1d", years_of_data=1)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert "timestamp" in df.columns
        assert "close" in df.columns
        # Check type conversion happened
        assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])
        assert pd.api.types.is_float_dtype(df["close"])

    def test_validate_raw_data_success(self, sample_raw_data):
        # Pre-process sample data to match what validate_raw_data expects (it expects raw output from fetch but fetch does some processing)
        # Actually fetch_bitcoin_klines returns processed DF, but validate_raw_data takes that output.
        # Let's look at pipeline.py: fetch -> raw_bitcoin_data -> validate -> validated_bitcoin_data
        # fetch_bitcoin_klines returns a DF with types already converted and 'ignore' dropped.

        # Wait, looking at nodes.py again:
        # fetch_bitcoin_klines does:
        # 1. requests
        # 2. pd.DataFrame(all_klines)
        # 3. pd.to_datetime, pd.to_numeric
        # 4. drop ignore
        # 5. drop_duplicates
        # 6. return df

        # So validate_raw_data receives the CLEANED dataframe from fetch_bitcoin_klines.

        # Let's create a valid dataframe matching that output
        df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime([1609459200000], unit="ms"),
                "open": [29000.0],
                "high": [29500.0],
                "low": [28500.0],
                "close": [29300.0],
                "volume": [1000.0],
                "close_time": pd.to_datetime([1609545599999], unit="ms"),
                "quote_volume": [29000000.0],
                "trades": [10000],
                "taker_buy_base": [500.0],
                "taker_buy_quote": [14500000.0],
            }
        )

        validated_df = validate_raw_data(df)
        assert validated_df.equals(df)

    def test_validate_raw_data_empty(self):
        with pytest.raises(ValueError, match="Raw data is empty"):
            validate_raw_data(pd.DataFrame())

    def test_validate_raw_data_missing_columns(self):
        df = pd.DataFrame({"timestamp": [1609459200000], "open": [29000.0]})
        with pytest.raises(ValueError, match="Missing required columns"):
            validate_raw_data(df)

    def test_validate_raw_data_negative_prices(self):
        df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime([1609459200000], unit="ms"),
                "open": [-100.0],  # Negative price
                "high": [29500.0],
                "low": [28500.0],
                "close": [29300.0],
                "volume": [1000.0],
            }
        )
        with pytest.raises(ValueError, match="Negative prices found"):
            validate_raw_data(df)
