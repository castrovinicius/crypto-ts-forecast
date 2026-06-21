"""
This module contains unit tests for the data processing pipeline nodes.
"""

import numpy as np
import pandas as pd
import pytest

from crypto_ts_forecast.pipelines.data_processing.nodes import (
    add_features,
    create_prophet_dataset,
    split_train_test,
)


@pytest.fixture
def sample_validated_data():
    dates = pd.date_range(start="2021-01-01", periods=10, freq="D")
    return pd.DataFrame(
        {
            "timestamp": dates,
            "open": np.random.rand(10) * 1000,
            "high": np.random.rand(10) * 1000,
            "low": np.random.rand(10) * 1000,
            "close": np.random.rand(10) * 1000,
            "volume": np.random.rand(10) * 100,
        }
    )


class TestDataProcessingNodes:
    def test_create_prophet_dataset(self, sample_validated_data):
        prophet_df = create_prophet_dataset(sample_validated_data, price_column="close")

        assert isinstance(prophet_df, pd.DataFrame)
        # ds, y plus a lag-1 regressor for every OHLCV column except the target
        assert list(prophet_df.columns) == [
            "ds",
            "y",
            "open_lag1",
            "high_lag1",
            "low_lag1",
            "volume_lag1",
        ]
        # The lag shift drops the first row (10 -> 9)
        assert len(prophet_df) == 9
        assert pd.api.types.is_datetime64_any_dtype(prophet_df["ds"])
        # Timezone must be stripped (tz-naive) for Prophet
        assert prophet_df["ds"].dt.tz is None

    def test_create_prophet_dataset_no_regressors(self, sample_validated_data):
        prophet_df = create_prophet_dataset(
            sample_validated_data, price_column="close", add_regressors=False
        )

        # Without regressors only the base Prophet columns remain
        assert list(prophet_df.columns) == ["ds", "y"]
        # No lag shift means no rows are dropped
        assert len(prophet_df) == 10

    def test_create_prophet_dataset_custom_regressors(self, sample_validated_data):
        prophet_df = create_prophet_dataset(
            sample_validated_data,
            price_column="close",
            regressor_columns=["volume"],
            lag_days=2,
        )

        assert list(prophet_df.columns) == ["ds", "y", "volume_lag2"]
        # lag_days=2 drops the first two rows (10 -> 8)
        assert len(prophet_df) == 8

    def test_add_features_with_moving_averages(self):
        prophet_df = pd.DataFrame(
            {
                "ds": pd.date_range(start="2021-01-01", periods=10, freq="D"),
                "y": range(10),
                "volume_lag1": range(10, 20),
            }
        )

        enhanced_df = add_features(prophet_df, add_moving_averages=True, ma_window=3)

        # A moving-average column is added for the regressor only (not ds/y)
        assert "volume_lag1_ma3" in enhanced_df.columns
        assert "y_ma3" not in enhanced_df.columns
        assert len(enhanced_df) == 10
        # With min_periods=1 the first MA value equals the first raw value
        assert enhanced_df["volume_lag1_ma3"].iloc[0] == 10

    def test_add_features_disabled(self):
        prophet_df = pd.DataFrame(
            {
                "ds": pd.date_range(start="2021-01-01", periods=5, freq="D"),
                "y": range(5),
                "volume_lag1": range(5),
            }
        )

        enhanced_df = add_features(prophet_df, add_moving_averages=False)

        # Nothing is added when moving averages are disabled
        assert list(enhanced_df.columns) == ["ds", "y", "volume_lag1"]

    def test_split_train_test(self):
        dates = pd.date_range(start="2021-01-01", periods=10, freq="D")
        prophet_df = pd.DataFrame({"ds": dates, "y": range(10)})

        # Split last 3 days as test
        train, test = split_train_test(prophet_df, test_size_days=3)

        assert len(train) == 7
        assert len(test) == 3
        assert train["ds"].max() < test["ds"].min()
        assert len(train) + len(test) == 10
