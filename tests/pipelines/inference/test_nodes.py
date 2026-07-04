"""
This module contains unit tests for the inference pipeline nodes.
"""

from unittest.mock import Mock

import pandas as pd
import pytest

from crypto_ts_forecast.pipelines.inference.nodes import (
    create_forecast_summary,
    create_future_dataframe,
    extract_future_predictions,
    generate_forecast,
)


@pytest.fixture
def sample_prophet_data():
    dates = pd.date_range(start="2021-01-01", periods=30, freq="D")
    return pd.DataFrame({"ds": dates, "y": range(30), "volume": [100.0] * 30})


class TestInferenceNodes:
    def test_create_future_dataframe(self, sample_prophet_data):
        # The model argument is unused by the node; the historical data and the
        # chosen strategy drive the future regressor values.
        mock_model = Mock()

        future_df = create_future_dataframe(
            model=mock_model,
            prophet_data=sample_prophet_data,
            forecast_days=5,
        )

        # Only the future rows are returned (one per day for daily data)
        assert len(future_df) == 5
        assert "volume" in future_df.columns
        # All future dates fall beyond the last historical date
        assert future_df["ds"].min() > sample_prophet_data["ds"].max()
        # The default 'ma' strategy fills volume with the mean of recent
        # values (all 100.0 in the fixture)
        assert future_df["volume"].iloc[-1] == 100.0

    def test_generate_forecast(self):
        mock_model = Mock()
        future_df = pd.DataFrame(
            {"ds": pd.date_range(start="2021-01-01", periods=5, freq="D")}
        )

        # Mock predict output
        forecast_return = future_df.copy()
        forecast_return["yhat"] = [10, 11, 12, 13, 14]
        forecast_return["yhat_lower"] = [9, 10, 11, 12, 13]
        forecast_return["yhat_upper"] = [11, 12, 13, 14, 15]
        forecast_return["trend"] = [10, 11, 12, 13, 14]
        forecast_return["trend_lower"] = [9, 10, 11, 12, 13]
        forecast_return["trend_upper"] = [11, 12, 13, 14, 15]

        mock_model.predict.return_value = forecast_return

        forecast = generate_forecast(mock_model, future_df)

        assert "predicted_price" in forecast.columns
        assert "predicted_price_lower" in forecast.columns
        assert "predicted_price_upper" in forecast.columns
        assert len(forecast) == 5

    def test_extract_future_predictions(self, sample_prophet_data):
        # Create forecast with historical + future dates
        dates = pd.date_range(start="2021-01-01", periods=35, freq="D")
        forecast = pd.DataFrame({"ds": dates, "predicted_price": range(35)})

        future_preds = extract_future_predictions(forecast, sample_prophet_data)

        # Should have 5 future predictions (35 total - 30 historical)
        assert len(future_preds) == 5
        assert future_preds["ds"].min() > sample_prophet_data["ds"].max()

    def test_create_forecast_summary(self, sample_prophet_data):
        future_dates = pd.date_range(start="2021-01-31", periods=5, freq="D")
        future_predictions = pd.DataFrame(
            {
                "ds": future_dates,
                "predicted_price": [30, 31, 32, 33, 34],
                "predicted_price_lower": [29, 30, 31, 32, 33],
                "predicted_price_upper": [31, 32, 33, 34, 35],
            }
        )

        summary = create_forecast_summary(future_predictions, sample_prophet_data)

        assert summary["last_historical_price"] == 29.0  # Last value in range(30) is 29
        assert summary["forecast_days"] == 5
        assert summary["predictions"]["max_predicted"] == 34.0
