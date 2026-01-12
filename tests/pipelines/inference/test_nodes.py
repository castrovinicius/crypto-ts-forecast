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
        mock_model = Mock()
        # Mock make_future_dataframe
        future_dates = pd.date_range(
            start="2021-01-01", periods=35, freq="D"
        )  # 30 historical + 5 future
        mock_model.make_future_dataframe.return_value = pd.DataFrame(
            {"ds": future_dates}
        )

        future_df = create_future_dataframe(
            model=mock_model,
            prophet_data=sample_prophet_data,
            forecast_days=5,
            add_volume_regressor=True,
        )

        assert len(future_df) == 35
        assert "volume" in future_df.columns
        # Check if future volumes are filled (should be 100.0 as avg of last 30 is 100)
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
