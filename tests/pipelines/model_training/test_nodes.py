"""
This module contains unit tests for the model training pipeline nodes.
"""

from unittest.mock import Mock, patch

import pandas as pd
import pytest

from crypto_ts_forecast.pipelines.model_training.nodes import (
    create_model_report,
    evaluate_model,
    train_prophet_model,
)


@pytest.fixture
def sample_train_data():
    dates = pd.date_range(start="2021-01-01", periods=30, freq="D")
    return pd.DataFrame({"ds": dates, "y": range(30), "volume": range(30)})


@pytest.fixture
def sample_test_data():
    dates = pd.date_range(start="2021-01-31", periods=10, freq="D")
    return pd.DataFrame({"ds": dates, "y": range(30, 40), "volume": range(30, 40)})


class TestModelTrainingNodes:
    @patch("crypto_ts_forecast.pipelines.model_training.nodes.Prophet")
    def test_train_prophet_model(self, mock_prophet_cls, sample_train_data):
        mock_model = Mock()
        mock_prophet_cls.return_value = mock_model

        model = train_prophet_model(
            train_data=sample_train_data,
            seasonality_mode="additive",
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=True,
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=10.0,
            add_volume_regressor=True,
            changepoint_range=0.8,
        )

        assert model == mock_model
        mock_prophet_cls.assert_called_once()
        mock_model.add_regressor.assert_called_with("volume")
        mock_model.add_seasonality.assert_called()  # Check for halving cycle
        mock_model.fit.assert_called_with(sample_train_data)

    def test_evaluate_model(self, sample_test_data):
        mock_model = Mock()
        # Mock predict return
        future_df = sample_test_data[["ds"]].copy()
        if "volume" in sample_test_data.columns:
            future_df["volume"] = sample_test_data["volume"]

        # Return perfect predictions for easy metric check
        forecast = pd.DataFrame(
            {
                "ds": sample_test_data["ds"],
                "yhat": sample_test_data["y"],  # Perfect prediction
            }
        )
        mock_model.predict.return_value = forecast

        metrics = evaluate_model(
            model=mock_model, test_data=sample_test_data, add_volume_regressor=True
        )

        assert metrics["mae"] == 0.0
        assert metrics["rmse"] == 0.0
        assert metrics["r2"] == 1.0
        assert metrics["test_samples"] == 10

    def test_create_model_report(self, sample_train_data, sample_test_data):
        metrics = {"mae": 10.0, "rmse": 15.0}
        report = create_model_report(metrics, sample_train_data, sample_test_data)

        assert report["model_type"] == "Prophet"
        assert report["metrics"] == metrics
        assert report["training_info"]["samples"] == 30
        assert report["test_info"]["samples"] == 10
