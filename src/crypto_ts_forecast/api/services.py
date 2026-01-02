"""Service layer for interacting with Kedro and model artifacts."""

import json
import logging
import pickle
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from kedro.framework.project import pipelines
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project
from prophet import Prophet

logger = logging.getLogger(__name__)

# Binance API base URL
BINANCE_BASE_URL = "https://api.binance.com"


class ForecastService:
    """Service for managing forecasting operations."""

    def __init__(self, project_path: str | Path):
        """Initialize the forecast service.

        Args:
            project_path: Path to the Kedro project root.
        """
        self.project_path = Path(project_path)
        self._bootstrap_project()

    def _bootstrap_project(self) -> None:
        """Bootstrap the Kedro project."""
        bootstrap_project(self.project_path)
        logger.info(f"Kedro project bootstrapped at {self.project_path}")

    def run_pipeline(
        self,
        pipeline_name: str = "__default__",
    ) -> dict[str, Any]:
        """Run a Kedro pipeline.

        Args:
            pipeline_name: Name of the pipeline to run.

        Returns:
            Dictionary with execution results.
        """
        start_time = time.time()

        try:
            with KedroSession.create(
                project_path=self.project_path,
            ) as session:
                session.run(pipeline_name=pipeline_name)

            duration = time.time() - start_time

            return {
                "status": "success",
                "pipeline_name": pipeline_name,
                "message": f"Pipeline '{pipeline_name}' completed successfully",
                "duration_seconds": round(duration, 2),
            }

        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Pipeline execution failed: {e}")
            return {
                "status": "error",
                "pipeline_name": pipeline_name,
                "message": f"Pipeline execution failed: {str(e)}",
                "duration_seconds": round(duration, 2),
            }

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the trained model.

        Returns:
            Dictionary with model information.
        """
        model_path = self.project_path / "data/06_models/prophet_model.pkl"
        report_path = self.project_path / "data/08_reporting/model_report.json"

        if not model_path.exists():
            return {
                "model_exists": False,
                "model_type": None,
                "training_date": None,
                "metrics": None,
                "training_info": None,
            }

        result: dict[str, Any] = {
            "model_exists": True,
            "model_type": "Prophet",
            "training_date": datetime.fromtimestamp(
                model_path.stat().st_mtime
            ).isoformat(),
        }

        if report_path.exists():
            with open(report_path) as f:
                report = json.load(f)
                result["metrics"] = report.get("metrics")
                result["training_info"] = report.get("training_info")

        return result

    def load_model(self) -> Prophet | None:
        """Load the trained Prophet model.

        Returns:
            Prophet model or None if not found.
        """
        model_path = self.project_path / "data/06_models/prophet_model.pkl"

        if not model_path.exists():
            return None

        with open(model_path, "rb") as f:
            return pickle.load(f)

    def get_forecast(
        self,
        days_ahead: int = 30,
        retrain: bool = False,
    ) -> dict[str, Any]:
        """Generate forecast for the specified number of days.

        Args:
            days_ahead: Number of days to forecast.
            retrain: Whether to retrain the model first.

        Returns:
            Dictionary with forecast data.
        """
        # Retrain if requested or if model doesn't exist
        model = self.load_model()
        if retrain or model is None:
            logger.info("Training model before forecasting...")
            result = self.run_pipeline("__default__")
            if result["status"] == "error":
                return result
            model = self.load_model()

        if model is None:
            return {
                "status": "error",
                "message": "No trained model available. Run the training pipeline first.",
            }

        # Load historical data
        data_path = self.project_path / "data/04_feature/prophet_full.parquet"
        if not data_path.exists():
            return {
                "status": "error",
                "message": "No historical data available. Run the pipeline first.",
            }

        prophet_data = pd.read_parquet(data_path)

        # Create future dataframe
        future = model.make_future_dataframe(periods=days_ahead, freq="D")

        # Generate forecast
        forecast = model.predict(future)

        # Filter to future only
        last_date = prophet_data["ds"].max()
        future_forecast = forecast[forecast["ds"] > last_date].copy()

        # Prepare predictions list
        predictions = []
        for _, row in future_forecast.iterrows():
            predictions.append(
                {
                    "date": row["ds"].date().isoformat(),
                    "predicted_price": round(row["yhat"], 2),
                    "predicted_price_lower": round(row["yhat_lower"], 2),
                    "predicted_price_upper": round(row["yhat_upper"], 2),
                    "trend": round(row["trend"], 2),
                }
            )

        # Load summary if available
        summary_path = self.project_path / "data/08_reporting/forecast_summary.json"
        summary = {}
        if summary_path.exists():
            with open(summary_path) as f:
                summary = json.load(f)

        # Get parameters
        symbol = "BTCUSDT"  # Default

        return {
            "status": "success",
            "symbol": symbol,
            "last_historical_date": last_date.date().isoformat(),
            "last_historical_price": round(float(prophet_data["y"].iloc[-1]), 2),
            "forecast_days": len(predictions),
            "predictions": predictions,
            "summary": summary,
        }

    def get_predictions_dataframe(self) -> pd.DataFrame | None:
        """Get the stored predictions as a DataFrame.

        Returns:
            DataFrame with predictions or None.
        """
        predictions_path = (
            self.project_path / "data/07_model_output/future_predictions.parquet"
        )

        if not predictions_path.exists():
            return None

        return pd.read_parquet(predictions_path)

    @staticmethod
    def get_current_price(symbol: str = "BTCUSDT") -> dict[str, Any]:
        """Get current price from Binance API.

        Args:
            symbol: Trading pair symbol.

        Returns:
            Dictionary with current price information.
        """
        endpoint = f"{BINANCE_BASE_URL}/api/v3/ticker/price"

        try:
            response = requests.get(endpoint, params={"symbol": symbol}, timeout=10)
            response.raise_for_status()
            data = response.json()

            return {
                "symbol": data["symbol"],
                "price": float(data["price"]),
                "timestamp": datetime.now().isoformat(),
            }

        except requests.RequestException as e:
            logger.error(f"Error fetching current price: {e}")
            return {
                "symbol": symbol,
                "price": None,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    def get_available_pipelines(self) -> list[str]:
        """Get list of available pipelines.

        Returns:
            List of pipeline names.
        """
        return list(pipelines.keys())
