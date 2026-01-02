"""FastAPI application factory and routes."""

import logging
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from crypto_ts_forecast import __version__

from .schemas import (
    CurrentPriceResponse,
    ErrorResponse,
    ForecastRequest,
    ForecastResponse,
    HealthResponse,
    ModelInfoResponse,
    PipelineRunRequest,
    PipelineRunResponse,
    PredictionPoint,
)
from .services import ForecastService

logger = logging.getLogger(__name__)

# Global service instance
_forecast_service: ForecastService | None = None


def get_forecast_service() -> ForecastService:
    """Get the forecast service instance."""
    if _forecast_service is None:
        raise HTTPException(
            status_code=500,
            detail="Forecast service not initialized",
        )
    return _forecast_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    global _forecast_service

    # Get project path from app state or use default
    project_path = getattr(app.state, "project_path", Path.cwd())
    _forecast_service = ForecastService(project_path)
    logger.info("Forecast service initialized")

    yield

    # Cleanup
    _forecast_service = None
    logger.info("Forecast service cleaned up")


def create_app(project_path: str | Path | None = None) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        project_path: Path to the Kedro project root.

    Returns:
        Configured FastAPI application.
    """
    app = FastAPI(
        title="Bitcoin Price Forecast API",
        description="""
        REST API for Bitcoin price forecasting using Prophet and Kedro.
        
        ## Features
        
        - **Forecast**: Get Bitcoin price predictions for up to 365 days
        - **Pipeline Management**: Run Kedro pipelines for data ingestion and model training
        - **Model Info**: Check trained model status and metrics
        - **Current Price**: Get real-time Bitcoin price from Binance
        
        ## Pipelines
        
        - `data_ingestion`: Fetch data from Binance API
        - `data_processing`: Transform data for Prophet
        - `model_training`: Train and evaluate Prophet model
        - `inference`: Generate forecasts
        - `__default__`: Run all pipelines
        """,
        version=__version__,
        lifespan=lifespan,
        responses={
            500: {"model": ErrorResponse, "description": "Internal Server Error"},
        },
    )

    # Store project path in app state
    if project_path:
        app.state.project_path = Path(project_path)
    else:
        app.state.project_path = Path.cwd()

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register routes
    register_routes(app)

    return app


def register_routes(app: FastAPI) -> None:
    """Register all API routes."""

    @app.get(
        "/",
        response_model=HealthResponse,
        tags=["Health"],
        summary="Health check",
    )
    async def health_check() -> HealthResponse:
        """Check API health status."""
        return HealthResponse(
            status="healthy",
            version=__version__,
            timestamp=datetime.now(),
        )

    @app.get(
        "/health",
        response_model=HealthResponse,
        tags=["Health"],
        summary="Health check",
    )
    async def health() -> HealthResponse:
        """Check API health status."""
        return HealthResponse(
            status="healthy",
            version=__version__,
            timestamp=datetime.now(),
        )

    @app.get(
        "/api/v1/price/current",
        response_model=CurrentPriceResponse,
        tags=["Price"],
        summary="Get current Bitcoin price",
    )
    async def get_current_price(
        symbol: str = Query(default="BTCUSDT", description="Trading pair symbol"),
    ) -> CurrentPriceResponse:
        """Get the current Bitcoin price from Binance API."""
        result = ForecastService.get_current_price(symbol)

        if "error" in result:
            raise HTTPException(
                status_code=502,
                detail=f"Failed to fetch price from Binance: {result['error']}",
            )

        return CurrentPriceResponse(
            symbol=result["symbol"],
            price=result["price"],
            timestamp=datetime.fromisoformat(result["timestamp"]),
        )

    @app.get(
        "/api/v1/model/info",
        response_model=ModelInfoResponse,
        tags=["Model"],
        summary="Get model information",
    )
    async def get_model_info() -> ModelInfoResponse:
        """Get information about the trained model."""
        service = get_forecast_service()
        info = service.get_model_info()
        return ModelInfoResponse(**info)

    @app.get(
        "/api/v1/pipelines",
        tags=["Pipelines"],
        summary="List available pipelines",
    )
    async def list_pipelines() -> dict[str, list[str]]:
        """Get list of available Kedro pipelines."""
        service = get_forecast_service()
        return {"pipelines": service.get_available_pipelines()}

    @app.post(
        "/api/v1/pipelines/run",
        response_model=PipelineRunResponse,
        tags=["Pipelines"],
        summary="Run a pipeline",
    )
    async def run_pipeline(request: PipelineRunRequest) -> PipelineRunResponse:
        """Run a Kedro pipeline.

        Available pipelines:
        - `data_ingestion`: Fetch data from Binance
        - `data_processing`: Transform data for Prophet
        - `model_training`: Train the Prophet model
        - `inference`: Generate forecasts
        - `__default__`: Run all pipelines
        """
        service = get_forecast_service()
        result = service.run_pipeline(
            pipeline_name=request.pipeline_name,
        )

        if result["status"] == "error":
            raise HTTPException(
                status_code=500,
                detail=result["message"],
            )

        return PipelineRunResponse(**result)

    @app.post(
        "/api/v1/forecast",
        response_model=ForecastResponse,
        tags=["Forecast"],
        summary="Generate forecast",
    )
    async def generate_forecast(request: ForecastRequest) -> ForecastResponse:
        """Generate Bitcoin price forecast.

        This endpoint will:
        1. Optionally retrain the model if `retrain=true`
        2. Generate predictions for the specified number of days
        3. Return predictions with uncertainty intervals
        """
        service = get_forecast_service()
        result = service.get_forecast(
            days_ahead=request.days_ahead,
            retrain=request.retrain,
        )

        if result["status"] == "error":
            raise HTTPException(
                status_code=500,
                detail=result.get("message", "Forecast generation failed"),
            )

        # Convert predictions to PredictionPoint objects
        predictions = [
            PredictionPoint(
                prediction_date=p["date"],
                predicted_price=p["predicted_price"],
                predicted_price_lower=p["predicted_price_lower"],
                predicted_price_upper=p["predicted_price_upper"],
                trend=p["trend"],
            )
            for p in result["predictions"]
        ]

        return ForecastResponse(
            status="success",
            symbol=result["symbol"],
            last_historical_date=result["last_historical_date"],
            last_historical_price=result["last_historical_price"],
            forecast_days=result["forecast_days"],
            predictions=predictions,
            summary=result["summary"],
        )

    @app.get(
        "/api/v1/forecast",
        response_model=ForecastResponse,
        tags=["Forecast"],
        summary="Get latest forecast",
    )
    async def get_forecast(
        days_ahead: int = Query(
            default=30,
            ge=1,
            le=365,
            description="Number of days to forecast",
        ),
    ) -> ForecastResponse:
        """Get Bitcoin price forecast without retraining.

        If no model exists, this will return an error.
        Use POST /api/v1/forecast with retrain=true to train first.
        """
        service = get_forecast_service()
        result = service.get_forecast(days_ahead=days_ahead, retrain=False)

        if result["status"] == "error":
            raise HTTPException(
                status_code=404,
                detail=result.get("message", "No forecast available"),
            )

        predictions = [
            PredictionPoint(
                prediction_date=p["date"],
                predicted_price=p["predicted_price"],
                predicted_price_lower=p["predicted_price_lower"],
                predicted_price_upper=p["predicted_price_upper"],
                trend=p["trend"],
            )
            for p in result["predictions"]
        ]

        return ForecastResponse(
            status="success",
            symbol=result["symbol"],
            last_historical_date=result["last_historical_date"],
            last_historical_price=result["last_historical_price"],
            forecast_days=result["forecast_days"],
            predictions=predictions,
            summary=result["summary"],
        )

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Any, exc: Exception) -> JSONResponse:
        """Global exception handler."""
        logger.exception(f"Unhandled exception: {exc}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": "Internal server error",
                "detail": str(exc),
            },
        )
