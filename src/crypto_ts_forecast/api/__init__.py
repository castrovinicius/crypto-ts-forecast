"""FastAPI application for Bitcoin price forecasting.

This module provides REST API endpoints for:
- Running Kedro pipelines
- Getting forecasts
- Checking model status
"""

from .app import create_app

__all__ = ["create_app"]
