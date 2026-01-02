"""Data ingestion pipeline.

This pipeline fetches Bitcoin historical data from Binance API.
"""

from .pipeline import create_pipeline

__all__ = ["create_pipeline"]
