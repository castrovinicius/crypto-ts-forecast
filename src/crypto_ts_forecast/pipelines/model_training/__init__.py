"""Model training pipeline.

This pipeline trains a Prophet model for Bitcoin price forecasting.
"""

from .pipeline import create_pipeline

__all__ = ["create_pipeline"]
