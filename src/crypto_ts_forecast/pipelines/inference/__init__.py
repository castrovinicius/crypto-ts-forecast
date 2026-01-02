"""Inference pipeline.

This pipeline generates future predictions using the trained Prophet model.
"""

from .pipeline import create_pipeline

__all__ = ["create_pipeline"]
