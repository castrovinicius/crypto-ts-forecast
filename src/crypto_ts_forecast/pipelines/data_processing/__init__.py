"""Data processing pipeline.

This pipeline transforms raw Bitcoin data into Prophet-ready format.
"""

from .pipeline import create_pipeline

__all__ = ["create_pipeline"]
