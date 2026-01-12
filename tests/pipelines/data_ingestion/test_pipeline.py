"""
This module contains tests for the data ingestion pipeline.
"""

from kedro.pipeline import Pipeline

from crypto_ts_forecast.pipelines.data_ingestion.pipeline import create_pipeline


def test_data_ingestion_pipeline():
    pipeline = create_pipeline()
    assert isinstance(pipeline, Pipeline)
    assert len(pipeline.nodes) == 2

    node_names = [node.name for node in pipeline.nodes]
    assert "fetch_bitcoin_klines_node" in node_names
    assert "validate_raw_data_node" in node_names
