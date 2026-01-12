"""
This module contains tests for the data processing pipeline.
"""

from kedro.pipeline import Pipeline

from crypto_ts_forecast.pipelines.data_processing.pipeline import create_pipeline


def test_data_processing_pipeline():
    pipeline = create_pipeline()
    assert isinstance(pipeline, Pipeline)
    assert len(pipeline.nodes) == 3

    node_names = [node.name for node in pipeline.nodes]
    assert "create_prophet_dataset_node" in node_names
    assert "add_features_node" in node_names
    assert "split_train_test_node" in node_names
