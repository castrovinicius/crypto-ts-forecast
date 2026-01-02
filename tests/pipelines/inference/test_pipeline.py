"""
This module contains tests for the inference pipeline.
"""
from kedro.pipeline import Pipeline
from crypto_ts_forecast.pipelines.inference.pipeline import create_pipeline

def test_inference_pipeline():
    pipeline = create_pipeline()
    assert isinstance(pipeline, Pipeline)
    assert len(pipeline.nodes) == 4
    
    node_names = [node.name for node in pipeline.nodes]
    assert "create_future_dataframe_node" in node_names
    assert "generate_forecast_node" in node_names
    assert "extract_future_predictions_node" in node_names
    assert "create_forecast_summary_node" in node_names
