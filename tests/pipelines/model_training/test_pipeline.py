"""
This module contains tests for the model training pipeline.
"""
from kedro.pipeline import Pipeline
from crypto_ts_forecast.pipelines.model_training.pipeline import create_pipeline

def test_model_training_pipeline():
    pipeline = create_pipeline()
    assert isinstance(pipeline, Pipeline)
    assert len(pipeline.nodes) == 3
    
    node_names = [node.name for node in pipeline.nodes]
    assert "train_prophet_model_node" in node_names
    assert "evaluate_model_node" in node_names
    assert "create_model_report_node" in node_names
