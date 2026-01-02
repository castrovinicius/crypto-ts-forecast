"""Inference pipeline definition."""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    create_forecast_summary,
    create_future_dataframe,
    extract_future_predictions,
    generate_forecast,
)


def create_pipeline(**kwargs) -> Pipeline:
    """Create the inference pipeline.

    This pipeline generates future predictions using the trained model.

    Returns:
        A Kedro Pipeline object.
    """
    return pipeline(
        [
            node(
                func=create_future_dataframe,
                inputs={
                    "model": "prophet_model",
                    "prophet_data": "prophet_full_dataset",
                    "forecast_days": "params:forecast.days_ahead",
                    "add_volume_regressor": "params:prophet.add_volume_regressor",
                },
                outputs="future_dataframe",
                name="create_future_dataframe_node",
                tags=["inference"],
            ),
            node(
                func=generate_forecast,
                inputs={
                    "model": "prophet_model",
                    "future_df": "future_dataframe",
                },
                outputs="full_forecast",
                name="generate_forecast_node",
                tags=["inference"],
            ),
            node(
                func=extract_future_predictions,
                inputs={
                    "forecast": "full_forecast",
                    "prophet_data": "prophet_full_dataset",
                },
                outputs="future_predictions",
                name="extract_future_predictions_node",
                tags=["inference"],
            ),
            node(
                func=create_forecast_summary,
                inputs={
                    "future_predictions": "future_predictions",
                    "prophet_data": "prophet_full_dataset",
                },
                outputs="forecast_summary",
                name="create_forecast_summary_node",
                tags=["inference", "reporting"],
            ),
        ],
    )
