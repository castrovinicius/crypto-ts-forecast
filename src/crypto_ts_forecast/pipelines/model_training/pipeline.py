"""Model training pipeline definition."""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import create_model_report, evaluate_model, train_prophet_model


def create_pipeline(**kwargs) -> Pipeline:
    """Create the model training pipeline.

    This pipeline trains and evaluates a Prophet model.

    Returns:
        A Kedro Pipeline object.
    """
    return pipeline(
        [
            node(
                func=train_prophet_model,
                inputs={
                    "train_data": "prophet_full_dataset",
                    "seasonality_mode": "params:prophet.seasonality_mode",
                    "yearly_seasonality": "params:prophet.yearly_seasonality",
                    "weekly_seasonality": "params:prophet.weekly_seasonality",
                    "daily_seasonality": "params:prophet.daily_seasonality",
                    "changepoint_prior_scale": "params:prophet.changepoint_prior_scale",
                    "seasonality_prior_scale": "params:prophet.seasonality_prior_scale",
                    "add_volume_regressor": "params:prophet.add_volume_regressor",
                    "changepoint_range": "params:prophet.changepoint_range",
                },
                outputs="prophet_model",
                name="train_prophet_model_node",
                tags=["training"],
            ),
            node(
                func=evaluate_model,
                inputs={
                    "model": "prophet_model",
                    "test_data": "test_dataset",
                    "add_volume_regressor": "params:prophet.add_volume_regressor",
                },
                outputs="model_metrics",
                name="evaluate_model_node",
                tags=["training", "evaluation"],
            ),
            node(
                func=create_model_report,
                inputs={
                    "metrics": "model_metrics",
                    "train_data": "prophet_full_dataset",
                    "test_data": "test_dataset",
                },
                outputs="model_report",
                name="create_model_report_node",
                tags=["training", "reporting"],
            ),
        ],
    )
