"""Data processing pipeline definition."""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    add_features,
    create_prophet_dataset,
    fetch_fear_greed_data,
    merge_fear_greed_data,
    split_train_test,
)


def create_pipeline(**kwargs) -> Pipeline:
    """Create the data processing pipeline.

    This pipeline transforms validated data into Prophet-ready format.

    Returns:
        A Kedro Pipeline object.
    """
    return pipeline(
        [
            node(
                func=fetch_fear_greed_data,
                inputs={},
                outputs="fear_greed_raw_data",
                name="fetch_fear_greed_node",
                tags=["processing", "external"],
            ),
            node(
                func=merge_fear_greed_data,
                inputs={
                    "validated_data": "validated_bitcoin_data",
                    "fear_greed_data": "fear_greed_raw_data",
                },
                outputs="enriched_bitcoin_data",
                name="merge_fear_greed_node",
                tags=["processing", "external"],
            ),
            node(
                func=create_prophet_dataset,
                inputs={
                    "validated_data": "enriched_bitcoin_data",
                    "price_column": "params:prophet.price_column",
                    "add_regressors": "params:prophet.add_regressors",
                    "lag_days": "params:prophet.lag_days",
                    "regressor_columns": "params:prophet.regressor_columns",
                },
                outputs="prophet_base_dataset",
                name="create_prophet_dataset_node",
                tags=["processing"],
            ),
            node(
                func=add_features,
                inputs={
                    "prophet_df": "prophet_base_dataset",
                    "add_moving_averages": "params:prophet.add_moving_averages",
                    "ma_window": "params:prophet.ma_window",
                },
                outputs="prophet_full_dataset",
                name="add_features_node",
                tags=["processing"],
            ),
            node(
                func=split_train_test,
                inputs={
                    "prophet_df": "prophet_full_dataset",
                    "test_size_days": "params:prophet.test_size_days",
                },
                outputs=["train_dataset", "test_dataset"],
                name="split_train_test_node",
                tags=["processing"],
            ),
        ],
    )
