"""Data processing pipeline definition."""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import add_features, create_prophet_dataset, split_train_test


def create_pipeline(**kwargs) -> Pipeline:
    """Create the data processing pipeline.

    This pipeline transforms validated data into Prophet-ready format.

    Returns:
        A Kedro Pipeline object.
    """
    return pipeline(
        [
            node(
                func=create_prophet_dataset,
                inputs={
                    "validated_data": "validated_bitcoin_data",
                    "price_column": "params:prophet.price_column",
                },
                outputs="prophet_base_dataset",
                name="create_prophet_dataset_node",
                tags=["processing"],
            ),
            node(
                func=add_features,
                inputs={
                    "prophet_df": "prophet_base_dataset",
                    "add_volume": "params:prophet.add_volume_regressor",
                    "validated_data": "validated_bitcoin_data",
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
