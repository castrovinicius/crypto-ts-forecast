"""Data ingestion pipeline definition."""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import fetch_bitcoin_klines, validate_raw_data


def create_pipeline(**kwargs) -> Pipeline:
    """Create the data ingestion pipeline.

    This pipeline fetches Bitcoin data from Binance and validates it.

    Returns:
        A Kedro Pipeline object.
    """
    return pipeline(
        [
            node(
                func=fetch_bitcoin_klines,
                inputs={
                    "symbol": "params:binance.symbol",
                    "interval": "params:binance.interval",
                    "years_of_data": "params:binance.years_of_data",
                },
                outputs="raw_bitcoin_data",
                name="fetch_bitcoin_klines_node",
                tags=["ingestion"],
            ),
            node(
                func=validate_raw_data,
                inputs="raw_bitcoin_data",
                outputs="validated_bitcoin_data",
                name="validate_raw_data_node",
                tags=["ingestion"],
            ),
        ],
    )
