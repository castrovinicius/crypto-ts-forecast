"""Nodes for model training pipeline.

This module contains functions to train and evaluate Prophet models.
"""

import logging
from typing import Any

import mlflow
import pandas as pd
from prophet import Prophet

logger = logging.getLogger(__name__)


def optimize_hyperparameters(
    train_data: pd.DataFrame,
    n_trials: int = 30,
    cv_initial: str = "365 days",
    cv_period: str = "30 days",
    cv_horizon: str = "30 days",
    ma_window: int = 21,
    metric: str = "rmse",
) -> dict[str, Any]:
    """Find optimal Prophet hyperparameters using Optuna with cross-validation.

    Uses the training data only — test data must remain unseen during tuning.
    Each trial trains a model with sampled hyperparameters and evaluates it
    with Prophet's time-series cross-validation.

    Args:
        train_data: Training dataset (must NOT include test period).
        n_trials: Number of Optuna trials to run.
        cv_initial: Initial training window for cross-validation (e.g. "365 days").
        cv_period: Spacing between cutoff dates (e.g. "30 days").
        cv_horizon: Forecast horizon evaluated at each cutoff (e.g. "30 days").
        ma_window: Moving average window used when building the model.
        metric: Metric to minimise — "rmse", "mae", or "mape".

    Returns:
        Dictionary with best_params, best_value, metric, study summary, and
        a trials DataFrame for further inspection.
    """
    import optuna
    from prophet.diagnostics import cross_validation, performance_metrics

    # Suppress verbose Stan/Prophet/Optuna output during trials
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    logging.getLogger("prophet").setLevel(logging.WARNING)
    logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

    logger.info(
        f"Starting Optuna hyperparameter search: {n_trials} trials, "
        f"metric={metric}, cv_horizon={cv_horizon}"
    )

    def objective(trial: optuna.Trial) -> float:
        params = {
            "changepoint_prior_scale": trial.suggest_float(
                "changepoint_prior_scale", 0.001, 0.15, log=True
            ),
            "seasonality_prior_scale": trial.suggest_float(
                "seasonality_prior_scale", 0.1, 10.0, log=True
            ),
            "seasonality_mode": trial.suggest_categorical(
                "seasonality_mode", ["additive", "multiplicative"]
            ),
            "changepoint_range": trial.suggest_float(
                "changepoint_range", 0.7, 0.85
            ),
        }

        model = train_prophet_model(
            train_data=train_data,
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            add_regressors=True,
            ma_window=ma_window,
            **params,
        )

        df_cv = cross_validation(
            model,
            initial=cv_initial,
            period=cv_period,
            horizon=cv_horizon,
            parallel=None,
            disable_tqdm=True,
        )
        df_perf = performance_metrics(df_cv, rolling_window=1)
        return float(df_perf[metric].mean())

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_params
    trials_df = study.trials_dataframe()

    logger.info(f"Best {metric}: {study.best_value:.4f}")
    logger.info(f"Best params: {best}")

    return {
        "best_params": best,
        "best_value": float(study.best_value),
        "metric": metric,
        "n_trials": n_trials,
        "trials_df": trials_df,
        "study": study,
    }


def prepare_future_regressors(
    train_data: pd.DataFrame,
    future_dates: pd.Series,
    ma_window: int = 21,
    strategy: str = "last_value",
) -> pd.DataFrame:
    """Prepare future values for regressors using different strategies.

    Since we don't know future values of OHLCV, we need to estimate them.
    Different strategies are available:
    - 'last_value': Use the last known value (recommended for short-term forecasts)
    - 'ma': Use the moving average of last ma_window days
    - 'trend': Project forward using linear trend of last ma_window days

    Args:
        train_data: Training data with regressor columns.
        future_dates: Series of future dates to predict.
        ma_window: Window size for moving averages (default: 21 days).
        strategy: Strategy for estimating future values ('last_value', 'ma', or 'trend').

    Returns:
        DataFrame with ds column and regressor values for future dates.
    """
    # Identify regressor columns (exclude ds, y, and MA columns)
    regressor_columns = [
        col for col in train_data.columns 
        if col not in ["ds", "y"] and not col.endswith(f"_ma{ma_window}")
    ]

    if not regressor_columns:
        return pd.DataFrame({"ds": future_dates})

    logger.info(f"Preparing future values for regressors: {regressor_columns}")
    logger.info(f"Strategy: {strategy}")

    # Create future dataframe
    future_df = pd.DataFrame({"ds": future_dates})
    num_future_days = len(future_dates)

    # For each regressor, estimate future values based on strategy
    for col in regressor_columns:
        if strategy == "last_value":
            # Use the last known value (best for short-term, assumes stability)
            last_value = train_data[col].iloc[-1]
            future_df[col] = last_value
            logger.info(f"  {col}: using last value {last_value:.2f}")

        elif strategy == "ma":
            # Use moving average
            ma_col_name = f"{col}_ma{ma_window}"
            if ma_col_name in train_data.columns:
                last_ma_value = train_data[ma_col_name].iloc[-1]
                future_df[col] = last_ma_value
                logger.info(f"  {col}: using MA value {last_ma_value:.2f}")
            else:
                last_values = train_data[col].tail(ma_window)
                mean_value = last_values.mean()
                future_df[col] = mean_value
                logger.info(f"  {col}: using mean value {mean_value:.2f}")

        elif strategy == "trend":
            # Project forward using linear trend
            last_values = train_data[col].tail(ma_window).values
            x = list(range(len(last_values)))

            # Simple linear regression
            from numpy import polyfit
            if len(last_values) > 1:
                slope, intercept = polyfit(x, last_values, 1)

                # Project into future
                future_x = list(range(len(last_values), len(last_values) + num_future_days))
                future_values = [slope * fx + intercept for fx in future_x]
                future_df[col] = future_values
                logger.info(f"  {col}: using trend projection (slope: {slope:.2f})")
            else:
                # Fallback to last value
                future_df[col] = train_data[col].iloc[-1]
                logger.info(f"  {col}: insufficient data for trend, using last value")
        else:
            raise ValueError(f"Unknown strategy: {strategy}. Use 'last_value', 'ma', or 'trend'.")

    return future_df


def train_prophet_model(
    train_data: pd.DataFrame,
    seasonality_mode: str,
    yearly_seasonality: bool,
    weekly_seasonality: bool,
    daily_seasonality: bool,
    changepoint_prior_scale: float,
    seasonality_prior_scale: float,
    changepoint_range: float,
    add_regressors: bool = True,
    ma_window: int = 21,
) -> Prophet:
    """Train a Prophet model on the training data.

    Args:
        train_data: Training dataset with ds and y columns (and optional regressors).
        seasonality_mode: 'additive' or 'multiplicative'.
        yearly_seasonality: Whether to include yearly seasonality.
        weekly_seasonality: Whether to include weekly seasonality.
        daily_seasonality: Whether to include daily seasonality.
        changepoint_prior_scale: Flexibility of trend changes.
        seasonality_prior_scale: Flexibility of seasonality.
        changepoint_range: Proportion of data to consider for changepoints.
        add_regressors: Whether to add available OHLCV columns as regressors.
        ma_window: Window size for moving averages (default: 21).

    Returns:
        Trained Prophet model.
    """
    logger.info("Initializing Prophet model...")

    model = Prophet(
        seasonality_mode=seasonality_mode,
        yearly_seasonality=yearly_seasonality,
        weekly_seasonality=weekly_seasonality,
        daily_seasonality=daily_seasonality,
        changepoint_prior_scale=changepoint_prior_scale,
        seasonality_prior_scale=seasonality_prior_scale,
        changepoint_range=changepoint_range,
    )

    # Automatically add regressors if available
    if add_regressors:
        # Identify regressor columns (exclude ds, y, and MA columns)
        regressor_columns = [
            col for col in train_data.columns 
            if col not in ["ds", "y"] and not col.endswith(f"_ma{ma_window}")
        ]

        for col in regressor_columns:
            model.add_regressor(col)
            logger.info(f"Added regressor: {col}")

    # Halving cycle removed for hourly data: 2 years of data is not enough
    # to reliably estimate a 4-year cycle (need at least one full cycle)

    logger.info(f"Training Prophet model on {len(train_data)} samples...")
    model.fit(train_data)

    logger.info("Prophet model training completed")

    # Log model parameters to MLflow if active run exists
    if mlflow.active_run():
        mlflow.log_param("model_type", "Prophet")
        mlflow.log_param("training_samples", len(train_data))
        mlflow.log_param("training_start_date", str(train_data["ds"].min()))
        mlflow.log_param("training_end_date", str(train_data["ds"].max()))
        logger.info("Logged training metadata to MLflow")

    return model


def evaluate_model(
    model: Prophet,
    test_data: pd.DataFrame,
    ma_window: int = 21,
) -> dict[str, Any]:
    """Evaluate the trained model on test data.

    Args:
        model: Trained Prophet model.
        test_data: Test dataset with ds and y columns (and optional regressors).
        ma_window: Window size for moving averages (default: 21).

    Returns:
        Dictionary with evaluation metrics.
    """
    logger.info("Evaluating model on test data...")

    # Prepare future dataframe for test period
    future = test_data.copy()
    
    # Remove MA columns if present (they're not needed for prediction)
    ma_columns = [col for col in future.columns if col.endswith(f"_ma{ma_window}")]
    if ma_columns:
        future = future.drop(columns=ma_columns)

    # Make predictions
    forecast = model.predict(future)

    # Calculate metrics
    y_true = test_data["y"].values
    y_pred = forecast["yhat"].values

    # Mean Absolute Error
    mae = float(abs(y_true - y_pred).mean())

    # Mean Absolute Percentage Error
    # Filter out zero values to avoid division by zero
    non_zero_mask = y_true != 0
    if non_zero_mask.any():
        mape = float(
            (
                abs(y_true[non_zero_mask] - y_pred[non_zero_mask])
                / abs(y_true[non_zero_mask])
            ).mean()
            * 100
        )
    else:
        # If all values are zero, MAPE is undefined; use NaN
        mape = float("nan")

    # Root Mean Squared Error
    rmse = float(((y_true - y_pred) ** 2).mean() ** 0.5)

    # R-squared
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    if abs(ss_tot) < 1e-12:
        # R-squared is undefined when there is no variance in y_true
        r2 = float("nan")
        logger.warning(
            "R-squared is undefined because the variance of y_true is zero; "
            "setting r2 to NaN."
        )
    else:
        r2 = float(1 - (ss_res / ss_tot))

    metrics = {
        "mae": mae,
        "mape": mape,
        "rmse": rmse,
        "r2": r2,
        "test_samples": len(test_data),
        "test_start_date": str(test_data["ds"].min()),
        "test_end_date": str(test_data["ds"].max()),
    }

    logger.info("Model evaluation results:")
    logger.info(f"  MAE: ${mae:,.2f}")
    logger.info(f"  MAPE: {mape:.2f}%")
    logger.info(f"  RMSE: ${rmse:,.2f}")
    logger.info(f"  R²: {r2:.4f}")

    return metrics


def create_model_report(
    metrics: dict[str, Any],
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
) -> dict[str, Any]:
    """Create a comprehensive model report.

    Args:
        metrics: Evaluation metrics.
        train_data: Training data for additional stats.
        test_data: Test data for additional stats.

    Returns:
        Comprehensive report dictionary.
    """
    report = {
        "model_type": "Prophet",
        "training_info": {
            "samples": len(train_data),
            "start_date": str(train_data["ds"].min()),
            "end_date": str(train_data["ds"].max()),
            "price_range": {
                "min": float(train_data["y"].min()),
                "max": float(train_data["y"].max()),
                "mean": float(train_data["y"].mean()),
            },
        },
        "test_info": {
            "samples": len(test_data),
            "start_date": str(test_data["ds"].min()),
            "end_date": str(test_data["ds"].max()),
        },
        "metrics": metrics,
    }

    logger.info("Model report created successfully")

    return report
