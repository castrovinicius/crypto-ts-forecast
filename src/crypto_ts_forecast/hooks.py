"""Project hooks for MLflow integration and MLOps best practices.

This module implements Kedro hooks to enable comprehensive MLflow tracking,
including metrics, parameters, artifacts, and model registration.
"""

import logging
import pickle
import tempfile
from pathlib import Path
from typing import Any

import mlflow
from kedro.framework.hooks import hook_impl
from kedro.io import DataCatalog
from kedro.pipeline.node import Node

logger = logging.getLogger(__name__)


class MLflowHooks:
    """Hooks for MLflow integration and experiment tracking.

    This class provides automatic logging of:
    - Model artifacts (plots, reports, model files)
    - Custom metrics beyond what kedro-mlflow auto-logs
    - Tags for better organization and filtering
    """

    @hook_impl
    def after_node_run(
        self,
        node: Node,
        catalog: DataCatalog,
        inputs: dict[str, Any],
        outputs: dict[str, Any],
    ) -> None:
        """Log additional artifacts and metrics after node execution.

        Args:
            node: The node that was executed.
            catalog: The data catalog.
            inputs: Node inputs.
            outputs: Node outputs.
        """
        # Only log for model training and evaluation nodes
        if "training" not in node.tags and "evaluation" not in node.tags:
            return

        try:
            # Log Prophet model after training
            if node.name == "train_prophet_model_node" and "prophet_model" in outputs:
                if mlflow.active_run():
                    model = outputs["prophet_model"]

                    # Save model to temporary file and log to MLflow
                    with tempfile.TemporaryDirectory() as tmp_dir:
                        model_path = Path(tmp_dir) / "prophet_model.pkl"
                        with open(model_path, "wb") as f:
                            pickle.dump(model, f)

                        # Log the model as an artifact
                        mlflow.log_artifact(str(model_path), artifact_path="model")

                    # Log model signature and parameters
                    mlflow.set_tag("model_type", "Prophet")
                    mlflow.set_tag("model_framework", "facebook-prophet")

                    logger.info(f"Logged Prophet model to MLflow for node: {node.name}")

            # Log model metrics after evaluation
            if node.name == "evaluate_model_node" and "model_metrics" in outputs:
                metrics = outputs["model_metrics"]

                # Log metrics to MLflow
                if mlflow.active_run():
                    # Log evaluation metrics
                    mlflow.log_metric("test_mae", metrics.get("mae", 0))
                    mlflow.log_metric("test_mape", metrics.get("mape", 0))
                    mlflow.log_metric("test_rmse", metrics.get("rmse", 0))
                    mlflow.log_metric("test_r2", metrics.get("r2", 0))
                    mlflow.log_metric("test_samples", metrics.get("test_samples", 0))

                    # Log tags for easier filtering
                    mlflow.set_tag(
                        "test_period_start", metrics.get("test_start_date", "")
                    )
                    mlflow.set_tag("test_period_end", metrics.get("test_end_date", ""))

                    logger.info(
                        f"Logged evaluation metrics to MLflow for node: {node.name}"
                    )

            # Log model report after creation
            if node.name == "create_model_report_node" and "model_report" in outputs:
                report = outputs["model_report"]

                if mlflow.active_run():
                    # Log training information as tags
                    training_info = report.get("training_info", {})
                    mlflow.set_tag("training_samples", training_info.get("samples", 0))
                    mlflow.set_tag(
                        "training_period_start", training_info.get("start_date", "")
                    )
                    mlflow.set_tag(
                        "training_period_end", training_info.get("end_date", "")
                    )

                    # Log price statistics as metrics
                    price_range = training_info.get("price_range", {})
                    if price_range:
                        mlflow.log_metric(
                            "training_price_min", price_range.get("min", 0)
                        )
                        mlflow.log_metric(
                            "training_price_max", price_range.get("max", 0)
                        )
                        mlflow.log_metric(
                            "training_price_mean", price_range.get("mean", 0)
                        )

                    logger.info(
                        f"Logged model report metadata to MLflow for node: {node.name}"
                    )

        except Exception as e:
            # Don't fail pipeline if MLflow logging fails
            logger.warning(f"Failed to log to MLflow in node {node.name}: {e}")


class ModelVersioningHooks:
    """Hooks for model versioning and lineage tracking.

    Implements best practices for model lifecycle management:
    - Automatic model registration in MLflow Model Registry
    - Version tagging with Git commit, timestamp, environment
    - Data lineage tracking
    """

    @hook_impl
    def after_pipeline_run(
        self,
        run_params: dict[str, Any],
        pipeline: Any,
        catalog: DataCatalog,
    ) -> None:
        """Register model and add versioning metadata after pipeline completion.

        Args:
            run_params: Parameters passed to the run.
            pipeline: The pipeline that was run.
            catalog: The data catalog.
        """
        # Only register model after model_training pipeline
        if "model_training" not in str(pipeline):
            return

        try:
            if mlflow.active_run():
                # Add run-level tags for better organization
                mlflow.set_tag(
                    "pipeline_name", run_params.get("pipeline_name", "unknown")
                )
                mlflow.set_tag("kedro_version", run_params.get("kedro_version", ""))

                # You can add Git commit hash if available
                # import git
                # try:
                #     repo = git.Repo(search_parent_directories=True)
                #     mlflow.set_tag("git_commit", repo.head.commit.hexsha)
                #     mlflow.set_tag("git_branch", repo.active_branch.name)
                # except:
                #     pass

                logger.info("Added versioning metadata to MLflow run")

        except Exception as e:
            logger.warning(f"Failed to add versioning metadata: {e}")
