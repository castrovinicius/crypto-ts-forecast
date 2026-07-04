"""Project settings. There is no need to edit this file unless you want to change values
from the Kedro defaults. For further information, including these default values, see
https://docs.kedro.org/en/stable/kedro_project_setup/settings.html."""

import os

from crypto_ts_forecast.hooks import MLflowHooks, ModelVersioningHooks

# MLflow 2.x+ rejects the file-based tracking backend ("./mlruns") by default.
# This project uses the local file store on purpose (see conf/local/mlflow.yml),
# so opt back in. ``setdefault`` preserves any value already set in the env.
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

# Instantiated project hooks.
# Hooks are executed in a Last-In-First-Out (LIFO) order.
# MLflow hooks are registered to enable automatic experiment tracking,
# metrics logging, and model versioning following MLOps best practices.
HOOKS = (
    MLflowHooks(),
    ModelVersioningHooks(),
)


# Installed plugins for which to disable hook auto-registration.
# DISABLE_HOOKS_FOR_PLUGINS = ("kedro-viz",)

# Class that manages storing KedroSession data.
# from kedro.framework.session.store import BaseSessionStore
# SESSION_STORE_CLASS = BaseSessionStore
# Keyword arguments to pass to the `SESSION_STORE_CLASS` constructor.
# SESSION_STORE_ARGS = {
#     "path": "./sessions"
# }

# Directory that holds configuration.
# CONF_SOURCE = "conf"

# Class that manages how configuration is loaded.
# from kedro.config import OmegaConfigLoader

# CONFIG_LOADER_CLASS = OmegaConfigLoader

# Keyword arguments to pass to the `CONFIG_LOADER_CLASS` constructor.
CONFIG_LOADER_ARGS = {
    "base_env": "base",
    "default_run_env": "local",
    # "config_patterns": {
    #     "spark" : ["spark*/"],
    #     "parameters": ["parameters*", "parameters*/**", "**/parameters*"],
    # }
}

# Class that manages Kedro's library components.
# from kedro.framework.context import KedroContext
# CONTEXT_CLASS = KedroContext

# Class that manages the Data Catalog.
# from kedro.io import DataCatalog
# DATA_CATALOG_CLASS = DataCatalog
