import os
from typing import Any, Dict, Tuple, Union

import mlflow
import numpy as np
from stable_baselines3 import DDPG
from stable_baselines3.common.logger import KVWriter


# Fuction to save model parameters and log them to MLflow and then delete the model file
def save_model(model, model_name):
    """
    Saves the model and logs it to MLflow.
    """
    model.save(model_name)
    mlflow.log_artifact(f"{model_name}.zip")
    # Optionally, you can delete the model file after logging
    os.remove(f"{model_name}.zip")


# Function to download the model from MLflow and load it
def load_model(run_id, artifact_path="model.zip"):
    """
    Constructs the artifact URI from run_id and artifact_path, downloads the model from MLflow, and loads it.
    """
    artifact_uri = f"runs:/{run_id}/{artifact_path}"
    mlflow_model_path = mlflow.artifacts.download_artifacts(artifact_uri)
    return DDPG.load(mlflow_model_path)


# Copied from https://stable-baselines3.readthedocs.io/en/master/guide/integrations.html#mlflow
class MLflowOutputFormat(KVWriter):
    """
    Dumps key/value pairs into MLflow's numeric format.
    """

    def write(
        self,
        key_values: Dict[str, Any],
        key_excluded: Dict[str, Union[str, Tuple[str, ...]]],
        step: int = 0,
    ) -> None:

        for (key, value), (_, excluded) in zip(
            sorted(key_values.items()), sorted(key_excluded.items())
        ):

            if excluded is not None and "mlflow" in excluded:
                continue

            if isinstance(value, np.ScalarType):
                if not isinstance(value, str):
                    mlflow.log_metric(key, value, step)
