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


# Stable Baselines3 learn Callback to implement early stopping based on critic loss
class EarlyStoppingCallback:
    """
    Callback to stop training if the critic loss does not improve for a certain number of steps.
    """

    def __init__(self, patience: int = 1000):
        self.patience = patience
        self.best_critic_loss = np.inf
        self.steps_without_improvement = 0

    def __call__(self, locals_, globals_):
        critic_loss = locals_.get("critic_loss", np.inf)
        if critic_loss < self.best_critic_loss:
            self.best_critic_loss = critic_loss
            self.steps_without_improvement = 0
        else:
            self.steps_without_improvement += 1

        return self.steps_without_improvement < self.patience


# Stable Baselines3 learn callback to record the best reward and stop training if it does not improve for a certain number of steps
class EarlyStoppingBasedOnRewardCallback:
    """
    Callback to record the best reward and stop training if it does not improve for a certain number of steps.
    """

    def __init__(self, patience: int = 1000):
        self.best_reward = -np.inf
        self.patience = patience
        self.steps_without_improvement = 0

    def __call__(self, locals_, globals_):
        current_reward = locals_.get("reward", -np.inf)
        if current_reward > self.best_reward:
            self.best_reward = current_reward
            self.steps_without_improvement = 0
        else:
            self.steps_without_improvement += 1

        return self.steps_without_improvement < self.patience


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


# Stable Baselines3 learn callback to calculate custom training metrics and log them to MLflow
class CustomMetricsCallback:
    """
    Callback to calculate custom training metrics and log them to MLflow.
    """

    def __init__(self):
        pass

    def calculate_metrics(self, locals_, globals_) -> Dict[str, Any]:
        """
        Override this method to calculate and return custom metrics as a dictionary.
        """
        # Example: return {"loss": locals_.get("loss", 0)}
        return {}

    def __call__(self, locals_, globals_):
        metrics = self.calculate_metrics(locals_, globals_)
        for key, value in metrics.items():
            mlflow.log_metric(key, value)
        return True
