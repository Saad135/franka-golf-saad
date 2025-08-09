import os
from typing import Any, Callable, Dict, Tuple, Union

import mlflow
import numpy as np
from stable_baselines3 import DDPG
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.logger import KVWriter
from stable_baselines3.common.vec_env import DummyVecEnv


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


# Extend the EvalCallback to log to mlflow
class EvalCallbackWithMLflow(EvalCallback):
    """
    Custom EvalCallback that logs evaluation metrics to MLflow.
    """

    def _on_step(self) -> bool:
        super()._on_step()
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Log evaluation metrics to MLflow
            mlflow.log_metric(
                f"eval_last_mean_reward", self.last_mean_reward, step=self.n_calls
            )

        return True
