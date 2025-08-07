import sys

import mlflow
from sai_rl import SAIClient
from stable_baselines3 import PPO
from stable_baselines3.common.logger import HumanOutputFormat, KVWriter, Logger

from mlflow_tools import MLflowOutputFormat

experiment_name = "SB3 Default PPO Example"
mlflow.set_experiment(experiment_name)

loggers = Logger(
    folder=None,
    output_formats=[HumanOutputFormat(sys.stdout), MLflowOutputFormat()],
)


with mlflow.start_run(log_system_metrics=True) as run:
    ## Initialize the SAI client
    sai = SAIClient(comp_id="franka-ml-hiring")

    # Create params dict
    params = {
        "total_timesteps": 2048 * 4,
    }
    # Log parameters to MLflow
    mlflow.log_params(params)

    ## Make the environment
    env = sai.make_env()

    ## Define the model
    model = PPO("MlpPolicy", env)
    model.set_logger(loggers)
    model.learn(
        total_timesteps=params["total_timesteps"], log_interval=1, progress_bar=True
    )

    # Save the model to MLflow

    ## Benchmark the model locally
    # sai.benchmark(model, use_custom_eval=True)

## Save and submit the model
# sai.submit("Default PPO template", model)

## Watch the model in the SAI dashboard
# sai.watch(model, use_custom_eval=True)
