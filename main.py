import sys

import mlflow
import numpy as np
from sai_rl import SAIClient
from stable_baselines3 import DDPG, PPO, HerReplayBuffer
from stable_baselines3.common.logger import HumanOutputFormat, KVWriter, Logger
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.ddpg import MultiInputPolicy

from mlflow_tools import MLflowOutputFormat

# https://doi.org/10.3390/act14040165
experiment_name = "DDPG Honelign et al."
mlflow.set_experiment(experiment_name)

loggers = Logger(
    folder=None,
    output_formats=[HumanOutputFormat(sys.stdout), MLflowOutputFormat()],
)

## Initialize the SAI client
sai = SAIClient(comp_id="franka-ml-hiring")


with mlflow.start_run(log_system_metrics=True) as run:
    # Create params dict
    params = {
        "total_timesteps": 10000 * 5,
        "learning_rate": 1e-3,
        "buffer_size": 100000,
        "batch_size": 2048,
        "tau": 0.005,
        "gamma": 0.95,
        # "replay_buffer_class": HerReplayBuffer,
        # "n_sampled_goal": 4,
        # "goal_selection_strategy": "future",
        "policy_class": "MlpPolicy",
        # "net_arch": [256, 256, 256],
    }
    # Log parameters to MLflow
    mlflow.log_params(params)

    ## Make the environment
    env = sai.make_env()

    # The noise objects for DDPG
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions)
    )

    model = DDPG(
        policy=params["policy_class"],
        env=env,
        learning_rate=params["learning_rate"],
        buffer_size=params["buffer_size"],
        batch_size=params["batch_size"],
        tau=params["tau"],
        gamma=params["gamma"],
        action_noise=action_noise,
        # replay_buffer_class=params["replay_buffer_class"],
        # replay_buffer_kwargs=dict(
        #     n_sampled_goal=params["n_sampled_goal"],
        #     goal_selection_strategy=params["goal_selection_strategy"],
        # ),
        # policy_kwargs=dict(net_arch=params["net_arch"]),
        verbose=2,
    )
    model.set_logger(loggers)
    model.learn(
        total_timesteps=params["total_timesteps"], log_interval=1, progress_bar=True
    )

# Save the model
model.save("ddpg_Honelign_et_al")
