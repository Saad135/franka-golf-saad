import os
import sys

import mlflow
import numpy as np
from sai_rl import SAIClient
from stable_baselines3 import DDPG, PPO, HerReplayBuffer
from stable_baselines3.common.logger import HumanOutputFormat, KVWriter, Logger
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.ddpg import MultiInputPolicy

from mlflow_tools import MLflowOutputFormat, save_model


def get_params():
    """
    Returns the parameters for the DDPG model.
    """
    return {
        "total_timesteps": 10000 * 2,
        "learning_rate": 1e-3,
        "buffer_size": 100000,
        "batch_size": 2048,
        "tau": 0.005,
        "gamma": 0.95,
        "policy_class": "MlpPolicy",  # Use MultiInputPolicy for environments with dict observation space
    }


def get_action_noise(env):
    """
    Returns the action noise for the DDPG model.
    """
    n_actions = env.action_space.shape[-1]
    return NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))


def main():
    # https://doi.org/10.3390/act14040165
    experiment_name = "DDPG Honelign et al."
    mlflow.set_experiment(experiment_name)

    loggers = Logger(
        folder=None,
        output_formats=[HumanOutputFormat(sys.stdout), MLflowOutputFormat()],
    )

    ## Initialize the SAI client
    sai = SAIClient(comp_id="franka-ml-hiring")

    # Create params dict
    params = get_params()

    with mlflow.start_run(log_system_metrics=True) as run:
        # Log parameters to MLflow
        mlflow.log_params(params)

        ## Make the environment
        env = sai.make_env()

        # The noise objects for DDPG
        action_noise = get_action_noise(env)

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
        model_name = "ddpg_Honelign_et_al"
        save_model(model, model_name)


if __name__ == "__main__":
    main()
