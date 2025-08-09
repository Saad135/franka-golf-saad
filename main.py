import sys

import mlflow
import numpy as np
from sai_rl import SAIClient
from stable_baselines3 import DDPG
from stable_baselines3.common.callbacks import (
    EvalCallback,
    StopTrainingOnNoModelImprovement,
)
from stable_baselines3.common.logger import HumanOutputFormat, Logger
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise

from mlflow_tools import EvalCallbackWithMLflow, MLflowOutputFormat, save_model
from sai_tools import evaluation_fn


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


def get_learn_callbacks(eval_env):
    """
    Returns the callbacks for the training of the DDPG model.
    """
    # Stop training if there is no improvement after more than 3 evaluations
    stop_train_callback = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=3,
        min_evals=5,
        verbose=1,
    )
    eval_callback = EvalCallbackWithMLflow(
        eval_env,
        eval_freq=1000,
        callback_after_eval=stop_train_callback,
        best_model_save_path="./logs/",
        verbose=1,
    )

    return [
        # Add any other callbacks you want to use
        eval_callback,
    ]


def get_eval_env(comp_id):
    """
    Returns the evaluation environment.
    """
    return Monitor(SAIClient(comp_id=comp_id).make_env())


def main():
    # https://doi.org/10.3390/act14040165
    experiment_name = "DDPG Honelign et al."
    mlflow.set_experiment(experiment_name)

    loggers = Logger(
        folder=None,
        output_formats=[HumanOutputFormat(sys.stdout), MLflowOutputFormat()],
    )

    comp_id = "franka-ml-hiring"  # Change this to your competition ID if needed
    ## Initialize the SAI client
    sai = SAIClient(comp_id=comp_id)

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

        # Get the evaluation environment
        eval_env = get_eval_env(comp_id)

        # Get the callbacks
        callbacks = get_learn_callbacks(eval_env)

        model.learn(
            total_timesteps=params["total_timesteps"],
            log_interval=1,
            progress_bar=True,
            callback=callbacks,
        )

        # Save the best model
        best_model_path = f"./logs/best_model"
        mlflow.log_artifact(f"{best_model_path}.zip")

        # Save the final model
        model_name = f"final_ddpg_{params['total_timesteps']}_steps"
        save_model(model, model_name)


if __name__ == "__main__":
    main()
