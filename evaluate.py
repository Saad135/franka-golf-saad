import os
import sys

import mlflow
import numpy as np
from sai_rl import SAIClient
from stable_baselines3 import DDPG, PPO, HerReplayBuffer
from stable_baselines3.common.logger import HumanOutputFormat, KVWriter, Logger
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.ddpg import MultiInputPolicy

from mlflow_tools import MLflowOutputFormat, load_model

## Initialize the SAI client
sai = SAIClient(comp_id="franka-ml-hiring")
env = sai.make_env()

# Download the model zip from MLflow and load it
run_id = "e9e728d441f7472e81eb573eac80f093"
artifact_path = "ddpg_Honelign_et_al.zip"
model = load_model(run_id=run_id, artifact_path=artifact_path)

# sai.benchmark(model=model, use_custom_eval=True)

sai.watch(model=model)
