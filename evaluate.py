import sys

import mlflow
import numpy as np
from sai_rl import SAIClient
from stable_baselines3 import DDPG, PPO, HerReplayBuffer
from stable_baselines3.common.logger import HumanOutputFormat, KVWriter, Logger
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.ddpg import MultiInputPolicy

from mlflow_tools import MLflowOutputFormat

## Initialize the SAI client
sai = SAIClient(comp_id="franka-ml-hiring")
env = sai.make_env()

model = DDPG.load("ddpg_Honelign_et_al.zip", env=env)

# sai.benchmark(model=model, use_custom_eval=True)

sai.watch(model=model)
