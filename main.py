import numpy as np
import torch.nn.functional as F
from sai_rl import SAIClient

from ddpg import DDPG_FF
from training import training_loop

## Initialize the SAI client
sai = SAIClient(comp_id="franka-golf-challenge")

## Make the environment
env = sai.make_env()

## Create the model
model = DDPG_FF(
    n_features=env.observation_space.shape[0],  # type: ignore
    action_space=env.action_space,  # type: ignore
    neurons=[24, 12, 6],
    activation_function=F.relu,
    learning_rate=0.0001,
)


## Define an action function
def action_function(policy):
    expected_bounds = [-1, 1]
    action_percent = (policy - expected_bounds[0]) / (
        expected_bounds[1] - expected_bounds[0]
    )
    bounded_percent = np.minimum(np.maximum(action_percent, 0), 1)
    return (
        env.action_space.low
        + (env.action_space.high - env.action_space.low) * bounded_percent
    )


## Train the model
training_loop(env, model, action_function)

## Watch
sai.watch(model, action_function)

## Benchmark the model locally
sai.benchmark(model, action_function)

## Save and submit the model
# sai.submit("My First Model", model, action_function)
