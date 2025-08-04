from sai_rl import SAIClient
from stable_baselines3 import PPO

## Initialize the SAI client
sai = SAIClient(comp_id="franka-ml-hiring")

## Make the environment
env = sai.make_env()

## Define the model
model = PPO("MlpPolicy", env)
model.learn(total_timesteps=100)

## Benchmark the model locally
sai.benchmark(model, use_custom_eval=True)

## Save and submit the model
# sai.submit("Default PPO template", model)
