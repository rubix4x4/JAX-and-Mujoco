import gymnasium as gym
import os
from stable_baselines3 import SAC
from MujocoDroneFolder.MujocoDroneEnv import MjDroneEnv
from MujocoDroneFolder.MujocoDroneEnvRandom import MjDroneEnvRand
import time

env = MjDroneEnvRand(render_mode = 'human')
models_dir = f"models/MjDroneRand/190000"
model = SAC.load(models_dir)

obs, info = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()