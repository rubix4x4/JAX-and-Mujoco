import gymnasium as gym
import numpy as np
import time
from stable_baselines3 import SAC



env = gym.make("Pusher", render_mode=None)
##
model = SAC("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000, log_interval=4)
model.save("PusherPolicy")
del model # remove to demonstrate saving and loading

model = SAC.load("PusherPolicy")

# Reset Model to include render
env = gym.make("Pusher", render_mode="human")
obs, info = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()