import gymnasium as gym
import numpy as np
import time
from stable_baselines3 import SAC

env = gym.make("Pusher", render_mode="human")
##
policy_kwargs = dict(net_arch = [256,256,256]) # Change the baseline perceptron policy to 3 layers with 256 elements
lr = 0.005

model = SAC("MlpPolicy", env, verbose=1, policy_kwargs = policy_kwargs,learning_rate = lr)
model.learn(total_timesteps=10000, log_interval=4)
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