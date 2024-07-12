import gymnasium as gym
import os
from stable_baselines3 import SAC
from MujocoDroneFolder.MujocoDroneEnv import MjDroneEnv
import time

models_dir = f"models/MjDrone3/"
logdir = f"logs/DroneTraining/"

if not os.path.exists(models_dir):
	os.makedirs(models_dir)

if not os.path.exists(logdir):
	os.makedirs(logdir)

env = MjDroneEnv(render_mode = 'None')
##
model = SAC("MlpPolicy", env, verbose=1, tensorboard_log=logdir, learning_rate = 0.001)

TIMESTEPS = 5000
iters = 0

while True:
    iters += 1
    model.learn(total_timesteps = TIMESTEPS, reset_num_timesteps = False, tb_log_name=f"DroneTest3")
    model.save(f"{models_dir}/{TIMESTEPS*iters}")
