import gymnasium as gym
import os
from stable_baselines3 import SAC
from MujocoDroneFolder.MujocoDroneEnv import MjDroneEnv
import time

models_dir = f"models/MjDrone2/"
logdir = f"logs/DroneTraining/"

if not os.path.exists(models_dir):
	os.makedirs(models_dir)

if not os.path.exists(logdir):
	os.makedirs(logdir)

env = MjDroneEnv(render_mode = 'human')
##
Lr = 0.0003      # Base = 0.0003
gamma = 0.99    # Base = 0.99
tau = 0.005     # Base = 0.005
model = SAC("MlpPolicy", env, verbose=1, tensorboard_log=logdir, learning_rate = Lr, gamma= gamma, tau= tau)

TIMESTEPS = 5000
iters = 0

while True:
    iters += 1
    model.learn(total_timesteps = TIMESTEPS, reset_num_timesteps = False, tb_log_name=f"DroneTest2")
    model.save(f"{models_dir}/{TIMESTEPS*iters}")
