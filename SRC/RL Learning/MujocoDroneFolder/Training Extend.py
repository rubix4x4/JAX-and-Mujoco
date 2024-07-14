# Training Extend
# This script loads a model that has already been trained, loads a new environment, and then continues training of the model in a new environment.

import gymnasium as gym
import os
from stable_baselines3 import SAC
#from MujocoDroneFolder.MujocoDroneEnv import MjDroneEnv
from MujocoDroneFolder.MujocoDroneEnvRandom import MjDroneEnvRand

import time

model_load = f"models/MjDroneRand/190000"
models_dir = f"models/MjDroneRandCont/"
logdir = f"logs/DroneTraining/"

if not os.path.exists(models_dir):
	os.makedirs(models_dir)

if not os.path.exists(logdir):
	os.makedirs(logdir)

env = MjDroneEnvRand(render_mode = 'human')
##
Lr = 0.00005      # Base = 0.0003
gamma = 0.99    # Base = 0.99
tau = 0.005     # Base = 0.005
model = SAC.load(model_load, custom_objects = {'learning_rate': Lr})
model.set_env(env)
TIMESTEPS = 10000
iters = 0

while True:
    iters += 1
    model.learn(total_timesteps = TIMESTEPS, reset_num_timesteps = False, tb_log_name=f"DroneTestRandCont", log_interval= 10, )
    model.save(f"{models_dir}/{TIMESTEPS*iters}")
