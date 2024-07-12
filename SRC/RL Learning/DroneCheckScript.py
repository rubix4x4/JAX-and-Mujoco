import gymnasium as gym
from stable_baselines3.common.env_checker import check_env
from MujocoDroneFolder.MujocoDroneEnv import MjDroneEnv

env = MjDroneEnv()
check_env(env)

