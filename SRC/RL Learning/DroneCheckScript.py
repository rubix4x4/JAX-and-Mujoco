import gymnasium as gym
from stable_baselines3.common.env_checker import check_env
from MujocoDroneFolder.MujocoDroneEnv import MjDroneEnv

env = MjDroneEnv(render_mode = 'None')
check_env(env)

