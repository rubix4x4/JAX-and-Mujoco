import mujoco
import os

import gymnasium as gym
from gymnasium import spaces
import random
import time

Path_Total = os.getcwd()

DroneXMLPath = Path_Total + "\\LIB\\mujoco_menagerie-main\\skydio_x2\\scene.xml"
AssetXMLPath = Path_Total + "\\LIB\\mujoco_menagerie-main\\skydio_x2\\assets"
model = mujoco.MjModel.from_xml_path(DroneXMLPath)
data = mujoco.MjData(model)

# Step through the model once
for i in range(20):
    mujoco.mj_step(model,data)

class MjDroneEnv(gym.Env):
    metadata = {'render.modes': ['human','rgbarray','None']}
    def __init__():
        super(MjDroneEnv, self) .__init__()
        
        
        self.action_space = spaces.Box(4) # Four Motors o n the drone
        self.observation_space = spaces.Box(3) # XYZ, maybe quaternion? Start with just cartesian pos

print("Checkpoint")