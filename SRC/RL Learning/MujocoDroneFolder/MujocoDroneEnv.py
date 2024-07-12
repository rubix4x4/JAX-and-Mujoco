import mujoco
import os
import gymnasium as gym
from gymnasium import spaces
import random
import time
import numpy as np

class MjDroneEnv(gym.Env):
    metadata = {'render.modes': ['human','rgb_array','None']}
    def __init__(self):
        super(MjDroneEnv, self) .__init__()
        
        # Load Mujoco Model
        Path_Total = os.getcwd()
        DroneXMLPath = Path_Total + "\\LIB\\mujoco_menagerie-main\\skydio_x2\\scene.xml"
        # AssetXMLPath = Path_Total + "\\LIB\\mujoco_menagerie-main\\skydio_x2\\assets"
        self.model = mujoco.MjModel.from_xml_path(DroneXMLPath)
        self.data = mujoco.MjData(self.model)
        self.viewer = None
        
        # Action and Observation Spaces
        ControlMax = np.array([13,13,13,13]) # Consider Normalizing These to -1 and 1 later
        ControlMin = np.array([0,0,0,0])
        self.action_space = spaces.Box(low=ControlMin, high = ControlMax, dtype=np.float32) # Four Motors on the drone
        self.observation_space = spaces.Box(low = -np.inf, high = np.inf, shape = (self.model.nq + self.model.nv,), dtype=np.float32) # XYZ, maybe quaternion? Start with just cartesian pos
        # Note in above, shape input needs to be an iterable, hence the , comma

    def step(self,action):
        self.data.ctrl = action # Action must be a numpy array
        mujoco.mj_step(self.model,self.data)
        
        self.observation = self.get_obs()
        
        self.reward = self.reward_func()
        
        self.termination = self.termination_func()
        
        self.truncation = self.trunc_func()
        info = {}
        # obs, reward, terminated, truncated, info
        return self.observation, self.reward, self.termination, self.truncation, info
    
    
    def reset(self, seed = None):
        mujoco.mj_resetData(self.model,self.data)
        self.observation = self.get_obs()
        info = {}
        return self.observation, info
        
    def reward_func(self):
        return 0
    
    def termination_func(self):
        return False
        
    def get_obs(self):
        return np.concatenate((self.data.qpos, self.data.qvel), dtype=np.float32)
    
    def trunc_func(self):
        if self.data.time > 20:
            return True
        else:
            return False

print("Checkpoint")