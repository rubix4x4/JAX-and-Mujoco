import mujoco
import mujoco.viewer
import os
import gymnasium as gym
from gymnasium import spaces
import random
import time
import math
import numpy as np

class MjDroneEnv(gym.Env):
    metadata = {'render.modes': ['human','None'], "render_fps":4}
    def __init__(self, render_mode):
        super(MjDroneEnv, self) .__init__()
        # Load Mujoco Model
        Path_Total = os.getcwd()
        DroneXMLPath = Path_Total + "\\LIB\\mujoco_menagerie-main\\skydio_x2\\scene.xml"
        # AssetXMLPath = Path_Total + "\\LIB\\mujoco_menagerie-main\\skydio_x2\\assets"
        self.model = mujoco.MjModel.from_xml_path(DroneXMLPath)
        self.data = mujoco.MjData(self.model)
        self.viewer = None
        self.render_mode = render_mode
        # This is where the drone wants to go
        self.targetLocation = np.array([0,0,2,1,0,0,0])
        self.targetVel = np.array([0,0,0,0,0,0])
        # Action and Observation Spaces
        ControlMax = np.array([1,1,1,1]) # Consider Normalizing These to -1 and 1 later
        ControlMin = np.array([-1,-1,-1,-1])
        self.action_space = spaces.Box(low=ControlMin, high = ControlMax, dtype=np.float32) # Four Motors on the drone
        self.observation_space = spaces.Box(low = -np.inf, high = np.inf, shape = (self.model.nq + self.model.nv,), dtype=np.float32) # XYZ, maybe quaternion? Start with just cartesian pos
        # Note in above, shape input needs to be an iterable, hence the , comma
        
        
        if self.render_mode == 'human':
            self.viewer = mujoco.viewer.launch_passive(self.model,self.data)
            self.viewer.cam.distance = 10
            self.viewer.cam.elevation = -15
            
        elif self.render_mode == None:
            self.viewer = None
            
        
    def step(self,action,):
        
        self.render()
        scaledaction = action*(13/2) + (13/2)   # rescale action from -1,1 back to 0,13
        self.data.ctrl = scaledaction
        
        mujoco.mj_step(self.model,self.data)
        
        self.observation = self.get_obs()       # Retrieve Observations
        
        self.reward = self.reward_func()        # Compute Rewards
        
        self.termination = self.termination_func()  # Compute if termination
        
        self.truncation = self.trunc_func()     # Compuite if truncated
        info = {}
        # obs, reward, terminated, truncated, info
        return self.observation, self.reward, self.termination, self.truncation, info
    
    
    def reset(self, seed = None):
        mujoco.mj_resetData(self.model,self.data)
        self.observation = self.get_obs()
        info = {}
        return self.observation, info
        
    def reward_func(self):
        Dist_to_target = math.dist(self.targetLocation,self.data.qpos) # first XYZ elements of qpos
        DistReward = (5)/(1+Dist_to_target) # Increase the reward as you approach target, and continues reward for staying there
        
        # Penalize flipping over/ excessive angular velocity
        # Angular Velocities are expressed in Radians
        if Dist_to_target < 0.5:
            VelocityAbs = abs(self.data.qvel)
            VelocityValue = sum(VelocityAbs)/len(VelocityAbs)
            VelocityLog = math.log(VelocityValue,10)
            VelPenalty = -1*(VelocityLog/1) # no velocity penalty for now
        else:
            VelPenalty = 0
        # Penalize hovering close to ground
        if self.data.qpos[2] < .5:
            HeightPenalty = -1
        else:
            HeightPenalty = 0
            
        Reward = DistReward + VelPenalty + HeightPenalty
        return Reward
    
    def termination_func(self):
        # This is where I would write a case where the simulation would terminate early
        Accelerations = self.data.qacc[0:3]
        if any(Accelerations > 100):
            return True
        return False
        
    def get_obs(self):
        return np.concatenate((self.data.qpos - self.targetLocation, self.data.qvel), dtype=np.float32)
    
    def trunc_func(self):
        if self.data.time > 5:
            return True
        else:
            return False
    def render(self):
        if self.render_mode == 'human':
            self.viewer.sync()
        elif self.render_mode == 'None':
            self.viewer = None