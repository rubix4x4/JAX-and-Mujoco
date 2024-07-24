# This code lets me figure out how to use mujoco and it's various structs in python
import mujoco
import os
import numpy as np
import mujoco.viewer

import gymnasium as gym
from gymnasium import spaces
import random
import time

import matplotlib.pyplot as plt

Path_Total = os.getcwd()

CarXMLPath = Path_Total + f"\SRC\RL Learning\MJCCar\CarXML\Car_Adjusted.xml"
# Build Model from the XML string
#model = mujoco.MjModel.from_xml_string(xml) 
model = mujoco.MjModel.from_xml_path(CarXMLPath)

# Pull Data from current model
data = mujoco.MjData(model)

# Step through the model once
mujoco.mj_step(model,data)

# Launches the generic model viewer
viewer = mujoco.viewer.launch_passive(model,data)
viewer.cam.distance = 10
viewer.cam.elevation = -15

# Pulls image data from a camera
renderer = mujoco.Renderer(model,480,480)
renderer.update_scene(data, camera='TestCam')
pixels = renderer.render()
plt.ioff()
plt.imshow(pixels, interpolation='nearest')
plt.savefig("TestCam_Output.png")




Timevec = []
PosVec = []
# Step through the model once
# with mujoco.viewer.launch_passive(model,data) as viewer:
#     # View Settings
#     viewer.cam.distance = 10
#     viewer.cam.elevation = -15
#     for i in range(1000):
#         mujoco.mj_step(model,data)
#         viewer.sync()
#         data.ctrl = (i/100)*np.ones(4)
#         Timevec.append(data.time)
#         PosVec.append(data.qpos[2])
#         print('time ', data.time, 'position ', data.qpos[2])

# NOTES
# position and velocity vector are organized as follows
# x y z , quaternion

#mujoco.mj_resetData(model,data)

# plt.style.use('_mpl-gallery')
# fig, ax = plt.subplots()
# ax.plot(Timevec,PosVec, linewidth = 2.0)
# ax.set(xlim = (0,max(Timevec)), ylim = (0, max(PosVec)))

# print("End")