# This code is pulling the skydio xml scene and going through a single step
import mujoco
import os

import gymnasium as gym
from gymnasium import spaces
import random
import time

import matplotlib.pyplot as plt

Path_Total = os.getcwd()

DroneXMLPath = Path_Total + "\\LIB\\mujoco_menagerie-main\\skydio_x2\\scene.xml"
AssetXMLPath = Path_Total + "\\LIB\\mujoco_menagerie-main\\skydio_x2\\assets"
# Build Model from the XML string
#model = mujoco.MjModel.from_xml_string(xml)
 
model = mujoco.MjModel.from_xml_path(DroneXMLPath)

# Pull Data from current model
data = mujoco.MjData(model)

# Step through the model once
mujoco.mj_step(model,data)

Timevec = []
PosVec = []
# Step through the model once
for i in range(1000):
    mujoco.mj_step(model,data)
    Timevec.append(data.time)
    PosVec.append(data.qpos[2])
    print('time ', data.time, 'position ', data.qpos[2])

# NOTES
# position and velocity vector are organized as follows
# x y z , quaternion


plt.style.use('_mpl-gallery')
fig, ax = plt.subplots()
ax.plot(Timevec,PosVec, linewidth = 2.0)
ax.set(xlim = (0,2), ylim = (0, 0.1))

plt.show()
print("End")