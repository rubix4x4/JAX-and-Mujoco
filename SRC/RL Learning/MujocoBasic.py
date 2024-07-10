# This code is pulling the skydio xml scene and going through a single step

import mujoco
import os

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



print("Checkpoint")