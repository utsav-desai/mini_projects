import numpy as np
import pybullet as p
import pybullet_data
import time

p.connect(p.GUI)
p.resetSimulation()
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8)
p.setRealTimeSimulation(0)

# load assets
p.loadURDF('plane.urdf', [0, 0, 0], [0, 0, 0, 1])   # ground
targid = p.loadURDF('franka_panda/panda.urdf', [0, 0, 0], [0, 0, 0, 1], useFixedBase = True)
obj_of_focus = targid

for step in range(100):
    focus_position, _ = p.getBasePositionAndOrientation(targid)
    p.resetDebugVisualizerCamera(cameraDistance = 3, cameraYaw = 0, cameraPitch = -40, cameraTargetPosition = focus_position)
    p.stepSimulation()
    time.sleep(0.01)
