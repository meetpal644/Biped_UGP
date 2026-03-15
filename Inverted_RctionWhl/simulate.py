import mujoco
import mujoco.viewer
import numpy as np

model = mujoco.MjModel.from_xml_path("Rction_whl.xml")
data = mujoco.MjData(model)

K1 = 50
K2 = 20
K3 = 0

with mujoco.viewer.launch_passive(model,data) as viewer:
    while viewer.is_running():
        theta = data.sensordata[0]
        theta_dot = data.sensordata[1]
        phi_dot = data.sensordata[2]

        tau = K1 * theta + K2 * theta_dot + K3 * phi_dot
        data.ctrl[0] = tau

        mujoco.mj_step(model,data)
        viewer.sync(

        )