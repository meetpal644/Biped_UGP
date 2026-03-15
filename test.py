import mujoco
import mujoco.viewer
import numpy as np

model = mujoco.MjModel.from_xml_path("low_dof_bot.xml")
data = mujoco.MjData(model)


# Make sure state is nominal
data.qpos[:] = env.init_qpos
data.qvel[:] = env.init_qvel

mujoco.mj_forward(model, data)

root_id = model.body("torso").id
nominal_com_height = data.subtree_com[root_id][2]

with mujoco.viewer.launch_passive(model,data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model,data)
        viewer.sync()