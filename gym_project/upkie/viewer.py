import mujoco
import mujoco.viewer
import numpy as np

# Load model
model = mujoco.MjModel.from_xml_path("upkie.xml")
data = mujoco.MjData(model)

# ---------------------------------------------------------
# Reset to keyframe "nominal"
# ---------------------------------------------------------

key_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_KEY,
    "nominal"
)

if key_id == -1:
    raise RuntimeError("No keyframe named 'nominal' found in XML")

# Apply keyframe
data.qpos[:] = model.key_qpos[key_id]
data.qvel[:] = model.key_qvel[key_id]

# Forward kinematics
mujoco.mj_forward(model, data)

# ---------------------------------------------------------
# Get whole-robot COM height
# ---------------------------------------------------------

root_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_BODY,
    "torso"
)

nominal_com_height = data.subtree_com[root_id][2]

print("\nNominal Upright COM Height:", nominal_com_height, "meters\n")

# ---------------------------------------------------------
# Launch viewer (optional)
# ---------------------------------------------------------

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        viewer.sync()