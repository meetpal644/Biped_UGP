import mujoco
import numpy as np
from mujoco import viewer
import time
model = mujoco.MjModel.from_xml_string("""
<mujoco>
  <worldbody>
    <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
    <geom type="plane" size="5 5 .1"/>
  </worldbody>
</mujoco>
""")

data = mujoco.MjData(model)

with viewer.launch_passive(model, data) as v:
        while v.is_running():
            mujoco.mj_step(model, data)
            v.sync()
            time.sleep(model.opt.timestep)