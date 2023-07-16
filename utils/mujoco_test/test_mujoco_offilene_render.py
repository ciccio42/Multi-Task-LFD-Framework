import os, math
from mujoco_py import load_model_from_xml, MjSim

model = load_model_from_xml("test_model.xml")
sim = MjSim(model)
# viewer = MjViewer(sim)
while True:
    res = sim.render(255, 255, camera_name="rgb")
    print(type(res), res.shape)
    sim.step()
    # viewer.render()
    if t > 100 and os.getenv('TESTING') is not None:
        break