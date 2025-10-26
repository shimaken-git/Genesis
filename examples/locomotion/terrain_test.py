import numpy as np
import genesis as gs

# 1. initialise Genesis
gs.init(seed=0, backend=gs.gpu)  # use gs.cpu for CPU backend

# 2. create a scene
scene = gs.Scene(show_viewer=True)

# 3. prepare a height map (here a simple bump for demo)
hf = np.zeros((40, 40), dtype=np.int16)
hf[10:30, 10:30] = 200 * np.hanning(20)[:, None] * np.hanning(20)[None, :]

horizontal_scale = 0.25  # metres between grid points
vertical_scale   = 0.005  # metres per height-field unit

# 4. add the terrain entity
scene.add_entity(
    morph=gs.morphs.Terrain(
        height_field=hf,
        horizontal_scale=horizontal_scale,
        vertical_scale=vertical_scale,
    ),
)

scene.build()

# run the sim so you can inspect the surface
for _ in range(1_000):
    scene.step()