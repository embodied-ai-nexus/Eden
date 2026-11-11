"""Visualize a live Eden scene in the browser with the Viser viewer.

---
tier: recipe
capabilities: [render.viser, robot.go1]
hardware: cpu
optional_deps: [viser]
run: uv run python examples/recipes/rendering/viser_visualization.py
---
"""

import time
import eden as en
from eden.envs.base import EnvBase
from eden.options import EnvOptions, PlaneOptions, SceneOptions
from eden.options.entities import EntityOptions

# from eden.robots import Go2
# from eden.robots import UR5e
from eden.extensions.visualization.viser import ViserViewer
from eden.options.camera import CameraOptions, CamerasOptions


def main():
    en.init(debug=True)

    # Create scene with a plane and a box
    # Using SceneOptions directly to add arbitrary entities
    scene_options = SceneOptions(
        plane=PlaneOptions(),
        apple=EntityOptions(
            file="objects/apples/013_apple/textured.obj",
            is_fixed_base=False,
            default_root_pos=(0.0, 0.0, 1.0),
        ),
        # box=BoxOptions(
        #     pos=(0.0, 0.5, 10.0), size=(0.2, 0.2, 0.2),
        #     fixed=False,
        #     default_root_pos=(0.0, 0.5, 2.0) # Start slightly higher to fall
        # ),
        robot=en.robots.Go1(),
        # robot=Go2()
        # robot=UR5e()
    )

    env = EnvBase(
        env_options=EnvOptions(num_envs=4, show_world_frame=True, env_spacing=(1.5, 1.5)),
        scene_options=scene_options,
        cameras_options=CamerasOptions(
            camera=CameraOptions(
                cam_pos=(2.0, 0.0, 1.0),
                cam_lookat=(0.0, 0.0, 0.0),
                cam_fov=40,
            )
        ),
        show_viewer=False,  # We use Viser instead
    )
    env.build()

    # Initialize Viser
    vis = ViserViewer(env, port=8080)
    vis.build()

    env.reset()

    # Simulation loop
    while True:
        env.step()
        vis.update()
        time.sleep(0.1)


if __name__ == "__main__":
    main()
