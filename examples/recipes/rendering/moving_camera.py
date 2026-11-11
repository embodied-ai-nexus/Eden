"""Sweep a camera through the scene each step and render RGB from the moving viewpoint.

---
tier: recipe
capabilities: [render.camera, robot.franka]
hardware: gpu
optional_deps: []
run: uv run python examples/recipes/rendering/moving_camera.py
---

Eden port of Genesis' ``examples/rendering/moving_camera.py``.

A plane and a Franka Emika Panda sit in the scene while a single camera slides
along +x, one small step per frame, re-rendering RGB from the new pose. It shows
the imperative camera API on the Eden ``Camera`` wrapper: ``set_pose(pos=...)``
to reposition a (legacy/rasterizer-backed) camera between steps, then ``render``.

Cameras are declared up front via ``cameras_options`` on the env constructor and
reached through ``env.cameras["cam"]``.

    uv run python examples/recipes/rendering/moving_camera.py        # headless render
    uv run python examples/recipes/rendering/moving_camera.py -v     # viewer (needs a display)
"""

import argparse

import genesis as gs

import eden as en
from eden.envs.base import EnvBase
from eden.options import EnvOptions, SceneOptions
from eden.options.camera import CameraOptions, CamerasOptions
from eden.options.entities import PlaneOptions
from eden.options.robots import FrankaEmikaPanda


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-v", "--vis", action="store_true", help="Open the interactive viewer")
    args = parser.parse_args()

    en.init(backend=gs.gpu)

    env = EnvBase(
        env_options=EnvOptions(num_envs=1),
        scene_options=SceneOptions(
            plane=PlaneOptions(),
            franka=FrankaEmikaPanda(),
        ),
        cameras_options=CamerasOptions(
            cam=CameraOptions(
                cam_pos=(3.5, 0.0, 2.5), cam_lookat=(0.0, 0.0, 0.5), cam_resolution=(1280, 960), cam_fov=30.0
            ),
        ),
        show_viewer=args.vis,
    )
    env.build()

    cam = env.cameras["cam"]
    for i in range(1000 if not en.is_smoke() else 5):
        env.scene.step()
        cam.set_pose(pos=(i / 100, 0, 2.5))
        cam.render(rgb=True)


if __name__ == "__main__":
    main()
