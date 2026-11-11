"""A camera declaratively follows a free-flying cube, keeping it centred each frame.

---
tier: recipe
capabilities: [render.camera]
hardware: gpu
optional_deps: []
run: uv run python examples/recipes/rendering/follow_entity.py
---

Eden port of Genesis' ``examples/rendering/follow_entity.py``.

A free (non-fixed) box is given an initial linear + angular velocity so it flies
across the scene, and a camera tracks it via the declarative
``CameraOptions(follow_entity_name="cube")``. Each step re-renders RGB, so the
cube stays centred in frame.

Note: Eden's follow is declarative and always keeps a fixed orientation
(``fix_orientation=True`` is hardcoded in the ``Camera`` wrapper). There is no
per-call ``follow()``/no ``--fix`` flag as in the Genesis original — the follow
target is set once through ``cameras_options``.

    uv run python examples/recipes/rendering/follow_entity.py
"""

import argparse

import genesis as gs

import eden as en
from eden.envs.base import EnvBase
from eden.options import EnvOptions, SceneOptions
from eden.options.camera import CameraOptions, CamerasOptions
from eden.options.entities import BoxOptions, PlaneOptions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-v", "--vis", action="store_true", help="Open the interactive viewer")
    args = parser.parse_args()

    en.init(backend=gs.gpu)

    env = EnvBase(
        env_options=EnvOptions(num_envs=1),
        scene_options=SceneOptions(
            plane=PlaneOptions(),
            cube=BoxOptions(size=(0.1, 0.1, 0.1), default_root_pos=(0.0, -0.9, 1.0)),
        ),
        cameras_options=CamerasOptions(
            cam=CameraOptions(
                cam_pos=(2.0, 0.0, 1.5),
                cam_lookat=(0.0, 0.0, 0.7),
                cam_resolution=(640, 480),
                cam_fov=40.0,
                follow_entity_name="cube",
            ),
        ),
        show_viewer=args.vis,
    )
    env.build()

    # Send the free body flying: 6 DOFs (linear xyz, angular xyz) on the Genesis
    # free-joint root, mirroring the original's ``cube.set_dofs_velocity(...)``.
    env.entities["cube"].entity.set_dofs_velocity([0.0, 5.0, 0.0, 0.0, 0.0, 1.0])

    cam = env.cameras["cam"]
    for _ in range(100 if not en.is_smoke() else 5):
        env.scene.step()
        cam.render_rgb()


if __name__ == "__main__":
    main()
