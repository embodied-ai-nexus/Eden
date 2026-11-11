"""Benchmark camera rendering throughput (RGB + depth) and print frames per second.

---
tier: recipe
capabilities: [render.camera]
hardware: gpu
optional_deps: []
run: uv run python examples/recipes/rendering/speed_test.py
---

Eden port of Genesis' ``examples/rendering/speed_test.py``.

A minimal plane + box scene with one camera. It times a tight loop of
``render(rgb=True, depth=True)`` calls (no physics stepping) and reports the
achieved rendering FPS — a quick throughput sanity check for the rasterizer
backend on the current GPU.

    uv run python examples/recipes/rendering/speed_test.py
"""

import argparse
from time import time

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
        env_options=EnvOptions(num_envs=1, sim_dt=0.01),
        scene_options=SceneOptions(
            plane=PlaneOptions(),
            box=BoxOptions(size=(0.5, 0.5, 0.5), default_root_pos=(0.0, 0.0, 0.25)),
        ),
        cameras_options=CamerasOptions(
            cam=CameraOptions(
                cam_pos=(3.5, 0.0, 2.5), cam_lookat=(0.0, 0.0, 0.5), cam_resolution=(640, 480), cam_fov=30.0
            ),
        ),
        show_viewer=args.vis,
    )
    env.build()

    cam = env.cameras["cam"]
    n = 2000 if not en.is_smoke() else 5
    t = time()
    for _ in range(n):
        cam.render(rgb=True, depth=True)
    print(f"{n / (time() - t):.1f} FPS")


if __name__ == "__main__":
    main()
