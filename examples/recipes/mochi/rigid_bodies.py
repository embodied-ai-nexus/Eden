"""Mochi rigid bodies: a sphere and a cube dropped onto a table standing on the ground.

---
tier: recipe
capabilities: [solver.mochi]
hardware: gpu-fp64
optional_deps: []
run: uv run python examples/recipes/mochi/rigid_bodies.py -v
---

Eden port of the Genesis fork's ``examples/mochi/rigid_bodies.py``. The MochiSolver is a fully implicit
solver with smooth penalty contact, so a 16.7 ms step is stable. Contact between the bodies uses the
analytic sphere/box distance fields of the primitives and the signed-distance grid of the table
(``collider_type="sdf"``). Every entity carries a Mochi material: a Mochi scene holds Mochi entities only.

    uv run python examples/recipes/mochi/rigid_bodies.py -v
    uv run python examples/recipes/mochi/rigid_bodies.py -c -r -o renders/mochi   # CPU, record a video
"""

import argparse

import genesis as gs
import numpy as np
from _mochi_common import MOCHI_GRAVITY, VideoRecorder, init, newton_iterations, recording_cameras

import eden as en
from eden.engine.solvers.mochi.options import MochiOptions
from eden.envs.base import EnvBase
from eden.options import EnvOptions, SceneOptions
from eden.options.entities import BoxOptions, PlaneOptions, SphereOptions
from eden.options.materials import MochiRigidMaterialOptions

CAM_POS, CAM_LOOKAT = (3.0, -3.0, 2.0), (0.0, 0.0, 0.5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-v", "--vis", action="store_true", help="Open the interactive viewer")
    parser.add_argument("-c", "--cpu", action="store_true", help="Run on the CPU backend instead of the GPU")
    parser.add_argument("-s", "--steps", type=int, default=300, help="Number of simulation steps")
    parser.add_argument("-r", "--record", action="store_true", help="Record a video of the run")
    parser.add_argument(
        "-o", "--output-dir", type=str, default="renders/mochi", help="Directory the recorded video is written to"
    )
    args = parser.parse_args()
    init(args)
    steps = 5 if en.is_smoke() else args.steps

    env = EnvBase(
        env_options=EnvOptions(
            num_envs=1,
            sim_dt=1.0 / 60.0,
            mochi_options=MochiOptions(gravity=MOCHI_GRAVITY, n_newton_iterations=8),
        ),
        scene_options=SceneOptions(
            plane=PlaneOptions(material=MochiRigidMaterialOptions()),
            table=BoxOptions(
                size=(1.6, 1.0, 0.1),
                default_root_pos=(0.0, 0.0, 0.75),
                is_fixed_base=True,
                material=MochiRigidMaterialOptions(collider_type="sdf"),
            ),
            sphere=SphereOptions(radius=0.2, default_root_pos=(-0.5, 0.0, 1.5), material=MochiRigidMaterialOptions()),
            cube=BoxOptions(
                size=(0.4, 0.4, 0.4),
                default_root_pos=(0.5, 0.1, 1.6),
                default_root_quat=tuple(gs.utils.geom.xyz_to_quat(np.array([20.0, 30.0, 0.0]), degrees=True).tolist()),
                material=MochiRigidMaterialOptions(),
            ),
        ),
        cameras_options=recording_cameras(args, pos=CAM_POS, lookat=CAM_LOOKAT),
        show_viewer=args.vis,
    )
    env.build()
    sphere, cube = env.entities["sphere"], env.entities["cube"]

    recorder = VideoRecorder(env, args, "rigid_bodies")
    for i_step in range(steps):
        env.scene.step()
        if i_step % 60 == 0:
            print(
                f"step {i_step:4d}: sphere z={float(sphere.get_pos()[0, 2]):.4f} cube z={float(cube.get_pos()[0, 2]):.4f} "
                f"newton iterations={newton_iterations(env)}"
            )
        recorder.frame()
    recorder.close()


if __name__ == "__main__":
    main()
