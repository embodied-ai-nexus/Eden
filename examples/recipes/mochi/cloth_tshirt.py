"""Mochi benchmark scene: the mochi t-shirt (3593 nodes) falling onto the ground with self-contact.

---
tier: recipe
capabilities: [solver.mochi]
hardware: gpu-fp64
optional_deps: []
run: uv run python examples/recipes/mochi/cloth_tshirt.py -v
---

Eden port of the Genesis fork's ``examples/mochi/cloth_tshirt.py``, the ``cloth_tshirt`` acceptance scene of
the fork's ``tests/mochi/benchmark`` run as a standalone demo, with mochi's own settings for this garment: a
Newton cap of 2 per step and the implicit normal force driving the friction dissipation. One setting differs:
the shell is 1 mm thick, where the fork's is 2 mm, so it drapes more softly. The shell collides
with itself through the spheres placed at its vertices; samples near a vertex in the rest configuration are
excluded, so the sleeves and the body of the shirt stack without passing through each other. The mesh is
``objects/mochi/tshirt_3593.obj`` of the ``Kashu7100/eden_assets`` Hub dataset (see its README).

    uv run python examples/recipes/mochi/cloth_tshirt.py -v
"""

import argparse

import genesis as gs
from _mochi_common import MOCHI_GRAVITY, VideoRecorder, init, newton_iterations, recording_cameras

import eden as en
from eden.engine.solvers.mochi.options import MochiOptions
from eden.envs.base import EnvBase
from eden.options import EnvOptions, SceneOptions
from eden.options.entities import EntityOptions, PlaneOptions
from eden.options.materials import MochiRigidMaterialOptions, MochiShellMaterialOptions
from eden.utils.assets import get_asset_path

CAM_POS, CAM_LOOKAT = (0.7, -1.2, 0.8), (-0.5, 0.0, 0.1)


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
            mochi_options=MochiOptions(
                gravity=MOCHI_GRAVITY, n_newton_iterations=2, implicit_normal_force_for_dissipation=True
            ),
        ),
        scene_options=SceneOptions(
            plane=PlaneOptions(material=MochiRigidMaterialOptions()),
            shirt=EntityOptions(
                file=get_asset_path("objects/mochi/tshirt_3593.obj"),
                default_root_pos=(-0.5, 0.0, 0.1),
                # Half the fork's 2 mm shell, on purpose: bending stiffness goes as thickness cubed and mass
                # per area as thickness, so the shirt folds 8x more easily at half the weight. Every other
                # parameter is the fork's.
                material=MochiShellMaterialOptions(
                    E=1e5, nu=0.25, rho=1000.0, thickness=1e-3, collider_radius=1.5e-2, self_contact=True
                ),
                surface=gs.surfaces.Default(color=(0.85, 0.3, 0.35)),
            ),
        ),
        cameras_options=recording_cameras(args, pos=CAM_POS, lookat=CAM_LOOKAT),
        show_viewer=args.vis,
    )
    env.build()
    shirt = env.entities["shirt"]

    recorder = VideoRecorder(env, args, "cloth_tshirt")
    for i_step in range(steps):
        env.scene.step()
        if i_step % 60 == 0:
            print(
                f"step {i_step:4d}: shirt zmax={float(shirt.get_vertex_positions()[..., 2].max()):.4f} "
                f"newton iterations={newton_iterations(env)}"
            )
        recorder.frame()
    recorder.close()


if __name__ == "__main__":
    main()
