"""Mochi benchmark scene: a helical spring (129 nodes) hanging from its first node, ringing under gravity.

---
tier: recipe
capabilities: [solver.mochi]
hardware: gpu-fp64
optional_deps: []
run: uv run python examples/recipes/mochi/rod_helix.py -v
---

Eden port of the Genesis fork's ``examples/mochi/rod_helix.py``, the ``rod_helix`` acceptance scene of the fork's
``tests/mochi/benchmark`` run as a standalone demo. The centerline is the helix asset of the original mochi engine
(``objects/mochi/helix_129.npy`` of the ``Kashu7100/eden_assets`` Hub dataset). The discrete elastic rod carries
stretching, bending and twisting (one twist angle per segment as an extra unknown); the stiffnesses are given
explicitly here because the benchmark sets ``E`` and ``G`` independently (1 GPa each), which the material's own
``nu`` cannot express. Contact is disabled: the spring never touches anything, it only rings on its own
elasticity.

    uv run python examples/recipes/mochi/rod_helix.py -v
"""

import argparse

import genesis as gs
import numpy as np
from _mochi_common import (
    MOCHI_GRAVITY,
    VideoRecorder,
    init,
    newton_iterations,
    recording_cameras,
    rod_section,
)

import eden as en
from eden.engine.solvers.mochi.options import MochiOptions
from eden.envs.base import EnvBase
from eden.options import EnvOptions, RodOptions, SceneOptions
from eden.options.entities import PlaneOptions
from eden.options.materials import MochiRigidMaterialOptions, MochiRodMaterialOptions
from eden.utils.assets import get_asset_path

CAM_POS, CAM_LOOKAT = (1.0, -1.0, 1.0), (0.0, 0.0, 0.7)
ROD_RADIUS, ROD_E, ROD_G, ROD_RHO = 1e-2, 1e9, 1e9, 1000.0


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
            mochi_options=MochiOptions(gravity=MOCHI_GRAVITY, n_newton_iterations=20),
        ),
        scene_options=SceneOptions(
            plane=PlaneOptions(material=MochiRigidMaterialOptions()),
            helix=RodOptions(
                points=np.load(get_asset_path("objects/mochi/helix_129.npy")),
                radius=ROD_RADIUS,
                default_root_pos=(0.0, 0.0, 1.0),
                material=MochiRodMaterialOptions(
                    collider_type="none",
                    **rod_section(ROD_RADIUS, E=ROD_E, G=ROD_G, rho=ROD_RHO),
                ),
                surface=gs.surfaces.Default(color=(0.7, 0.7, 0.75)),
            ),
        ),
        cameras_options=recording_cameras(args, pos=CAM_POS, lookat=CAM_LOOKAT),
        show_viewer=args.vis,
    )
    env.build()
    helix = env.entities["helix"]
    helix.set_vertices_fixed([0])

    recorder = VideoRecorder(env, args, "rod_helix")
    for i_step in range(steps):
        env.scene.step()
        if i_step % 60 == 0:
            print(
                f"step {i_step:4d}: tip z={float(helix.get_vertex_positions()[0, -1, 2]):.4f} "
                f"newton iterations={newton_iterations(env)}"
            )
        recorder.frame()
    recorder.close()


if __name__ == "__main__":
    main()
