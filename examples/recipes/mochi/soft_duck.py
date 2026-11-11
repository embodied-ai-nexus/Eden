"""Mochi benchmark scene: the mochi duck (1899 nodes, 8608 tetrahedra) dropped onto the ground.

---
tier: recipe
capabilities: [solver.mochi]
hardware: gpu-fp64
optional_deps: []
run: uv run python examples/recipes/mochi/soft_duck.py -v
---

Eden port of the Genesis fork's ``examples/mochi/soft_duck.py``, the ``soft_duck`` acceptance scene of the
fork's ``tests/mochi/benchmark`` run as a standalone demo. The mesh is the duck asset of the original mochi
engine, converted to a tetgen ``.node`` / ``.ele`` pair (``objects/mochi`` of the ``Kashu7100/eden_assets`` Hub
dataset, see its README; :func:`get_asset_path` fetches the ``.ele`` with the ``.node``). The
finite elements and the penalty contact against the plane enter one implicit Newton solve, so the 1/60 s
step stays stable through the impact.

Eden reads a tetrahedral simulation mesh straight from the ``.node`` file (its ``.ele`` sibling is opened
alongside); only a ``MochiElasticMaterialOptions`` entity may carry one.

    uv run python examples/recipes/mochi/soft_duck.py -v
"""

import argparse

import genesis as gs
from _mochi_common import MOCHI_GRAVITY, VideoRecorder, init, newton_iterations, recording_cameras

import eden as en
from eden.engine.solvers.mochi.options import MochiOptions
from eden.envs.base import EnvBase
from eden.options import EnvOptions, SceneOptions
from eden.options.entities import EntityOptions, PlaneOptions
from eden.options.materials import MochiElasticMaterialOptions, MochiRigidMaterialOptions
from eden.utils.assets import get_asset_path

CAM_POS, CAM_LOOKAT = (1.2, -0.7, 1.4), (-0.5, 1.0, 0.3)


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
            duck=EntityOptions(
                file=get_asset_path("objects/mochi/duck_1899.node"),
                default_root_pos=(-0.5, 1.0, 1.0),
                material=MochiElasticMaterialOptions(E=1e5, nu=0.45, rho=1000.0),
                surface=gs.surfaces.Default(color=(0.95, 0.8, 0.2)),
            ),
        ),
        cameras_options=recording_cameras(args, pos=CAM_POS, lookat=CAM_LOOKAT),
        show_viewer=args.vis,
    )
    env.build()
    duck = env.entities["duck"]

    recorder = VideoRecorder(env, args, "soft_duck")
    for i_step in range(steps):
        env.scene.step()
        if i_step % 60 == 0:
            print(
                f"step {i_step:4d}: duck zmin={float(duck.get_vertex_positions()[..., 2].min()):.4f} "
                f"newton iterations={newton_iterations(env)}"
            )
        recorder.frame()
    recorder.close()


if __name__ == "__main__":
    main()
