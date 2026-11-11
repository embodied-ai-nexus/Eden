"""Mochi deformable bodies: soft spheres and a soft cube dropped on the ground, a rigid box landing on a soft slab.

---
tier: recipe
capabilities: [solver.mochi]
hardware: gpu-fp64
optional_deps: []
run: uv run python examples/recipes/mochi/soft_bodies.py -v
---

Eden port of the Genesis fork's ``examples/mochi/soft_bodies.py``. Rigid and deformable bodies share one
implicit Newton solve: the tetrahedral finite elements (stable neo-Hookean) and the penalty contact against
the rigid colliders enter the same residual and Hessian, so no coupler is involved. Contact acts through
samples of the deformable surface, so the soft bodies are meshed finely enough for what lands on them: the
tetgen knobs go through the primitive's ``morph=`` override (``nobisect=False`` lets tetgen refine the surface).

    uv run python examples/recipes/mochi/soft_bodies.py -v
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
from eden.options.materials import MochiElasticMaterialOptions, MochiRigidMaterialOptions

CAM_POS, CAM_LOOKAT = (2.5, -2.5, 1.5), (0.0, 0.0, 0.3)


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

    rubber = MochiElasticMaterialOptions(E=5e4, nu=0.45, rho=1000.0)
    env = EnvBase(
        env_options=EnvOptions(
            num_envs=1,
            sim_dt=1.0 / 60.0,
            mochi_options=MochiOptions(gravity=MOCHI_GRAVITY, n_newton_iterations=8),
        ),
        scene_options=SceneOptions(
            plane=PlaneOptions(material=MochiRigidMaterialOptions()),
            sphere=SphereOptions(
                radius=0.15,
                default_root_pos=(-0.6, 0.0, 0.6),
                material=rubber,
                morph={"maxvolume": 0.0005, "nobisect": False},
                surface=gs.surfaces.Default(color=(0.9, 0.5, 0.2)),
            ),
            sphere2=SphereOptions(
                radius=0.1,
                default_root_pos=(-0.6, 0.0, 1.0),
                material=rubber,
                morph={"maxvolume": 0.0005, "nobisect": False},
                surface=gs.surfaces.Default(color=(0.9, 0.5, 0.2)),
            ),
            cube=BoxOptions(
                size=(0.3, 0.3, 0.3),
                default_root_pos=(0.0, 0.0, 0.7),
                default_root_quat=tuple(gs.utils.geom.xyz_to_quat(np.array([20.0, 30.0, 0.0]), degrees=True).tolist()),
                material=MochiElasticMaterialOptions(E=2e4, nu=0.4, rho=800.0, stiffness_damping=1e-3),
                morph={"maxvolume": 0.001},
                surface=gs.surfaces.Default(color=(0.3, 0.7, 0.9)),
            ),
            slab=BoxOptions(
                size=(0.6, 0.6, 0.1),
                default_root_pos=(0.8, 0.0, 0.05),
                material=MochiElasticMaterialOptions(E=1e5, nu=0.45, rho=1000.0),
                morph={"maxvolume": 0.0002, "nobisect": False},
                surface=gs.surfaces.Default(color=(0.5, 0.9, 0.5)),
            ),
            box=BoxOptions(
                size=(0.15, 0.15, 0.15),
                default_root_pos=(0.8, 0.0, 0.5),
                material=MochiRigidMaterialOptions(rho=1500.0),
            ),
        ),
        cameras_options=recording_cameras(args, pos=CAM_POS, lookat=CAM_LOOKAT),
        show_viewer=args.vis,
    )
    env.build()
    sphere, cube, slab, box = (env.entities[k] for k in ("sphere", "cube", "slab", "box"))

    recorder = VideoRecorder(env, args, "soft_bodies")
    for i_step in range(steps):
        env.scene.step()
        if i_step % 60 == 0:
            print(
                f"step {i_step:4d}: sphere zmin={float(sphere.get_vertex_positions()[..., 2].min()):.4f} "
                f"cube zmin={float(cube.get_vertex_positions()[..., 2].min()):.4f} "
                f"slab zmax={float(slab.get_vertex_positions()[..., 2].max()):.4f} "
                f"box z={float(box.get_pos()[0, 2]):.4f} newton iterations={newton_iterations(env)}"
            )
        recorder.frame()
    recorder.close()


if __name__ == "__main__":
    main()
