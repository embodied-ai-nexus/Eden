"""An MPM-elastic dragon is sliced by a rigid cross-cutter via two-way MPM-rigid coupling.

---
tier: recipe
capabilities: [solver.mpm, solver.rigid]
hardware: gpu
optional_deps: []
run: uv run python examples/recipes/physics/coupler_cut_dragon.py -v
---

Eden port of Genesis' ``examples/coupling/cut_dragon.py``. A fixed rigid cross-cutter
(``enable_CPIC=True`` on the MPM solver) slices a deformable dragon that falls onto it,
demonstrating cutting through continuous-particle interface conditions.

    uv run python examples/recipes/physics/coupler_cut_dragon.py -v
"""

import os
import argparse

import genesis as gs

import eden as en
from eden.envs.base import EnvBase
from eden.options import EnvOptions, SceneOptions
from eden.options.entities import EntityOptions, PlaneOptions
from eden.options.materials import MPMElasticMaterialOptions
from eden.utils.misc import get_gs_assets_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", help="Open the interactive viewer")
    args = parser.parse_args()

    en.init(backend=gs.gpu)

    env = EnvBase(
        env_options=EnvOptions(
            num_envs=1,
            sim_dt=4e-3,
            sim_substeps=10,
            mpm_options=gs.options.MPMOptions(
                lower_bound=(-0.5, -0.5, -0.05),
                upper_bound=(0.5, 0.5, 2.0),
                grid_density=64,
                enable_CPIC=True,
            ),
        ),
        scene_options=SceneOptions(
            plane=PlaneOptions(),
            cutter=EntityOptions(
                file=os.path.join(get_gs_assets_dir(), "meshes/cross_cutter.obj"),
                scale=0.4,
                default_root_pos=(0.0, 0.0, 0.15),
                default_root_quat=(0.7071068, 0.7071068, 0.0, 0.0),  # euler (90, 0, 0)
                is_fixed_base=True,
                convexify=False,
                surface=gs.surfaces.Iron(),
            ),
            dragon=EntityOptions(
                file=os.path.join(get_gs_assets_dir(), "meshes/dragon/dragon.obj"),
                scale=0.003,
                default_root_quat=(0.7071068, 0.7071068, 0.0, 0.0),  # euler (90, 0, 0)
                default_root_pos=(0.0, 0.0, 0.7),
                material=MPMElasticMaterialOptions(
                    sampler="pbs-64",
                ),
                surface=gs.surfaces.Rough(
                    color=(0.6, 1.0, 0.8, 1.0),
                    vis_mode="particle",
                ),
                vis_mode="particle",
            ),
        ),
        show_viewer=args.vis,
    )

    env.build()

    horizon = 800 if not en.is_smoke() else 5
    for _ in range(horizon):
        env.scene.step()

    env.reset()
    horizon = 800 if not en.is_smoke() else 5
    for _ in range(horizon):
        env.scene.step()


if __name__ == "__main__":
    main()
