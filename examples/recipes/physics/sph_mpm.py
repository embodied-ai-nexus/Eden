"""An MPM-elastic rubber duck drops into a pool of SPH water -- coupled SPH fluid and MPM solid.

---
tier: recipe
capabilities: [solver.sph, solver.mpm]
hardware: gpu
optional_deps: []
run: uv run python examples/recipes/physics/sph_mpm.py -v
---

Eden port of Genesis' ``examples/coupling/sph_mpm.py``.

SPH fluid and an MPM solid sharing one scene: an MPM-elastic rubber duck drops into
a pool of SPH water. The two solvers interact through their overlapping domains,
each configured on ``EnvOptions`` (``sph_options`` / ``mpm_options``). No explicit
coupler is needed — the fluid and the deformable solid are simulated by their own
solvers and exchange forces where they overlap.

    uv run python examples/recipes/physics/sph_mpm.py -v
"""

import os
import argparse

import genesis as gs

import eden as en
from eden.envs.base import EnvBase
from eden.options import EnvOptions, SceneOptions
from eden.options.entities import BoxOptions, EntityOptions
from eden.options.materials import MPMElasticMaterialOptions, SPHLiquidMaterialOptions
from eden.utils.misc import get_gs_assets_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-v", "--vis", action="store_true", help="Open the interactive viewer")
    args = parser.parse_args()

    en.init(backend=gs.gpu)

    env = EnvBase(
        env_options=EnvOptions(
            num_envs=1,
            sim_dt=2e-3,
            sim_substeps=10,
            mpm_options=gs.options.MPMOptions(
                lower_bound=(0.0, 0.0, -0.1),
                upper_bound=(1.0, 1.0, 1.0),
            ),
            sph_options=gs.options.SPHOptions(
                lower_bound=(-0.03, -0.03, -0.08),
                upper_bound=(1.03, 1.03, 1.0),
            ),
        ),
        scene_options=SceneOptions(
            water=BoxOptions(
                size=(0.7, 0.9, 0.5),
                default_root_pos=(0.4, 0.5, 0.25),
                material=SPHLiquidMaterialOptions(),
                surface=gs.surfaces.Default(color=(0.2, 0.6, 1.0, 1.0), vis_mode="particle"),
                vis_mode="particle",
            ),
            duck=EntityOptions(
                file=os.path.join(get_gs_assets_dir(), "meshes/duck.obj"),
                scale=0.07,
                default_root_pos=(0.5, 0.5, 0.7),
                default_root_quat=(0.5, 0.5, -0.5, 0.5),  # euler (90, 0, 90)
                material=MPMElasticMaterialOptions(rho=200.0),
                surface=gs.surfaces.Default(color=(0.9, 0.8, 0.2, 1.0), vis_mode="particle"),
                vis_mode="particle",
            ),
        ),
        show_viewer=args.vis,
    )

    env.build()

    horizon = 800 if not en.is_smoke() else 5
    for _ in range(horizon):
        env.scene.step()


if __name__ == "__main__":
    main()
