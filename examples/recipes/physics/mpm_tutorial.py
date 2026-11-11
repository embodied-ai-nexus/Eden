"""An elastic box, a liquid box, and an elastoplastic sphere fall onto the ground under the MPM solver.

---
tier: recipe
capabilities: [solver.mpm]
hardware: gpu
optional_deps: []
run: uv run python examples/recipes/physics/mpm_tutorial.py -v
---

Eden port of Genesis' ``examples/tutorials/mpm.py``.

MPM (Material Point Method) multi-material demo: an elastic box, a liquid box, and
an elastoplastic sphere all fall onto the ground plane, each simulated by the MPM
solver with a different constitutive model. The solver domain and resolution are
configured through ``EnvOptions.mpm_options``; the material choice is just the
material option class attached to each entity.

    uv run python examples/recipes/physics/mpm_tutorial.py -v
"""

import argparse

import genesis as gs

import eden as en
from eden.envs.base import EnvBase
from eden.options import EnvOptions, SceneOptions
from eden.options.entities import BoxOptions, PlaneOptions, SphereOptions
from eden.options.materials import (
    MPMElasticMaterialOptions,
    MPMElastoPlasticMaterialOptions,
    MPMLiquidMaterialOptions,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-v", "--vis", action="store_true", help="Open the interactive viewer")
    args = parser.parse_args()

    en.init(backend=gs.gpu)

    env = EnvBase(
        env_options=EnvOptions(
            num_envs=1,
            sim_dt=4e-3,
            sim_substeps=10,
            mpm_options=gs.options.MPMOptions(
                lower_bound=(-0.5, -1.0, 0.0),
                upper_bound=(0.5, 1.0, 1.0),
            ),
        ),
        scene_options=SceneOptions(
            plane=PlaneOptions(),
            elastic=BoxOptions(
                size=(0.2, 0.2, 0.2),
                default_root_pos=(0.0, -0.5, 0.25),
                material=MPMElasticMaterialOptions(),
                surface=gs.surfaces.Default(color=(1.0, 0.4, 0.4), vis_mode="visual"),
                vis_mode="visual",
            ),
            liquid=BoxOptions(
                size=(0.3, 0.3, 0.3),
                default_root_pos=(0.0, 0.0, 0.25),
                material=MPMLiquidMaterialOptions(),
                surface=gs.surfaces.Default(color=(0.3, 0.3, 1.0), vis_mode="particle"),
                vis_mode="particle",
            ),
            plastic=SphereOptions(
                radius=0.1,
                default_root_pos=(0.0, 0.5, 0.35),
                material=MPMElastoPlasticMaterialOptions(),
                surface=gs.surfaces.Default(color=(0.4, 1.0, 0.4), vis_mode="particle"),
                vis_mode="particle",
            ),
        ),
        show_viewer=args.vis,
    )

    env.build()

    horizon = 1000 if not en.is_smoke() else 5
    for _ in range(horizon):
        env.scene.step()


if __name__ == "__main__":
    main()
