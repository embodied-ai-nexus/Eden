"""A block of PBD liquid drops and settles under gravity inside the PBD solver domain.

---
tier: recipe
capabilities: [solver.pbd]
hardware: gpu
optional_deps: []
run: uv run python examples/recipes/physics/pbd_liquid.py -v
---

Eden port of Genesis' ``examples/pbd_liquid.py``.

A block of PBD (Position-Based Dynamics) liquid is dropped and settles under
gravity inside the PBD solver domain. The simulation domain and solver iteration
counts are configured through ``EnvOptions.pbd_options``; the fluid material is a
:class:`PBDLiquidMaterialOptions`. Position is baked into the box morph at
creation time (PBD/particle entities do not support ``set_pos``).

Run with ``-v`` to open the viewer:

    uv run python examples/recipes/physics/pbd_liquid.py -v
"""

import argparse

import genesis as gs

import eden as en
from eden.envs.base import EnvBase
from eden.options import EnvOptions, SceneOptions
from eden.options.entities import BoxOptions
from eden.options.materials import PBDLiquidMaterialOptions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", help="Open the interactive viewer")
    args = parser.parse_args()

    en.init(backend=gs.gpu)

    env = EnvBase(
        env_options=EnvOptions(
            num_envs=1,
            sim_dt=2e-3,
            sim_substeps=1,
            pbd_options=gs.options.PBDOptions(
                lower_bound=(0.0, 0.0, 0.0),
                upper_bound=(1.0, 1.0, 1.0),
                max_density_solver_iterations=10,
                max_viscosity_solver_iterations=1,
            ),
        ),
        scene_options=SceneOptions(
            # Box morph filled with fluid particles. ``default_root_pos`` is the
            # box center; ``size`` matches the lower/upper extents (0.2, 0.2, 0.4)
            # of the original Genesis example.
            liquid=BoxOptions(
                default_root_pos=(0.3, 0.2, 0.3),
                size=(0.2, 0.2, 0.4),
                material=PBDLiquidMaterialOptions(
                    sampler="regular",
                    rho=1.0,
                    density_relaxation=1.0,
                    viscosity_relaxation=0.0,
                ),
                surface=gs.surfaces.Default(
                    color=(0.2, 0.6, 1.0, 1.0),
                    vis_mode="particle",
                ),
                vis_mode="particle",
            ),
        ),
        show_viewer=args.vis,
    )

    env.build()

    horizon = 4000 if not en.is_smoke() else 5
    for _ in range(horizon):
        env.scene.step()


if __name__ == "__main__":
    main()
