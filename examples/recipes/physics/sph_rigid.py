"""A tilted rigid cube drops into a column of SPH water via two-way SPH-rigid coupling.

---
tier: recipe
capabilities: [solver.sph, solver.rigid]
hardware: gpu
optional_deps: []
run: uv run python examples/recipes/physics/sph_rigid.py -v
---

Eden port of Genesis' ``examples/coupling/sph_rigid.py``.

SPH (Smoothed-Particle Hydrodynamics) liquid coupled two-way with a rigid body: a
tilted rigid cube is dropped into a tall column of SPH water. Frictionless two-way
coupling is enabled per-entity on the cube's ``RigidMaterialOptions``
(``needs_coup=True``, ``coup_friction=0.0``); the SPH domain comes from
``EnvOptions.sph_options``.

    uv run python examples/recipes/physics/sph_rigid.py -v
"""

import argparse

import genesis as gs

import eden as en
from eden.envs.base import EnvBase
from eden.options import EnvOptions, SceneOptions
from eden.options.entities import BoxOptions, PlaneOptions
from eden.options.materials import RigidMaterialOptions, SPHLiquidMaterialOptions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-v", "--vis", action="store_true", help="Open the interactive viewer")
    args = parser.parse_args()

    en.init(backend=gs.gpu)

    env = EnvBase(
        env_options=EnvOptions(
            num_envs=1,
            sim_dt=1e-2,
            sim_substeps=10,
            sph_options=gs.options.SPHOptions(
                lower_bound=(0.0, -1.0, 0.0),
                upper_bound=(1.0, 1.0, 2.4),
            ),
        ),
        scene_options=SceneOptions(
            plane=PlaneOptions(),
            water=BoxOptions(
                size=(0.9, 1.6, 1.2),
                default_root_pos=(0.5, 0.0, 0.6),
                material=SPHLiquidMaterialOptions(mu=0.01, sampler="regular"),
                # SPH renders as particles; Eden's entity vis_mode defaults to
                # "visual", which the SPH solver rejects, so set it explicitly.
                surface=gs.surfaces.Default(color=(0.5, 0.7, 0.9, 1.0), vis_mode="particle"),
                vis_mode="particle",
            ),
            cube=BoxOptions(
                size=(0.2, 0.2, 0.2),
                default_root_pos=(0.5, 0.0, 2.4),
                default_root_quat=(0.9076734, 0.2432103, 0.3303661, 0.0885213),  # euler (30, 40, 0)
                fixed=False,
                material=RigidMaterialOptions(needs_coup=True, coup_friction=0.0),
            ),
        ),
        show_viewer=args.vis,
    )

    env.build()

    horizon = 500 if not en.is_smoke() else 5
    for _ in range(horizon):
        env.scene.step()


if __name__ == "__main__":
    main()
