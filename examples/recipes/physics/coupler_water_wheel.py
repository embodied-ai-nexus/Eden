"""A liquid jet (SPH or MPM) pours onto a rigid water wheel and spins it via two-way coupling.

---
tier: recipe
capabilities: [solver.sph, solver.mpm, solver.rigid]
hardware: gpu
optional_deps: []
run: uv run python examples/recipes/physics/coupler_water_wheel.py -v
---

Eden port of Genesis' ``examples/coupling/water_wheel.py``.

A liquid jet pours onto a fixed water wheel and spins it (the wheel's base is
pinned; its blades rotate on a joint driven purely by the fluid impact). The fluid
can be simulated with either the SPH or the MPM liquid solver via ``--solver``.

The jet is produced by a Genesis *emitter*, which Eden does not yet wrap — see the
README "Known gaps". The emitter is created through ``env.scene.add_emitter(...)``
after ``EnvBase(...)`` and before ``env.build()``.

    uv run python examples/recipes/physics/coupler_water_wheel.py -v
    uv run python examples/recipes/physics/coupler_water_wheel.py --solver mpm -v
"""

import os
import argparse

import numpy as np
import genesis as gs

import eden as en
from eden.envs.base import EnvBase
from eden.options import EnvOptions, SceneOptions
from eden.options.entities import EntityOptions, PlaneOptions
from eden.options.materials import MPMLiquidMaterialOptions, SPHLiquidMaterialOptions
from eden.utils.misc import get_gs_assets_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver", type=str, default="sph", choices=("sph", "mpm"), help="Solver variant to run")
    parser.add_argument("-v", "--vis", action="store_true", help="Open the interactive viewer")
    args = parser.parse_args()

    en.init(backend=gs.gpu)

    env = EnvBase(
        env_options=EnvOptions(
            num_envs=1,
            sim_dt=4e-3,
            sim_substeps=10,
            mpm_options=gs.options.MPMOptions(lower_bound=(0.0, -1.5, 0.0), upper_bound=(1.0, 1.5, 4.0)),
            sph_options=gs.options.SPHOptions(particle_size=0.02),
        ),
        scene_options=SceneOptions(
            plane=PlaneOptions(),
            wheel=EntityOptions(
                file=os.path.join(get_gs_assets_dir(), "urdf/wheel/fancy_wheel.urdf"),
                default_root_pos=(0.5, 0.25, 1.6),
                is_fixed_base=True,
                convexify=False,
            ),
        ),
        show_viewer=args.vis,
    )

    # --- raw-Genesis emitter (no Eden wrapper yet); add before build ---
    liquid = SPHLiquidMaterialOptions if args.solver == "sph" else MPMLiquidMaterialOptions
    emitter = env.scene.add_emitter(
        material=liquid(sampler="regular").to_genesis_material(),
        max_particles=100000,
        surface=gs.surfaces.Glass(color=(0.7, 0.85, 1.0, 0.7), vis_mode="particle"),
    )

    env.build()

    horizon = 500 if not en.is_smoke() else 5
    for _ in range(horizon):
        emitter.emit(
            pos=np.array([0.5, 1.0, 3.5]),
            direction=np.array([0.0, 0.0, -1.0]),
            speed=5.0,
            droplet_shape="circle",
            droplet_size=0.22,
        )
        env.scene.step()


if __name__ == "__main__":
    main()
