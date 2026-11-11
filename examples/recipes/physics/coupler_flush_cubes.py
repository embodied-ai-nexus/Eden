"""Two MPM-liquid jets flush three MPM-elastic solids across the floor in one all-MPM scene.

---
tier: recipe
capabilities: [solver.mpm]
hardware: gpu
optional_deps: []
run: uv run python examples/recipes/physics/coupler_flush_cubes.py -v
---

Eden port of Genesis' ``examples/coupling/flush_cubes.py``.

Two jets of MPM liquid flush three MPM-elastic solids (a box, a sphere, and a
cylinder) across the floor — an all-MPM scene where fluid and deformable solids
share the same grid.

The two jets are Genesis *emitters*, which Eden does not yet wrap — see the README
"Known gaps". They are created via ``env.scene.add_emitter(...)`` after
``EnvBase(...)`` and before ``env.build()``.

Note: each emitter is allowed up to 800k particles, so the full run is heavy; drop
``max_particles`` or the horizon if you are memory-constrained.

    uv run python examples/recipes/physics/coupler_flush_cubes.py -v
"""

import argparse

import numpy as np
import genesis as gs

import eden as en
from eden.envs.base import EnvBase
from eden.options import EnvOptions, SceneOptions
from eden.options.entities import BoxOptions, CylinderOptions, PlaneOptions, SphereOptions
from eden.options.materials import MPMElasticMaterialOptions, MPMLiquidMaterialOptions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-v", "--vis", action="store_true", help="Open the interactive viewer")
    args = parser.parse_args()

    en.init(backend=gs.gpu)

    def solid(color):
        return MPMElasticMaterialOptions(rho=400.0), gs.surfaces.Rough(color=color, vis_mode="particle")

    box_mat, box_surf = solid((1.0, 0.5, 0.5, 1.0))
    sph_mat, sph_surf = solid((1.0, 1.0, 0.5, 1.0))
    cyl_mat, cyl_surf = solid((0.5, 1.0, 1.0, 1.0))

    env = EnvBase(
        env_options=EnvOptions(
            num_envs=1,
            sim_dt=4e-3,
            sim_substeps=20,
            mpm_options=gs.options.MPMOptions(
                lower_bound=(-0.45, -0.65, -0.01),
                upper_bound=(0.45, 0.65, 1.0),
                grid_density=64,
            ),
        ),
        scene_options=SceneOptions(
            plane=PlaneOptions(),
            box=BoxOptions(
                size=(0.12, 0.12, 0.12),
                default_root_pos=(0.0, 0.25, 0.4),
                material=box_mat,
                surface=box_surf,
                vis_mode="particle",
            ),
            ball=SphereOptions(
                radius=0.06,
                default_root_pos=(0.15, 0.45, 0.5),
                material=sph_mat,
                surface=sph_surf,
                vis_mode="particle",
            ),
            cyl=CylinderOptions(
                radius=0.05,
                height=0.14,
                default_root_pos=(-0.15, 0.45, 0.6),
                material=cyl_mat,
                surface=cyl_surf,
                vis_mode="particle",
            ),
        ),
        show_viewer=args.vis,
    )

    # --- raw-Genesis emitters (no Eden wrapper yet); add before build ---
    emitter1 = env.scene.add_emitter(
        material=MPMLiquidMaterialOptions(sampler="random").to_genesis_material(),
        max_particles=800000,
        surface=gs.surfaces.Rough(color=(0.0, 0.9, 0.4, 1.0)),
    )
    emitter2 = env.scene.add_emitter(
        material=MPMLiquidMaterialOptions(sampler="random").to_genesis_material(),
        max_particles=800000,
        surface=gs.surfaces.Rough(color=(0.0, 0.4, 0.9, 1.0)),
    )

    env.build()

    horizon = 100 if not en.is_smoke() else 5
    for _ in range(horizon):
        emitter1.emit(
            pos=np.array([0.16, -0.4, 0.5]),
            direction=np.array([0.0, 0.0, -1.0]),
            speed=1.5,
            droplet_shape="circle",
            droplet_size=0.16,
        )
        emitter2.emit(
            pos=np.array([-0.16, -0.4, 0.5]),
            direction=np.array([0.0, 0.0, -1.0]),
            speed=1.5,
            droplet_shape="circle",
            droplet_size=0.16,
        )
        env.scene.step()


if __name__ == "__main__":
    main()
