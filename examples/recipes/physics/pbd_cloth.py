"""Two PBD cloths drop over a plane, pinned at four corners and at a single corner respectively.

---
tier: recipe
capabilities: [solver.pbd]
hardware: gpu
optional_deps: []
run: uv run python examples/recipes/physics/pbd_cloth.py -v
---

Eden port of Genesis' ``examples/tutorials/pbd_cloth.py``.

Two square PBD (Position-Based Dynamics) cloths are dropped over a plane. After
build, corner particles are pinned using the underlying Genesis entity's
``find_closest_particle`` / ``fix_particles`` (exposed via ``entity.entity``):
the first cloth is fixed at all four corners (taut sheet), the second at a single
corner (hanging sheet).

Cloth pose is baked into the mesh morph at creation time — PBD/particle entities
do not support ``set_pos`` / ``set_quat``. The PBD solver iteration counts come
from ``EnvOptions.pbd_options``.

Run with ``-v`` to open the viewer:

    uv run python examples/recipes/physics/pbd_cloth.py -v
"""

import os
import argparse

import genesis as gs

import eden as en
from eden.envs.base import EnvBase
from eden.options import EnvOptions, SceneOptions
from eden.options.entities import EntityOptions, PlaneOptions
from eden.options.materials import PBDClothMaterialOptions
from eden.utils.misc import get_gs_assets_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", help="Open the interactive viewer")
    args = parser.parse_args()

    en.init(backend=gs.gpu)

    cloth_file = os.path.join(get_gs_assets_dir(), "meshes/cloth.obj")

    env = EnvBase(
        env_options=EnvOptions(
            num_envs=1,
            sim_dt=4e-3,
            sim_substeps=10,
            pbd_options=gs.options.PBDOptions(
                max_stretch_solver_iterations=4,
                max_bending_solver_iterations=1,
            ),
        ),
        scene_options=SceneOptions(
            plane=PlaneOptions(),
            cloth_1=EntityOptions(
                file=cloth_file,
                scale=2.0,
                default_root_pos=(0.0, 0.0, 0.5),
                material=PBDClothMaterialOptions(),
                surface=gs.surfaces.Default(
                    color=(0.2, 0.4, 0.8, 1.0),
                    vis_mode="visual",
                ),
                vis_mode="visual",
                # cloth is a thin shell: keep the mesh as-is for the PBD solver
                convexify=False,
                decimate=False,
            ),
            cloth_2=EntityOptions(
                file=cloth_file,
                scale=2.0,
                default_root_pos=(0.0, 0.0, 1.0),
                material=PBDClothMaterialOptions(),
                surface=gs.surfaces.Default(
                    color=(0.8, 0.4, 0.2, 1.0),
                    vis_mode="particle",
                ),
                vis_mode="particle",
                convexify=False,
                decimate=False,
            ),
        ),
        show_viewer=args.vis,
    )

    env.build()

    # Pin cloth particles via the underlying Genesis PBD entity. ``fix_particles``
    # and ``find_closest_particle`` are Genesis-level APIs reached through
    # ``Entity.entity`` (Eden's ParticleEntity wrapper does not re-expose them).
    cloth_1 = env.entities["cloth_1"].entity
    cloth_1.fix_particles(cloth_1.find_closest_particle((-1, -1, 1.0)))
    cloth_1.fix_particles(cloth_1.find_closest_particle((1, 1, 1.0)))
    cloth_1.fix_particles(cloth_1.find_closest_particle((-1, 1, 1.0)))
    cloth_1.fix_particles(cloth_1.find_closest_particle((1, -1, 1.0)))

    cloth_2 = env.entities["cloth_2"].entity
    cloth_2.fix_particles(cloth_2.find_closest_particle((-1, -1, 1.0)))

    horizon = 1000 if not en.is_smoke() else 5
    for _ in range(horizon):
        env.scene.step()


if __name__ == "__main__":
    main()
