"""A PBD cloth sheet drapes over a fixed rigid sphere via two-way PBD-rigid coupling.

---
tier: recipe
capabilities: [solver.pbd, solver.rigid]
hardware: gpu
optional_deps: []
run: uv run python examples/recipes/physics/coupler_cloth_on_rigid.py -v
---

Eden port of Genesis' ``examples/coupling/cloth_on_rigid.py``.

A PBD cloth sheet drapes over a fixed rigid sphere. The rigid plane and sphere opt
into two-way coupling (``needs_coup=True``) with zero friction so the cloth slides
freely as it settles; the PBD particle resolution comes from
``EnvOptions.pbd_options``.

    uv run python examples/recipes/physics/coupler_cloth_on_rigid.py -v
"""

import os
import argparse

import genesis as gs

import eden as en
from eden.envs.base import EnvBase
from eden.options import EnvOptions, SceneOptions
from eden.options.entities import EntityOptions, PlaneOptions, SphereOptions
from eden.options.materials import PBDClothMaterialOptions, RigidMaterialOptions
from eden.utils.misc import get_gs_assets_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-v", "--vis", action="store_true", help="Open the interactive viewer")
    args = parser.parse_args()

    en.init(backend=gs.gpu)

    frictionless = RigidMaterialOptions(needs_coup=True, coup_friction=0.0)

    env = EnvBase(
        env_options=EnvOptions(
            num_envs=1,
            sim_dt=2e-3,
            sim_substeps=10,
            pbd_options=gs.options.PBDOptions(particle_size=1e-2),
        ),
        scene_options=SceneOptions(
            plane=PlaneOptions(material=frictionless),
            sphere=SphereOptions(
                radius=0.2,
                default_root_pos=(0.0, 0.0, 0.0),
                fixed=True,
                material=frictionless,
            ),
            cloth=EntityOptions(
                file=os.path.join(get_gs_assets_dir(), "meshes/cloth.obj"),
                scale=1.0,
                default_root_pos=(0.0, 0.0, 0.3),
                default_root_quat=(0.0, 1.0, 0.0, 0.0),  # euler (180, 0, 0)
                material=PBDClothMaterialOptions(),
                surface=gs.surfaces.Default(color=(0.2, 0.4, 0.8, 1.0), vis_mode="visual"),
                vis_mode="visual",
                # keep the thin shell mesh intact for the PBD solver
                convexify=False,
                decimate=False,
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
