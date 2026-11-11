"""A stream of MPM sand pours over a cascade of rigid wheels via two-way MPM-rigid coupling.

---
tier: recipe
capabilities: [solver.mpm, solver.rigid]
hardware: gpu
optional_deps: []
run: uv run python examples/recipes/physics/mpm_sand_wheel.py -v
---

Eden port of Genesis' ``examples/coupling/sand_wheel.py``.

A stream of MPM **sand** is poured onto a cascade of four fixed wheels and flows
down, two-way coupled to the rigid wheels. Uses the new ``MPMSandMaterialOptions``.

The sand is produced by a Genesis *emitter*. Eden does not (yet) wrap emitters, so
this example reaches through ``env.scene.add_emitter(...)`` — the one deliberate
raw-Genesis escape hatch in the physics examples (see the README "Known gaps").
The emitter must be created *after* ``EnvBase(...)`` (the scene exists) but *before*
``env.build()``.

    uv run python examples/recipes/physics/mpm_sand_wheel.py -v
"""

import os
import math
import argparse

import numpy as np
import genesis as gs

import eden as en
from eden.envs.base import EnvBase
from eden.options import EnvOptions, SceneOptions
from eden.options.entities import EntityOptions, PlaneOptions
from eden.options.materials import MPMSandMaterialOptions, RigidMaterialOptions
from eden.utils.misc import get_gs_assets_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-v", "--vis", action="store_true", help="Open the interactive viewer")
    args = parser.parse_args()

    en.init(backend=gs.gpu)

    wheel_mat = RigidMaterialOptions(needs_coup=True, coup_softness=0.0)
    wheel_file = os.path.join(get_gs_assets_dir(), "urdf/wheel/wheel.urdf")
    wheel_positions = [(0.5, -0.2, 1.6), (0.5, 0.3, 1.2), (0.5, -0.3, 0.8), (0.5, 0.4, 0.4)]
    wheels = {
        f"wheel_{i}": EntityOptions(
            file=wheel_file,
            scale=0.6,
            default_root_pos=pos,
            default_root_quat=(0.70710678, 0.0, 0.0, 0.70710678),  # euler (0, 0, 90)
            is_fixed_base=True,
            convexify=False,
            material=wheel_mat,
        )
        for i, pos in enumerate(wheel_positions)
    }

    env = EnvBase(
        env_options=EnvOptions(
            num_envs=1,
            sim_dt=3e-3,
            sim_substeps=10,
            mpm_options=gs.options.MPMOptions(
                lower_bound=(0.0, -1.0, -0.1),
                upper_bound=(0.57, 1.0, 2.4),
                grid_density=64,
            ),
        ),
        scene_options=SceneOptions(
            plane=PlaneOptions(material=RigidMaterialOptions(needs_coup=True, coup_friction=0.2)),
            **wheels,
        ),
        show_viewer=args.vis,
    )

    # --- raw-Genesis emitter (no Eden wrapper yet); add before build ---
    emitter = env.scene.add_emitter(
        material=MPMSandMaterialOptions().to_genesis_material(),
        max_particles=200000,
        surface=gs.surfaces.Rough(color=(1.0, 0.9, 0.6, 1.0)),
    )

    env.build()

    horizon = 1000 if not en.is_smoke() else 5
    for i in range(horizon):
        emitter.emit(
            pos=np.array([0.5, 0.0, 2.3]),
            direction=np.array([0.0, math.sin(i / 10) * 0.35, -1.0]),
            speed=8.0,
            droplet_shape="rectangle",
            droplet_size=[0.03, 0.05],
        )
        env.scene.step()


if __name__ == "__main__":
    main()
