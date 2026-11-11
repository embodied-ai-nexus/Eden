"""MuJoCo-style contype/conaffinity bitmask collision filtering across falling cubes and a dragon.

---
tier: recipe
capabilities: [solver.rigid, collision.contype]
hardware: gpu
optional_deps: []
run: uv run python examples/recipes/physics/rigid_contype.py -v
---

Eden port of Genesis' ``examples/recipes/collision/contype.py``.

MuJoCo-style collision filtering via ``contype`` / ``conaffinity`` bitmasks. Two
geoms collide only if the ``contype`` of one and the ``conaffinity`` of the other
share a set bit. Eden exposes both as plain fields on every entity option
(``EntityOptions.contype`` / ``.conaffinity``), so the whole demo is declarative.

Three unit cubes and a dragon mesh fall onto the ground plane:

- red cube   ``0b001`` — shares bit 0 with blue
- green cube ``0b010`` — shares bit 1 with blue
- blue cube  ``0b011`` — collides with both red and green
- dragon     ``0b100`` — collides with nothing but the plane

The plane keeps the default full mask (``0xFFFF``), so everything lands on it;
red and green (no shared bit) pass through each other, while blue interacts with
both.

    uv run python examples/recipes/physics/rigid_contype.py -v
"""

import os
import argparse

import genesis as gs

import eden as en
from eden.envs.base import EnvBase
from eden.options import EnvOptions, SceneOptions
from eden.options.entities import BoxOptions, EntityOptions, PlaneOptions
from eden.utils.misc import get_gs_assets_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-v", "--vis", action="store_true", help="Open the interactive viewer")
    args = parser.parse_args()

    en.init(backend=gs.gpu)

    env = EnvBase(
        env_options=EnvOptions(
            num_envs=1,
            sim_dt=0.01,
            sim_substeps=1,
        ),
        scene_options=SceneOptions(
            plane=PlaneOptions(),
            red=BoxOptions(
                size=(0.1, 0.1, 0.1),
                default_root_pos=(0.025, 0.0, 0.5),
                contype=0b001,
                conaffinity=0b001,
                surface=gs.surfaces.Default(color=(1.0, 0.0, 0.0, 1.0)),
            ),
            green=BoxOptions(
                size=(0.1, 0.1, 0.1),
                default_root_pos=(-0.025, 0.0, 1.0),
                contype=0b010,
                conaffinity=0b010,
                surface=gs.surfaces.Default(color=(0.0, 1.0, 0.0, 1.0)),
            ),
            blue=BoxOptions(
                size=(0.1, 0.1, 0.1),
                default_root_pos=(0.0, 0.0, 1.5),
                contype=0b011,
                conaffinity=0b011,
                surface=gs.surfaces.Default(color=(0.0, 0.0, 1.0, 1.0)),
            ),
            dragon=EntityOptions(
                file=os.path.join(get_gs_assets_dir(), "meshes/dragon/dragon.obj"),
                scale=0.004,
                default_root_pos=(-0.1, 0.0, 1.0),
                default_root_quat=(0.70710678, 0.0, 0.0, 0.70710678),  # euler (0, 0, 90)
                contype=0b100,
                conaffinity=0b100,
            ),
        ),
        show_viewer=args.vis,
    )

    env.build()

    horizon = 1000 if not en.is_smoke() else 5
    for _ in range(horizon):
        env.scene.step()

    # Final heights make the filtering visible headless: the dragon (0b100) shares
    # no bit with any cube, so it rests directly on the plane.
    for name in ("red", "green", "blue", "dragon"):
        print(f"  {name:6s} final z = {env.entities[name].get_pos()[0, 2].item():.3f} m")


if __name__ == "__main__":
    main()
