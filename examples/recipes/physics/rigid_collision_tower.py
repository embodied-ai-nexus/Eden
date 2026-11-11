"""A 100-box Jenga tower is crushed by a dropped object -- a rigid broad-phase collision stress test.

---
tier: recipe
capabilities: [solver.rigid]
hardware: cpu
optional_deps: []
run: uv run python examples/recipes/physics/rigid_collision_tower.py -v
---

Eden port of Genesis' ``examples/recipes/collision/tower.py``.

A rigid-body collision stress test: a 100-box Jenga-style tower (50 layers, two
boxes per layer, each layer rotated 90 degrees from the one below) is crushed by
a heavy object dropped from above. Exercises the rigid solver's broad-phase with
a raised ``max_collision_pairs`` and Genesis' ``performance_mode`` on the CPU
backend, exactly like the original.

Choose the dropped object with ``--object {sphere,cylinder,duck}``::

    uv run python examples/recipes/physics/rigid_collision_tower.py -v
    uv run python examples/recipes/physics/rigid_collision_tower.py --object duck -v
"""

import os
import argparse

import genesis as gs

import eden as en
from eden.envs.base import EnvBase
from eden.options import EnvOptions, SceneOptions
from eden.options.entities import BoxOptions, CylinderOptions, EntityOptions, PlaneOptions, SphereOptions
from eden.utils.misc import get_gs_assets_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--object",
        type=str,
        default="cylinder",
        choices=("sphere", "cylinder", "duck"),
        help="Object to spawn in the scene",
    )
    parser.add_argument("-v", "--vis", action="store_true", help="Open the interactive viewer")
    args = parser.parse_args()

    # CPU + performance_mode + fp32, matching the original stress test.
    en.init(backend=gs.cpu, performance_mode=True)

    # Build the tower: 50 layers, two boxes per layer, orientation alternating
    # every layer. ``box_height - 1e-3`` overlaps neighbouring layers slightly so
    # the stack starts in contact.
    box_width, box_length, box_height = 0.25, 2.0, 0.1
    num_stacks = 50
    boxes: dict[str, BoxOptions] = {}
    for i in range(num_stacks):
        z = i * (box_height - 1e-3) + 0.5 * box_height
        if i % 2 == 0:  # horizontal layer
            size = (box_width, box_length, box_height)
            positions = ((-0.4 * box_length, 0.0, z), (+0.4 * box_length, 0.0, z))
        else:  # vertical layer
            size = (box_length, box_width, box_height)
            positions = ((0.0, -0.4 * box_length, z), (0.0, +0.4 * box_length, z))
        for j, pos in enumerate(positions):
            boxes[f"box_{i}_{j}"] = BoxOptions(size=size, default_root_pos=pos)

    if args.object == "duck":
        scale = 0.8
        dropped = EntityOptions(
            file=os.path.join(get_gs_assets_dir(), "meshes/duck.obj"),
            scale=scale,
            default_root_pos=(0.0, -0.1, num_stacks * box_height + 10 * scale),
        )
    elif args.object == "sphere":
        r = 2.0
        dropped = SphereOptions(radius=r, default_root_pos=(0.0, 0.0, num_stacks * box_height + 5 * r))
    else:  # cylinder
        r, h = 2.0, 1.0
        dropped = CylinderOptions(radius=r, height=h, default_root_pos=(0.0, 0.0, num_stacks * box_height + 5 * h))

    env = EnvBase(
        env_options=EnvOptions(
            num_envs=1,
            sim_dt=0.0015,
            sim_substeps=1,
            max_collision_pairs=400,
        ),
        scene_options=SceneOptions(plane=PlaneOptions(), dropped=dropped, **boxes),
        show_viewer=args.vis,
    )

    env.build()

    horizon = 20 if en.is_smoke() else 10000
    for _ in range(horizon):
        env.scene.step()


if __name__ == "__main__":
    main()
