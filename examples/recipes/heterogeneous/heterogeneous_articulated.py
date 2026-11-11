"""Heterogeneous articulated Arctic object variants dropped across parallel environments.

---
tier: recipe
capabilities: [scene.heterogeneous]
hardware: cpu
optional_deps: []
run: uv run python examples/recipes/heterogeneous/heterogeneous_articulated.py -v
---

Four different Arctic household objects (box, ketchup, mixer, notebook) are loaded
as heterogeneous variants via GroupedEntityOptions — they share the same joint structure
(2 links + 1 revolute joint named "rotation") but have different meshes, masses, and
inertial properties.

Each variant is dropped from mid-air onto a ground plane across parallel environments.

Usage:
    python heterogeneous_articulated.py -v             # 4 envs, one per variant
    python heterogeneous_articulated.py -v -b 8        # 8 envs, two per variant
"""

import argparse

import numpy as np
import genesis as gs
import eden as en

from eden.envs.base import EnvBase
from eden.options import EnvOptions, SceneOptions
from eden.options.entities import GroupedEntityOptions, PlaneOptions
from eden.options.objects.arctic import ArcticBox, ArcticKetchup, ArcticMixer, ArcticNotebook


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", help="Open the interactive viewer")
    parser.add_argument("-c", "--cpu", action="store_true", help="Run on CPU instead of the GPU")
    parser.add_argument("-b", "--num-envs", type=int, default=4, help="Number of parallel environments")
    args = parser.parse_args()

    en.init(backend=gs.cpu if args.cpu else gs.gpu)

    drop_height = 0.7

    env = EnvBase(
        env_options=EnvOptions(
            num_envs=args.num_envs,
            env_spacing=(0.8, 0.8),
        ),
        scene_options=SceneOptions(
            ground=PlaneOptions(),
            obj=GroupedEntityOptions(
                grouped_entities=[
                    ArcticBox(default_root_pos=(0.0, 0.0, drop_height)),
                    ArcticKetchup(default_root_pos=(0.0, 0.0, drop_height)),
                    ArcticMixer(default_root_pos=(0.0, 0.0, drop_height)),
                    ArcticNotebook(default_root_pos=(0.0, 0.0, drop_height)),
                ],
            ),
        ),
        show_viewer=args.vis,
    )

    env.build()

    obj = env.entities["obj"]
    mass = obj.get_mass()
    print(f"Object mass per env: {mass}")

    rotation_dof_idx = obj.entity.get_joint("rotation").dofs_idx_local[0]
    swing_amplitude = 0.5  # rad
    swing_frequency_hz = 0.5
    phase_offsets = np.linspace(0.0, 2.0 * np.pi, args.num_envs, endpoint=False)

    for i in range(5 if en.is_smoke() else 500):
        t = i * env.sim_dt
        target_angle = swing_amplitude * np.sin(2.0 * np.pi * swing_frequency_hz * t + phase_offsets)
        obj.control_dofs_pos(target_angle, dofs_idx_local=[rotation_dof_idx])
        env.step()


if __name__ == "__main__":
    main()
