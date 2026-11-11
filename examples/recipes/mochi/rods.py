"""Mochi rods: a rope clamped at both ends, a stiff cantilever, and a loose rope dropped onto the ground.

---
tier: recipe
capabilities: [solver.mochi]
hardware: gpu-fp64
optional_deps: []
run: uv run python examples/recipes/mochi/rods.py -v
---

Eden port of the Genesis fork's ``examples/mochi/rods.py``. Rods are discrete elastic rods (stretching,
bending and twisting) whose nodes and segment twists are unknowns of the same implicit Newton solve as the
other Mochi bodies. A rod is declared with ``RodOptions`` (the centerline) and a ``MochiRodMaterialOptions``;
clamping means fixing nodes after build — two nodes to clamp the tangent as well.

    uv run python examples/recipes/mochi/rods.py -v
"""

import argparse

import genesis as gs
import numpy as np
from _mochi_common import MOCHI_GRAVITY, VideoRecorder, init, newton_iterations, recording_cameras

import eden as en
from eden.engine.solvers.mochi.options import MochiOptions
from eden.envs.base import EnvBase
from eden.options import EnvOptions, RodOptions, SceneOptions
from eden.options.entities import PlaneOptions
from eden.options.materials import MochiRigidMaterialOptions, MochiRodMaterialOptions

CAM_POS, CAM_LOOKAT = (2.5, -2.5, 1.5), (0.0, 0.0, 0.8)
N_SEGMENTS = 40


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-v", "--vis", action="store_true", help="Open the interactive viewer")
    parser.add_argument("-c", "--cpu", action="store_true", help="Run on the CPU backend instead of the GPU")
    parser.add_argument("-s", "--steps", type=int, default=300, help="Number of simulation steps")
    parser.add_argument("-r", "--record", action="store_true", help="Record a video of the run")
    parser.add_argument(
        "-o", "--output-dir", type=str, default="renders/mochi", help="Directory the recorded video is written to"
    )
    args = parser.parse_args()
    init(args)
    steps = 5 if en.is_smoke() else args.steps

    n_nodes = N_SEGMENTS + 1
    straight = np.stack([np.linspace(-0.75, 0.75, n_nodes), np.zeros(n_nodes), np.zeros(n_nodes)], axis=-1)
    wave = np.stack(
        [np.linspace(-0.5, 0.5, n_nodes), 0.15 * np.sin(np.linspace(0.0, 4.0 * np.pi, n_nodes)), np.zeros(n_nodes)],
        axis=-1,
    )
    rope_material = MochiRodMaterialOptions(E=5e6, nu=0.3, rho=1200.0, mass_damping=1.0)

    env = EnvBase(
        env_options=EnvOptions(
            num_envs=1,
            sim_dt=1.0 / 60.0,
            mochi_options=MochiOptions(gravity=MOCHI_GRAVITY, n_newton_iterations=8),
        ),
        scene_options=SceneOptions(
            plane=PlaneOptions(material=MochiRigidMaterialOptions()),
            rope=RodOptions(
                points=straight,
                radius=0.01,
                default_root_pos=(0.0, 0.0, 1.2),
                material=rope_material,
                surface=gs.surfaces.Default(color=(0.8, 0.6, 0.2)),
            ),
            beam=RodOptions(
                points=straight[: N_SEGMENTS // 2 + 1] + (0.75, 0.0, 0.0),
                radius=0.015,
                default_root_pos=(0.0, 0.5, 0.8),
                material=MochiRodMaterialOptions(E=2e9, nu=0.3, rho=2000.0, mass_damping=2.0),
                surface=gs.surfaces.Default(color=(0.3, 0.5, 0.9)),
            ),
            loose=RodOptions(
                points=wave,
                radius=0.01,
                default_root_pos=(0.0, -0.6, 0.4),
                default_root_quat=tuple(gs.utils.geom.xyz_to_quat(np.array([15.0, 0.0, 0.0]), degrees=True).tolist()),
                material=rope_material,
                surface=gs.surfaces.Default(color=(0.4, 0.8, 0.4)),
            ),
        ),
        cameras_options=recording_cameras(args, pos=CAM_POS, lookat=CAM_LOOKAT),
        show_viewer=args.vis,
    )
    env.build()
    rope, beam, loose = env.entities["rope"], env.entities["beam"], env.entities["loose"]
    rope.set_vertices_fixed([0, N_SEGMENTS])  # both ends
    beam.set_vertices_fixed([0, 1])  # a clamp needs two nodes: position and tangent

    recorder = VideoRecorder(env, args, "rods")
    for i_step in range(steps):
        env.scene.step()
        if i_step % 60 == 0:
            print(
                f"step {i_step:4d}: rope lowest z={float(rope.get_vertex_positions()[..., 2].min()):.4f} "
                f"beam tip z={float(beam.get_vertex_positions()[0, -1, 2]):.4f} "
                f"dropped rope mean z={float(loose.get_vertex_positions()[..., 2].mean()):.4f} "
                f"newton iterations={newton_iterations(env)}"
            )
        recorder.frame()
    recorder.close()


if __name__ == "__main__":
    main()
