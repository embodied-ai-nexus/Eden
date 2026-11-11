"""Mochi thin shells: a sheet draped over a rigid box, and a ball caught by a sheet whose held corners sway.

---
tier: recipe
capabilities: [solver.mochi]
hardware: gpu-fp64
optional_deps: []
run: uv run python examples/recipes/mochi/cloth.py -v
---

Eden port of the Genesis fork's ``examples/mochi/cloth.py``. The shell membrane and bending terms, the rigid
bodies and the penalty contact enter one implicit Newton solve. Shell samples collide against the rigid
colliders from both sides (this is also how the net catches the ball); the spheres placed at the cloth
vertices (point-cloud collider) only act against other cloths and rods.

Light cloth carries little inertia, so the default contact stiffness of 1e9 Pa/m makes the residual jump by
orders of magnitude when the sheet lands on an edge and trips the divergence check; 1e7 Pa/m keeps the
contact stiff enough for cloth. The net's corners are pinned after build and then driven along a sinusoidal
sway, which they reach exactly at the end of each step (``set_vertices_target``).

    uv run python examples/recipes/mochi/cloth.py -v
"""

import argparse

import genesis as gs
import numpy as np
from _mochi_common import (
    MOCHI_GRAVITY,
    VideoRecorder,
    init,
    newton_iterations,
    recording_cameras,
    sheet_mesh,
)

import eden as en
from eden.engine.solvers.mochi.options import MochiOptions
from eden.envs.base import EnvBase
from eden.options import EnvOptions, SceneOptions
from eden.options.entities import BoxOptions, EntityOptions, PlaneOptions, SphereOptions
from eden.options.materials import MochiRigidMaterialOptions, MochiShellMaterialOptions

CAM_POS, CAM_LOOKAT = (2.5, -2.5, 1.8), (0.0, 0.0, 0.5)
SHEET_CELLS, SHEET_SIZE = 12, 0.8
SWAY_START_STEP = 120


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

    sheet_path, _ = sheet_mesh(SHEET_CELLS, SHEET_SIZE)
    env = EnvBase(
        env_options=EnvOptions(
            num_envs=1,
            sim_dt=1.0 / 60.0,
            mochi_options=MochiOptions(gravity=MOCHI_GRAVITY, n_newton_iterations=8),
        ),
        scene_options=SceneOptions(
            plane=PlaneOptions(material=MochiRigidMaterialOptions()),
            box=BoxOptions(
                size=(0.3, 0.3, 0.3),
                default_root_pos=(-0.8, 0.0, 0.15),
                is_fixed_base=True,
                material=MochiRigidMaterialOptions(),
            ),
            drape=EntityOptions(
                file=sheet_path,
                default_root_pos=(-0.8, 0.0, 0.6),
                material=MochiShellMaterialOptions(
                    E=2e4,
                    nu=0.3,
                    rho=200.0,
                    thickness=2e-3,
                    friction=0.6,
                    collider_radius=0.02,
                    penalty_coefficient=1e7,
                ),
                surface=gs.surfaces.Default(color=(0.9, 0.4, 0.3)),
            ),
            net=EntityOptions(
                file=sheet_path,
                default_root_pos=(0.8, 0.0, 0.8),
                material=MochiShellMaterialOptions(
                    E=5e4, nu=0.3, rho=300.0, thickness=2e-3, collider_radius=0.02, penalty_coefficient=1e7
                ),
                surface=gs.surfaces.Default(color=(0.3, 0.6, 0.9)),
            ),
            ball=SphereOptions(
                radius=0.08, default_root_pos=(0.8, 0.0, 1.2), material=MochiRigidMaterialOptions(rho=300.0)
            ),
        ),
        cameras_options=recording_cameras(args, pos=CAM_POS, lookat=CAM_LOOKAT),
        show_viewer=args.vis,
    )
    env.build()
    drape, net, ball, box = (env.entities[k] for k in ("drape", "net", "ball", "box"))

    # The four corner vertices of the net: both in-plane coordinates at the half-size.
    half = 0.5 * SHEET_SIZE - 1e-6
    rest = net.init_positions
    corners = np.flatnonzero((np.abs(rest[:, 0]) > half) & (np.abs(rest[:, 1]) > half))
    net.set_vertices_fixed(corners)
    corners_rest = rest[corners]

    recorder = VideoRecorder(env, args, "cloth")
    for i_step in range(steps):
        if i_step >= SWAY_START_STEP:
            # The target is reached at the *end* of the step, hence i_step + 1.
            t = (i_step + 1 - SWAY_START_STEP) * env.env_options.sim_dt
            net.set_vertices_target(corners, corners_rest + np.array([0.15 * np.sin(np.pi * t), 0.0, 0.0]))
        env.scene.step()
        if i_step % 60 == 0:
            print(
                f"step {i_step:4d}: drape zmin={float(drape.get_vertex_positions()[..., 2].min()):.4f} "
                f"net zmin={float(net.get_vertex_positions()[..., 2].min()):.4f} "
                f"ball x={float(ball.get_pos()[0, 0]):.4f} z={float(ball.get_pos()[0, 2]):.4f} "
                f"box contact={float(box.get_links_net_contact_force()[0, 0, 2]):.2f} N "
                f"newton iterations={newton_iterations(env)}"
            )
        recorder.frame()
    recorder.close()


if __name__ == "__main__":
    main()
