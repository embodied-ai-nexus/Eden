"""Mochi benchmark scene: a Franka arm pressing the middle of a 64-node rope lying on the ground.

---
tier: recipe
capabilities: [solver.mochi, robot.franka]
hardware: gpu-fp64
optional_deps: []
run: uv run python examples/recipes/mochi/rope_arm.py -v
---

Eden port of the Genesis fork's ``examples/mochi/rope_arm.py``, the ``rope_arm`` acceptance scene of the
fork's ``tests/mochi/benchmark`` run as a standalone demo: the hand descends onto the rope under PD joint
control while the fingertips press with 1 N each, pinning it against the ground. The discrete elastic rod
(stretching, bending, twisting), the articulated arm and the penalty contact enter one implicit Newton solve
at mochi's default budget of four iterations per step; the rope collides with the ground and the fingertips
through its centerline samples, so it comes to rest with the centerline on the ground.

    uv run python examples/recipes/mochi/rope_arm.py -v
"""

import argparse

import genesis as gs
import numpy as np
import torch
from _mochi_common import (
    FRANKA_ARM_DOFS,
    FRANKA_FINGER_DOFS,
    FRANKA_PRESS_START_QPOS,
    FRANKA_PRESS_TARGET_QPOS,
    FRANKA_PRESS_WORK_XY,
    MOCHI_GRAVITY,
    VideoRecorder,
    init,
    newton_iterations,
    recording_cameras,
    rod_section,
)

import eden as en
from eden.engine.solvers.mochi.options import MochiOptions
from eden.envs.base import EnvBase
from eden.options import EnvOptions, RodOptions, SceneOptions
from eden.options.entities import PlaneOptions
from eden.options.materials import MochiRigidMaterialOptions, MochiRodMaterialOptions

CAM_POS, CAM_LOOKAT = (1.5, -0.8, 0.7), (0.65, 0.0, 0.15)
ROPE_NODES, ROPE_LENGTH, ROPE_RADIUS = 64, 0.5, 5e-3
ROPE_E, ROPE_G, ROPE_RHO = 1e7, 1e7, 1000.0


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

    # The rope lies on the ground along x through the work point; the descending hand presses its middle.
    xs = FRANKA_PRESS_WORK_XY[0] + np.linspace(-0.5 * ROPE_LENGTH, 0.5 * ROPE_LENGTH, ROPE_NODES)
    points = np.stack(
        [xs, np.full(ROPE_NODES, FRANKA_PRESS_WORK_XY[1]), np.full(ROPE_NODES, ROPE_RADIUS + 1e-3)], axis=-1
    )
    dofs_name = FRANKA_ARM_DOFS + FRANKA_FINGER_DOFS

    env = EnvBase(
        env_options=EnvOptions(
            num_envs=1,
            sim_dt=1.0 / 60.0,
            mochi_options=MochiOptions(gravity=MOCHI_GRAVITY, n_newton_iterations=4),
        ),
        scene_options=SceneOptions(
            plane=PlaneOptions(material=MochiRigidMaterialOptions()),
            franka=en.robots.FrankaEmikaPanda(
                material=MochiRigidMaterialOptions(friction=1.0, viscous_friction=1.0),
                default_dofs_pos=dict(zip(dofs_name, FRANKA_PRESS_START_QPOS)),
            ),
            rope=RodOptions(
                points=points,  # the pose is baked into the polyline
                radius=ROPE_RADIUS,
                material=MochiRodMaterialOptions(**rod_section(ROPE_RADIUS, E=ROPE_E, G=ROPE_G, rho=ROPE_RHO)),
                surface=gs.surfaces.Default(color=(0.8, 0.6, 0.3)),
            ),
        ),
        cameras_options=recording_cameras(args, pos=CAM_POS, lookat=CAM_LOOKAT),
        show_viewer=args.vis,
    )
    env.build()
    franka, rope = env.entities["franka"], env.entities["rope"]
    _, arm_dofs = franka.find_named_dofs_idx_local(FRANKA_ARM_DOFS)
    _, finger_dofs = franka.find_named_dofs_idx_local(FRANKA_FINGER_DOFS)
    franka.control_dofs_pos(torch.tensor([FRANKA_PRESS_TARGET_QPOS[:7]], device=env.device), dofs_idx_local=arm_dofs)
    franka.control_dofs_force(torch.full((1, 2), -1.0, device=env.device), dofs_idx_local=finger_dofs)

    recorder = VideoRecorder(env, args, "rope_arm")
    for i_step in range(steps):
        env.scene.step()
        if i_step % 60 == 0:
            print(
                f"step {i_step:4d}: rope mid z={float(rope.get_vertex_positions()[0, ROPE_NODES // 2, 2]):.4f} "
                f"hand z={float(franka.get_link('hand').get_pos()[0, 2]):.4f} "
                f"newton iterations={newton_iterations(env)}"
            )
        recorder.frame()
    recorder.close()


if __name__ == "__main__":
    main()
