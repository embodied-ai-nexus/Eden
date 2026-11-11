"""Mochi benchmark scene: a Franka arm pressing a 31x31 self-colliding cloth against the ground.

---
tier: recipe
capabilities: [solver.mochi, robot.franka]
hardware: gpu-fp64
optional_deps: []
run: uv run python examples/recipes/mochi/cloth_arm.py -v
---

Eden port of the Genesis fork's ``examples/mochi/cloth_arm.py``, the ``cloth_arm`` acceptance scene of the
fork's ``tests/mochi/benchmark`` (its reinforcement-learning-shaped scene) run as a standalone demo. The hand
starts about 0.30 m above the cloth and is driven down to about 0.13 m by PD joint control while the
fingertips press with 1 N each, wrinkling the sheet against the ground. The articulated arm, the shell
elements, the self-contact of the cloth and the penalty contact all enter one implicit Newton solve, at
mochi's default budget of four iterations per step.

    uv run python examples/recipes/mochi/cloth_arm.py -v
"""

import argparse

import genesis as gs
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
    sheet_mesh,
)

import eden as en
from eden.engine.solvers.mochi.options import MochiOptions
from eden.envs.base import EnvBase
from eden.options import EnvOptions, SceneOptions
from eden.options.entities import EntityOptions, PlaneOptions
from eden.options.materials import MochiRigidMaterialOptions, MochiShellMaterialOptions

CAM_POS, CAM_LOOKAT = (1.8, -1.0, 0.9), (0.65, 0.0, 0.2)
CLOTH_CELLS, CLOTH_SIZE, CLOTH_HEIGHT = 30, 0.3, 0.01


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

    dofs_name = FRANKA_ARM_DOFS + FRANKA_FINGER_DOFS
    sheet_path, _ = sheet_mesh(CLOTH_CELLS, CLOTH_SIZE)
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
            cloth=EntityOptions(
                file=sheet_path,
                default_root_pos=(*FRANKA_PRESS_WORK_XY, CLOTH_HEIGHT),
                material=MochiShellMaterialOptions(
                    E=2e4,
                    nu=0.3,
                    rho=200.0,
                    thickness=1e-3,
                    friction=0.6,
                    collider_radius=5e-3,
                    penalty_coefficient=1e7,
                    self_contact=True,
                ),
                surface=gs.surfaces.Default(color=(0.9, 0.4, 0.3)),
            ),
        ),
        cameras_options=recording_cameras(args, pos=CAM_POS, lookat=CAM_LOOKAT),
        show_viewer=args.vis,
    )
    env.build()
    franka, cloth = env.entities["franka"], env.entities["cloth"]
    _, arm_dofs = franka.find_named_dofs_idx_local(FRANKA_ARM_DOFS)
    _, finger_dofs = franka.find_named_dofs_idx_local(FRANKA_FINGER_DOFS)
    # One standing command: the arm presses down to the target pose while the fingertips squeeze.
    franka.control_dofs_pos(torch.tensor([FRANKA_PRESS_TARGET_QPOS[:7]], device=env.device), dofs_idx_local=arm_dofs)
    franka.control_dofs_force(torch.full((1, 2), -1.0, device=env.device), dofs_idx_local=finger_dofs)

    recorder = VideoRecorder(env, args, "cloth_arm")
    for i_step in range(steps):
        env.scene.step()
        if i_step % 60 == 0:
            print(
                f"step {i_step:4d}: cloth zmax={float(cloth.get_vertex_positions()[..., 2].max()):.4f} "
                f"hand z={float(franka.get_link('hand').get_pos()[0, 2]):.4f} "
                f"newton iterations={newton_iterations(env)}"
            )
        recorder.frame()
    recorder.close()


if __name__ == "__main__":
    main()
