"""Mochi articulated arm: a Franka Panda grasps and lifts a small soft box under PD joint control.

---
tier: recipe
capabilities: [solver.mochi, robot.franka]
hardware: gpu-fp64
optional_deps: []
run: uv run python examples/recipes/mochi/articulated_arm.py -v
---

Eden port of the Genesis fork's ``examples/mochi/articulated_arm.py``. Joint drives, joint limits, armature
and damping enter the same implicit Newton solve as the rigid-body inertia and the contact penalty, so stiff
position gains remain stable at a 10 ms step. The robot is the stock ``FrankaEmikaPanda`` options with a
Mochi material; its gains come from the options like any other Eden robot.

    uv run python examples/recipes/mochi/articulated_arm.py -v
"""

import argparse

import genesis as gs
import numpy as np
import torch
from _mochi_common import (
    FRANKA_ARM_DOFS,
    FRANKA_FINGER_DOFS,
    VideoRecorder,
    init,
    recording_cameras,
)

import eden as en
from eden.engine.solvers.mochi.options import MochiOptions
from eden.envs.base import EnvBase
from eden.options import EnvOptions, SceneOptions
from eden.options.entities import BoxOptions, PlaneOptions
from eden.options.materials import MochiElasticMaterialOptions, MochiRigidMaterialOptions

CAM_POS, CAM_LOOKAT = (2.0, -2.0, 1.5), (0.0, 0.0, 0.5)
INIT_QPOS = (-1.0124, 1.5559, 1.3662, -1.6878, -1.5799, 1.7757, 1.4602, 0.04, 0.04)
HOLD_QPOS = (-1.0095, 1.5617, 1.3595, -1.6830, -1.5855, 1.7817, 1.4595, 0.0400, 0.0400)
LIFT_QPOS = (-1.0418, 1.3805, 1.5885, -1.7021, -1.3798, 1.5783, 1.4467, 0.0188, 0.0188)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-v", "--vis", action="store_true", help="Open the interactive viewer")
    parser.add_argument("-c", "--cpu", action="store_true", help="Run on the CPU backend instead of the GPU")
    parser.add_argument("-s", "--steps", type=int, default=100, help="Number of simulation steps")
    parser.add_argument("-r", "--record", action="store_true", help="Record a video of the run")
    parser.add_argument(
        "-o", "--output-dir", type=str, default="renders/mochi", help="Directory the recorded video is written to"
    )
    args = parser.parse_args()
    init(args)
    steps = 5 if en.is_smoke() else args.steps

    dofs_name = FRANKA_ARM_DOFS + FRANKA_FINGER_DOFS
    env = EnvBase(
        env_options=EnvOptions(
            num_envs=1,
            sim_dt=0.01,
            sim_substeps=2,
            mochi_options=MochiOptions(n_newton_iterations=8),
        ),
        scene_options=SceneOptions(
            plane=PlaneOptions(material=MochiRigidMaterialOptions()),
            franka=en.robots.FrankaEmikaPanda(
                material=MochiRigidMaterialOptions(friction=1.0, viscous_friction=1.0),
                default_dofs_pos=dict(zip(dofs_name, INIT_QPOS)),
            ),
            box=BoxOptions(
                size=(0.04, 0.04, 0.04),
                default_root_pos=(0.65, 0.0, 0.02),
                material=MochiElasticMaterialOptions(
                    E=2e4, nu=0.4, rho=800.0, stiffness_damping=1e-3, friction=1.0, viscous_friction=1.0
                ),
                surface=gs.surfaces.Default(color=(0.3, 0.7, 0.9)),
            ),
        ),
        cameras_options=recording_cameras(args, pos=CAM_POS, lookat=CAM_LOOKAT, res=(1280, 720)),
        show_viewer=args.vis,
    )
    env.build()
    franka, box = env.entities["franka"], env.entities["box"]
    _, arm_dofs = franka.find_named_dofs_idx_local(FRANKA_ARM_DOFS)
    _, finger_dofs = franka.find_named_dofs_idx_local(FRANKA_FINGER_DOFS)
    finger_force = torch.full((1, 2), -1.0, device=env.device)

    def run(n_steps: int, *, squeeze: bool) -> None:
        for _ in range(min(n_steps, steps)):
            if squeeze:
                franka.control_dofs_force(finger_force, dofs_idx_local=finger_dofs)
            env.scene.step()
            recorder.frame()

    recorder = VideoRecorder(env, args, "articulated_arm")
    # Hold above the box, close the fingers on it, then lift with the fingers still squeezing.
    franka.control_dofs_pos(torch.tensor([HOLD_QPOS[:7]], device=env.device), dofs_idx_local=arm_dofs)
    run(50, squeeze=False)
    run(50, squeeze=True)
    franka.control_dofs_pos(torch.tensor([LIFT_QPOS[:7]], device=env.device), dofs_idx_local=arm_dofs)
    run(steps, squeeze=True)
    recorder.close()

    print(
        f"box z: start=0.020 m -> final={float(np.asarray(box.get_pos()[0, 2].cpu())):.3f} m "
        f"(hand z={float(franka.get_link('hand').get_pos()[0, 2]):.3f} m)"
    )


if __name__ == "__main__":
    main()
