"""A fixed-base Franka pick-and-places a cube using inverse kinematics and PD control (rigid only).

---
tier: recipe
capabilities: [solver.rigid, robot.franka]
hardware: gpu
optional_deps: []
run: uv run python examples/recipes/physics/rigid_franka_cube.py -v
---

Eden port of Genesis' ``examples/rigid/franka_cube.py``.

Pure rigid-body pick-and-place: a fixed-base Franka Emika Panda uses inverse
kinematics to reach above a small cube, descend, close its gripper, and lift.
No soft solver or coupler — just the rigid solver, IK, and PD position control.

The whole demo is expressed through Eden options: ``FrankaEmikaPanda`` (which
already carries sensible ``default_dofs_kp``/``kd`` for the arm and fingers) and
a ``BoxOptions`` cube. IK and joint control use the Genesis ``RigidEntity`` reached
through ``env.entities["franka"].entity``.

Run with ``-v`` to open the viewer::

    uv run python examples/recipes/physics/rigid_franka_cube.py -v
"""

import argparse

import genesis as gs
import torch

import eden as en
from eden.envs.base import EnvBase
from eden.options import EnvOptions, SceneOptions
from eden.options.entities import BoxOptions, PlaneOptions
from eden.options.robots import FrankaEmikaPanda


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
            # NOTE: Genesis' franka_cube sets ``box_box_detection=True`` for a
            # crisper box-on-gripper contact. Eden's EnvOptions does not surface
            # that rigid-solver flag yet; the grasp is stable without it here.
        ),
        scene_options=SceneOptions(
            plane=PlaneOptions(),
            franka=FrankaEmikaPanda(),
            cube=BoxOptions(
                size=(0.04, 0.04, 0.04),
                default_root_pos=(0.65, 0.0, 0.02),
                surface=gs.surfaces.Default(color=(0.2, 0.8, 0.2, 1.0)),
            ),
        ),
        show_viewer=args.vis,
    )

    env.build()

    franka = env.entities["franka"].entity
    motors_dof, fingers_dof = slice(0, 7), slice(7, 9)
    end_effector = franka.get_link("hand")

    # Eden runs with num_envs >= 1, so IK targets are batched (B, 3) / (B, 4).
    B, device = env.num_envs, env.device
    # Gripper pointing straight down (wxyz).
    target_quat = torch.tensor([[0.0, 1.0, 0.0, 0.0]], device=device).repeat(B, 1)

    def reach(z: float):
        target_pos = torch.tensor([[0.65, 0.0, z]], device=device).repeat(B, 1)
        qpos = franka.inverse_kinematics(link=end_effector, pos=target_pos, quat=target_quat)
        franka.control_dofs_position(qpos[..., motors_dof], dofs_idx_local=motors_dof)

    def run(steps: int):
        for _ in range(steps if not en.is_smoke() else 2):
            env.scene.step()

    cube = env.entities["cube"]
    z0 = cube.get_pos()[0, 2].item()

    reach(0.25)  # above the cube, gripper open
    franka.control_dofs_position(torch.full((B, 2), 0.04, device=device), dofs_idx_local=fingers_dof)
    run(100)

    reach(0.13)  # descend
    run(100)

    franka.control_dofs_position(torch.zeros((B, 2), device=device), dofs_idx_local=fingers_dof)  # close
    run(50)

    reach(0.28)  # lift
    run(200)

    zf = cube.get_pos()[0, 2].item()
    print(f"\ncube z: start={z0:.3f} m  ->  final={zf:.3f} m   (lifted {zf - z0:+.3f} m)")


if __name__ == "__main__":
    main()
