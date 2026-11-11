"""
Franka robot grasping a deformable FEM cube through IPC coupling.

---
tier: recipe
capabilities: [solver.ipc, solver.fem, solver.rigid, robot.franka]
hardware: gpu
optional_deps: [libuipc]
run: uv run python examples/recipes/physics/ipc_robot_grasp_cube.py -v
---

Demonstrates Eden's FEM + IPC interface with articulated robot:
- FrankaEmikaPanda with coup_type="two_way_soft_constraint"
- FEMElasticMaterialOptions for a deformable cube
- IPCCouplerOptions tuned for robot-object interaction

Equivalent to references/Genesis/examples/IPC_Solver/ipc_robot_grasp_cube.py
but expressed through Eden's config-driven scene construction.

Known workarounds (see examples/recipes/physics/README.md, "IPC coupler" section):
- Use panda_no_tendon.xml: Eden's bundled panda.xml uses tendon-approximated
  actuators that break RigidEntity.post_build's get_dofs_kp read. [Fix 2]
- sanity_check_enable=False: Eden applies default_dofs_pos after scene.build(),
  so IPC sees the neutral (all-zeros) qpos and flags panda self-collision. [Fix 3]
- batch_dofs_info=False: required by IPC coupler.
"""

import argparse
import os

import genesis as gs
import torch

import eden as en
from eden.envs.base import EnvBase
from eden.options import EnvOptions, SceneOptions
from eden.options.entities import BoxOptions, PlaneOptions
from eden.options.materials import FEMElasticMaterialOptions, RigidMaterialOptions
from eden.options.robots import FrankaEmikaPanda
from eden.utils.misc import get_gs_assets_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", help="Open the interactive viewer")
    parser.add_argument(
        "--coup-type",
        type=str,
        default="two_way_soft_constraint",
        choices=["two_way_soft_constraint", "external_articulation"],
        help="How the gripper is coupled into the IPC solve",
    )
    args = parser.parse_args()

    en.init(backend=gs.gpu)

    franka_material_kwargs = dict(
        coup_friction=0.8,
        coup_type=args.coup_type,
    )
    if args.coup_type == "two_way_soft_constraint":
        franka_material_kwargs["coup_links"] = ["left_finger", "right_finger"]

    panda_xml = os.path.join(get_gs_assets_dir(), "xml/franka_emika_panda/panda_non_overlap.xml")

    env = EnvBase(
        env_options=EnvOptions(
            num_envs=1,
            sim_dt=0.01,
            sim_substeps=1,
            batch_dofs_info=False,
            coupler_options=gs.options.IPCCouplerOptions(
                constraint_strength_translation=10.0,
                constraint_strength_rotation=10.0,
                enable_rigid_rigid_contact=False,
                enable_rigid_ground_contact=False,
                newton_translation_tolerance=10.0,
                sanity_check_enable=False,
            ),
        ),
        scene_options=SceneOptions(
            plane=PlaneOptions(),
            franka=FrankaEmikaPanda(
                file=panda_xml,
                material=RigidMaterialOptions(**franka_material_kwargs),
            ),
            cube=BoxOptions(
                size=(0.05, 0.05, 0.05),
                default_root_pos=(0.65, 0.0, 0.03),
                material=FEMElasticMaterialOptions(
                    E=5.0e4,
                    nu=0.45,
                    rho=1000.0,
                    friction_mu=0.5,
                    model="stable_neohookean",
                ),
                surface=gs.surfaces.Plastic(color=(0.2, 0.8, 0.2, 0.5)),
            ),
        ),
        show_viewer=args.vis,
    )

    env.build()

    franka = env.entities["franka"].entity
    motors_dof, fingers_dof = slice(0, 7), slice(7, 9)
    end_effector = franka.get_link("hand")

    # Eden runs with num_envs>=1 so IK targets must be batched (B, 3) / (B, 4).
    B = env.num_envs
    device = env.device
    target_quat = torch.tensor([[0.0, 1.0, 0.0, 0.0]], device=device).repeat(B, 1)

    def ik(z: float):
        target_pos = torch.tensor([[0.65, 0.0, z]], device=device).repeat(B, 1)
        return franka.inverse_kinematics(link=end_effector, pos=target_pos, quat=target_quat)

    qpos = ik(0.4)  # above the cube, gripper open
    franka.control_dofs_position(qpos[..., motors_dof], dofs_idx_local=motors_dof)
    franka.control_dofs_position(0.04, dofs_idx_local=fingers_dof)
    for _ in range(200 if not en.is_smoke() else 1):
        env.scene.step()

    qpos = ik(0.25)  # descend
    franka.control_dofs_position(qpos[..., motors_dof], dofs_idx_local=motors_dof)
    for _ in range(100 if not en.is_smoke() else 1):
        env.scene.step()

    qpos = ik(0.135)  # grasp height
    franka.control_dofs_position(qpos[..., motors_dof], dofs_idx_local=motors_dof)
    for _ in range(50 if not en.is_smoke() else 1):
        env.scene.step()

    franka.control_dofs_position(qpos[..., motors_dof], dofs_idx_local=motors_dof)
    franka.control_dofs_position(0.0, dofs_idx_local=fingers_dof)  # close
    for _ in range(10 if not en.is_smoke() else 1):
        env.scene.step()

    qpos = ik(0.3)  # lift
    franka.control_dofs_position(qpos[..., motors_dof], dofs_idx_local=motors_dof)
    for _ in range(50 if not en.is_smoke() else 1):
        env.scene.step()


if __name__ == "__main__":
    main()
