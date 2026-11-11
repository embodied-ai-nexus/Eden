"""
Interactive Franka teleoperation with two FEM cloths and a grid of rigid cubes.

---
tier: recipe
capabilities: [solver.ipc, solver.fem, solver.rigid, robot.franka, teleop.spacemouse]
hardware: gpu
optional_deps: [libuipc, spacemouse]
run: uv run python examples/recipes/physics/ipc_robot_cloth_teleop.py -v
---

Demonstrates Eden's FEM + IPC interface with multi-entity scenes:
- Two FEMClothMaterialOptions cloths at different scales and stiffness
- 16 RigidMaterialOptions cubes (coup_type="ipc_only") forming a grid
- FrankaEmikaPanda with two_way_soft_constraint gripper coupling
- Keyboard teleop via SE3Keyboard

Equivalent to references/Genesis/examples/IPC_Solver/ipc_robot_cloth_teleop.py
but expressed through Eden's config-driven scene construction.

Keyboard Controls (SE3Keyboard default bindings):
- WASD / QE:  translate XYZ
- IJKL / UO:  rotate RPY
- space:      toggle gripper
- x:          reset scene

Known workarounds (see examples/recipes/physics/README.md, "IPC coupler" section):
- panda_no_tendon.xml: avoids RigidEntity.post_build's get_dofs_kp read on
  tendon-approximated actuators. [Fix 2]
- sanity_check_enable=False: Eden applies default_dofs_pos after scene.build(),
  so IPC sees the neutral (all-zeros) qpos. [Fix 3]
- batch_dofs_info=False: required by IPC coupler.
- Cloth orientation: grid20x20 mesh is authored flat in XZ plane (mesh up = Y),
  so we pass up=(0, 1, 0) to reorient it horizontally.
"""

import argparse
import os

import genesis as gs

import eden as en
from eden.envs.base import EnvBase
from eden.managers.terms.actions.binary_actions import BinaryJointController
from eden.managers.terms.actions.task_space_actions import DifferentialIKController
from eden.options import ActionManagerOptions, EnvOptions, SceneOptions
from eden.options.entities import EntityOptions, PlaneOptions
from eden.options.materials import (
    FEMClothMaterialOptions,
    RigidMaterialOptions,
)
from eden.options.robots import FrankaEmikaPanda
from eden.utils.devices.spacemouse import SE3Spacemouse
from eden.utils.misc import get_gs_assets_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", help="Open the interactive viewer")
    args = parser.parse_args()

    en.init(backend=gs.gpu)

    panda_xml = os.path.join(get_gs_assets_dir(), "xml/franka_emika_panda/panda_non_overlap.xml")

    scene_entities = dict(
        plane=PlaneOptions(
            material=RigidMaterialOptions(coup_type="ipc_only"),
        ),
        franka=FrankaEmikaPanda(
            file=panda_xml,
            default_root_pos=(0.0, 0.0, 0.005),
            material=RigidMaterialOptions(
                gravity_compensation=1.0,
                coup_type="two_way_soft_constraint",
                coup_links=["left_finger", "right_finger"],
            ),
        ),
        cloth_small=EntityOptions(
            file="objects/shirt/womens_shirt.obj",
            scale=0.8,
            default_root_pos=(0.5, 0.0, 0.15),
            material=FEMClothMaterialOptions(
                E=6e4,
                nu=0.25,
                rho=200.0,
                thickness=0.001,
                bending_stiffness=2.0,
                friction_mu=1.0,
            ),
            surface=gs.surfaces.Plastic(color=(0.3, 0.5, 0.8, 1.0)),
        ),
    )

    env = EnvBase(
        env_options=EnvOptions(
            num_envs=1,
            sim_dt=0.02,
            sim_substeps=1,
            decimation=1,  # teleop drives the env with env.step(action), so decimation is live
            enable_default_keybinds=False,
            batch_dofs_info=False,
            coupler_options=gs.options.IPCCouplerOptions(
                constraint_strength_translation=100.0,
                constraint_strength_rotation=100.0,
                n_linesearch_iterations=8,
                newton_tolerance=1e-1,
                newton_translation_tolerance=1.0,
                newton_semi_implicit_enable=False,
                linear_system_tolerance=1e-3,
                contact_enable=True,
                enable_rigid_rigid_contact=True,
                contact_d_hat=0.001,
                contact_resistance=1e7,
                sanity_check_enable=True,
            ),
        ),
        scene_options=SceneOptions(**scene_entities),
        action_options=ActionManagerOptions(
            ee_controller=DifferentialIKController.configure(
                entity_name="franka",
                ee_link_name="hand",
                dofs_name=[f"joint{i}" for i in range(1, 8)],
                position_scale=0.05,
                orientation_scale=0.05,
            ),
            gripper_controller=BinaryJointController.configure(
                entity_name="franka",
                dofs_name=["finger_joint1", "finger_joint2"],
                open_action=0.04,
                close_action=-0.03,
            ),
        ),
        show_viewer=args.vis,
    )

    env.build()
    env.reset()

    en.logger.info(env.action_manager.summary())

    keyboard = SE3Spacemouse()
    keyboard.add_callback("x", lambda on_press: env.reset() if on_press else None)

    while True:
        command = keyboard.get_command()
        action = {
            "ee_controller": command[:6].unsqueeze(0).repeat(env.num_envs, 1).to(env.device),
            "gripper_controller": command[6:].unsqueeze(0).repeat(env.num_envs, 1).to(env.device),
        }
        env.step(action)


if __name__ == "__main__":
    main()
