"""Teleoperate a Franka Emika Panda in sim with on-screen GUI joint sliders.

---
tier: recipe
capabilities: [teleop.gui, robot.franka]
hardware: cpu
optional_deps: []
run: uv run python examples/recipes/teleop/gui/teleoperation_gui_franka.py
---
"""

from argparse import ArgumentParser

import torch
import genesis as gs

import eden as en
from eden.constants import ReferenceSource
from eden.envs.base import EnvBase
from eden.managers.terms.actions.joint_actions import ImplicitPDController
from eden.options import ActionManagerOptions, EnvOptions, SceneOptions
from eden.options.entities import PlaneOptions
from eden.options.materials import RigidMaterialOptions
from eden.options.robots.franka_emika_panda import FrankaEmikaPanda
from eden.utils.devices.gui_slider import SliderGUI


ARM_DOFS = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]
GRIPPER_DOFS = ["finger_joint1", "finger_joint2"]


def main():
    parser = ArgumentParser()
    parser.add_argument("-c", "--cpu", action="store_true", help="Run on CPU instead of the GPU")
    args = parser.parse_args()

    en.init(backend=gs.cpu if args.cpu else gs.gpu, debug=True)

    env = EnvBase(
        env_options=EnvOptions(num_envs=1, env_spacing=(1.5, 1.5)),
        scene_options=SceneOptions(
            floor=PlaneOptions(),
            robot=FrankaEmikaPanda(material=RigidMaterialOptions(gravity_compensation=1.0)),
        ),
        action_options=ActionManagerOptions(
            arm_controller=ImplicitPDController.configure(
                entity_name="robot",
                dofs_name=ARM_DOFS,
                reference_source=ReferenceSource.ZERO,
                scale=1.0,
            ),
            gripper_controller=ImplicitPDController.configure(
                entity_name="robot",
                dofs_name=GRIPPER_DOFS,
                reference_source=ReferenceSource.ZERO,
                scale=1.0,
            ),
        ),
        show_viewer=True,
    )

    env.build()
    env.reset()

    en.logger.info(env.action_manager.summary())

    robot = env.entities["robot"]
    all_dofs = ARM_DOFS + GRIPPER_DOFS
    _, dof_idx = robot.find_named_dofs_idx_local(all_dofs, preserve_order=True)
    lo, hi = robot.get_dofs_limit(dof_idx)
    limits = torch.stack([lo, hi], dim=-1)[0].cpu()
    init_pos = robot.get_dofs_pos(dof_idx).squeeze(0).cpu()

    slider = SliderGUI(
        viewer=env.scene.viewer,
        dofs_name=all_dofs,
        dofs_pos_limits=limits,
        initial_position=init_pos,
        reset_callback=env.reset,
    )

    while env.scene.viewer.is_alive():
        action = slider.get_command().to(env.device)
        env.step(
            {
                "arm_controller": action[:, : len(ARM_DOFS)],
                "gripper_controller": action[:, len(ARM_DOFS) :],
            }
        )

    slider.close()


if __name__ == "__main__":
    main()
