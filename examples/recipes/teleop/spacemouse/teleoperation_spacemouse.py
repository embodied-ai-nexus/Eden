"""SpaceMouse SE(3) teleoperation of a Franka Emika Panda with diffik or OSC end-effector control.

---
tier: recipe
capabilities: [teleop.spacemouse, robot.franka]
hardware: cpu
optional_deps: [spacemouse]
run: uv run python examples/recipes/teleop/spacemouse/teleoperation_spacemouse.py -v
---
"""

from argparse import ArgumentParser

import torch
import genesis as gs
import eden as en

from eden.options.entities import EntityOptions, PlaneOptions
from eden.envs.base import EnvBase
from eden.options import EnvOptions, SceneOptions, EventManagerOptions
from eden.options import ActionManagerOptions
from eden.options.materials import RigidMaterialOptions
from eden.managers.terms.actions.task_space_actions import (
    DifferentialIKController,
    OperationalSpaceController,
)
from eden.managers.terms.actions.binary_actions import BinaryJointController
from eden.terms import EventTerm
from eden.utils.devices.spacemouse import Spacemouse

from eden.managers.terms.events.placement import place_constant_xy, place_uniform
from eden.constants import EventMode


def main():
    parser = ArgumentParser()
    parser.add_argument("-c", "--cpu", action="store_true", help="Run on CPU instead of the GPU")
    parser.add_argument("-v", "--vis", action="store_true", help="Open the interactive viewer")
    parser.add_argument(
        "--action",
        type=str,
        default="diffik",
        choices=["diffik", "osc"],
        help="End-effector controller to drive the arm with",
    )
    args = parser.parse_args()

    en.init(backend=gs.cpu if args.cpu else gs.gpu, debug=True)

    env = EnvBase(
        env_options=EnvOptions(num_envs=1, env_spacing=(1.5, 1.5)),
        scene_options=SceneOptions(
            table=EntityOptions(
                file="objects/tables/work_table/work_table.obj", is_fixed_base=True, is_support_enabled=True
            ),
            plate=EntityOptions(
                file="objects/plates/86d4d4f8-ab98-4c48-9630-dc5c7710733b.glb",
                is_fixed_base=False,
            ),
            apple=EntityOptions(
                file="objects/apples/013_apple/textured.obj",
                is_fixed_base=False,
            ),
            floor=PlaneOptions(
                is_support_enabled=True,
                default_root_pos=(0.0, 0.0, -0.5),
            ),
            robot=en.robots.FrankaEmikaPanda(
                material=RigidMaterialOptions(
                    gravity_compensation=1.0,
                )
            ),
        ),
        event_options=EventManagerOptions(
            place_table=EventTerm.configure(
                mode=EventMode.STARTUP,
                func=place_constant_xy,
                params={
                    "entity_name": "table",
                    "support_entity_name": "floor",
                    "pos_xy": (0.45, 0.0),
                },
                priority=0,
            ),
            place_plate=EventTerm.configure(
                mode=EventMode.RESET,
                func=place_uniform,
                params={
                    "entity_name": "plate",
                    "support_entity_name": "table",
                    "samples_per_area": 100,
                },
                priority=1,
            ),
            place_apple=EventTerm.configure(
                mode=EventMode.RESET,
                func=place_uniform,
                params={
                    "entity_name": "apple",
                    "support_entity_name": "table",
                    "samples_per_area": 100,
                },
                priority=2,
            ),
        ),
        action_options=ActionManagerOptions(
            ee_controller={
                "diffik": DifferentialIKController,
                "osc": OperationalSpaceController,
            }[args.action].configure(
                entity_name="robot",
                ee_link_name="hand",
                dofs_name=[
                    "joint1",
                    "joint2",
                    "joint3",
                    "joint4",
                    "joint5",
                    "joint6",
                    "joint7",
                ],
                position_scale=0.05,
                orientation_scale=0.05,
            ),
            gripper_controller=BinaryJointController.configure(
                entity_name="robot",
                dofs_name=[
                    "finger_joint1",
                    "finger_joint2",
                ],
                open_action=0.04,
                close_action=0.0,
            ),
        ),
        show_viewer=args.vis,
    )

    env.build()
    env.reset()

    en.logger.info(env.action_manager.summary())

    spacemouse = Spacemouse(is_toggle_btn0=True, is_toggle_btn1=False)
    spacemouse.add_callback(1, lambda on_press: env.reset() if on_press else None)

    while True:
        action = spacemouse.get_command()
        gripper_action = spacemouse.is_button_pressed(0)
        action = {
            "ee_controller": action[:6].unsqueeze(0).repeat(env.num_envs, 1).to(env.device),
            "gripper_controller": torch.tensor([gripper_action], dtype=torch.float32)
            .unsqueeze(0)
            .repeat(env.num_envs, 1)
            .to(env.device),
        }
        env.step(action)


if __name__ == "__main__":
    main()
