"""Keyboard SE(3) teleoperation of a Franka Research 3 that grasps objects via suction-cup welding.

---
tier: recipe
capabilities: [teleop.keyboard, robot.franka]
hardware: cpu
optional_deps: []
run: uv run python examples/recipes/teleop/keyboard/teleoperation_welding_suction.py -v
---
"""

from argparse import ArgumentParser

import genesis as gs

import eden as en
from eden.constants import EventMode
from eden.envs.base import EnvBase
from eden.terms import EventTerm
from eden.managers.terms.actions.task_space_actions import (
    DifferentialIKController,
    OperationalSpaceController,
)
from eden.managers.terms.actions.welding_actions import SuctionCupWelding
from eden.managers.terms.events.placement import place_range_xy
from eden.options import ActionManagerOptions, EnvOptions, EventManagerOptions, SceneOptions
from eden.options.entities import EntityOptions, PlaneOptions
from eden.options.materials import RigidMaterialOptions
from eden.utils.devices.keyboard import SE3Keyboard


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
        env_options=EnvOptions(
            num_envs=1,
            env_spacing=(1.5, 1.5),
            enable_default_keybinds=False,
        ),
        scene_options=SceneOptions(
            apple=EntityOptions(
                file="objects/apples/013_apple/textured.obj",
                is_fixed_base=False,
            ),
            floor=PlaneOptions(
                is_support_enabled=True,
                # default_root_pos=(0.0, 0.0, -0.5),
                default_root_pos=(0.0, 0.0, 0.0),
            ),
            robot=en.robots.FrankaResearch3(
                material=RigidMaterialOptions(
                    gravity_compensation=1.0,
                )
            ),
        ),
        event_options=EventManagerOptions(
            place_apple=EventTerm.configure(
                mode=EventMode.RESET,
                func=place_range_xy,
                params={
                    "entity_name": "apple",
                    "support_entity_name": "floor",
                    "range_x": (0.1, 0.5),
                    "range_y": (-0.5, 0.5),
                },
            ),
        ),
        action_options=ActionManagerOptions(
            ee_controller={
                "diffik": DifferentialIKController,
                "osc": OperationalSpaceController,
            }[args.action].configure(
                entity_name="robot",
                ee_link_name="attachment",
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
            gripper_controller=SuctionCupWelding.configure(
                entity_name="robot",
                ee_link_name="link7",
                ignore_entities_name=["floor"],
            ),
        ),
        show_viewer=args.vis,
    )

    env.build()
    env.reset()

    en.logger.info(env.action_manager.summary())

    keyboard = SE3Keyboard(key_bindings=SE3Keyboard.wasd_key_bindings())
    keyboard.add_callback("x", lambda on_press: env.reset() if on_press else None)

    while True:
        action = keyboard.get_command()
        action = {
            "ee_controller": action[:6].unsqueeze(0).repeat(env.num_envs, 1).to(env.device),
            "gripper_controller": action[6:].unsqueeze(0).repeat(env.num_envs, 1).to(env.device),
        }
        env.step(action)


if __name__ == "__main__":
    main()
