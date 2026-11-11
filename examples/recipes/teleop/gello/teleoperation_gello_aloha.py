"""Teleoperate a bimanual ALOHA (dual ViperX-300) in sim from a GELLO leader device.

---
tier: recipe
capabilities: [teleop.gello, robot.aloha]
hardware: cpu
optional_deps: [gello]
run: uv run python examples/recipes/teleop/gello/teleoperation_gello_aloha.py -v
---
"""

from argparse import ArgumentParser

import genesis as gs

import eden as en
from eden.constants import EventMode, ReferenceSource
from eden.envs.base import EnvBase
from eden.terms import EventTerm
from eden.managers.terms.actions.joint_actions import ImplicitPDController
from eden.managers.terms.actions.binary_actions import BinaryJointController
from eden.managers.terms.events.placement import place_constant_xy, place_uniform
from eden.options import ActionManagerOptions, EnvOptions, EventManagerOptions, SceneOptions
from eden.options.entities import EntityOptions, PlaneOptions
from eden.options.materials import RigidMaterialOptions
from eden.options.robots.aloha_vx300s import Aloha
from eden.utils.devices.gello.zmq_server import GelloZMQServer


def main():
    parser = ArgumentParser()
    parser.add_argument("-c", "--cpu", action="store_true", help="Run on CPU instead of the GPU")
    parser.add_argument("-v", "--vis", action="store_true", help="Open the interactive viewer")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Hostname of the device server")
    parser.add_argument("--port", type=int, default=6001, help="Port of the device server")
    args = parser.parse_args()

    en.init(backend=gs.cpu if args.cpu else gs.gpu, debug=True)

    robot_options = Aloha(
        default_dofs_pos={
            "*/waist": 0.0,
            "*/shoulder": -1.85,
            "*/elbow": 1.6,
            "*/forearm_roll": 0.0,
            "*/wrist_angle": 0.0,
            "*/wrist_rotate": 0.0,
            "*/left_finger": 0.04,
            "*/right_finger": 0.04,
        },
        material=RigidMaterialOptions(gravity_compensation=1.0),
    )

    env = EnvBase(
        env_options=EnvOptions(
            num_envs=1,
            env_spacing=(1.5, 1.5),
            enable_default_keybinds=False,
            sim_dt=0.005,
        ),
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
            robot=robot_options,
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
            left_arm_dof_controller=ImplicitPDController.configure(
                entity_name="robot",
                dofs_name=[
                    "left/waist",
                    "left/shoulder",
                    "left/elbow",
                    "left/forearm_roll",
                    "left/wrist_angle",
                    "left/wrist_rotate",
                ],
                reference_source=ReferenceSource.ZERO,
                scale=1.0,
            ),
            left_gripper_controller=BinaryJointController.configure(
                entity_name="robot",
                dofs_name=[
                    "left/left_finger",
                    "left/right_finger",
                ],
                open_action={
                    "left/left_finger": Aloha.open_dofs_pos["left/left_finger"],
                    "left/right_finger": Aloha.open_dofs_pos["left/right_finger"],
                },
                close_action={
                    "left/left_finger": Aloha.close_dofs_pos["left/left_finger"],
                    "left/right_finger": Aloha.close_dofs_pos["left/right_finger"],
                },
                scale=1.0,
            ),
            right_arm_dof_controller=ImplicitPDController.configure(
                entity_name="robot",
                dofs_name=[
                    "right/waist",
                    "right/shoulder",
                    "right/elbow",
                    "right/forearm_roll",
                    "right/wrist_angle",
                    "right/wrist_rotate",
                ],
                reference_source=ReferenceSource.ZERO,
                scale=1.0,
            ),
            right_gripper_controller=BinaryJointController.configure(
                entity_name="robot",
                dofs_name=[
                    "right/left_finger",
                    "right/right_finger",
                ],
                open_action={
                    "right/left_finger": Aloha.open_dofs_pos["right/left_finger"],
                    "right/right_finger": Aloha.open_dofs_pos["right/right_finger"],
                },
                close_action={
                    "right/left_finger": Aloha.close_dofs_pos["right/left_finger"],
                    "right/right_finger": Aloha.close_dofs_pos["right/right_finger"],
                },
                scale=1.0,
            ),
        ),
        show_viewer=args.vis,
    )

    env.build()
    env.reset()

    en.logger.info(env.action_manager.summary())

    server_left = GelloZMQServer(
        env=env,
        arm_entity_name="robot",
        arm_dofs_name=[
            "left/waist",
            "left/shoulder",
            "left/elbow",
            "left/forearm_roll",
            "left/wrist_angle",
            "left/wrist_rotate",
        ],
        ee_link_name="left/gripper_base",
        gripper_entity_name="robot",
        gripper_dofs_name=[
            "left/left_finger",
            "left/right_finger",
        ],
        host=args.host,
        port=args.port,
    )
    server_right = GelloZMQServer(
        env=env,
        arm_entity_name="robot",
        arm_dofs_name=[
            "right/waist",
            "right/shoulder",
            "right/elbow",
            "right/forearm_roll",
            "right/wrist_angle",
            "right/wrist_rotate",
        ],
        ee_link_name="right/gripper_base",
        gripper_entity_name="robot",
        gripper_dofs_name=[
            "right/left_finger",
            "right/right_finger",
        ],
        host=args.host,
        port=args.port + 1,
    )

    try:
        print("Starting Gello server...")
        server_left.start()
        server_right.start()

        while True:
            cmd_left = server_left.get_command().to(env.device)
            cmd_right = server_right.get_command().to(env.device)
            action = {
                "left_arm_dof_controller": cmd_left[:6].unsqueeze(0),
                "right_arm_dof_controller": cmd_right[:6].unsqueeze(0),
                "left_gripper_controller": cmd_left[6:].unsqueeze(0),
                "right_gripper_controller": cmd_right[6:].unsqueeze(0),
            }
            env.step(action)

    finally:
        server_left.stop()
        server_right.stop()


if __name__ == "__main__":
    main()
