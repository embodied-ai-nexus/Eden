"""Keyboard teleoperation of the real Franka Emika Panda via panda-py.

---
tier: recipe
capabilities: [teleop.keyboard, robot.franka, deploy.real_robot]
hardware: real-robot
optional_deps: []
run: LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 uv run python examples/recipes/deployment/real_franka_teleoperation_keyboard.py
---

Drives the real arm with a 6-DoF SE3 keyboard while mirroring the live joint
state into the sim viewer. Like every deployment, the arm **boots in STOP,
holding the pose it is measured at**: nothing the keyboard commands reaches it
until you press ``o`` (START). ``i`` first ramps it to the home pose
(``default_dofs_pos``), ``x`` stops and holds, ``k`` exits.

The operator keys are read by the same keyboard device as the teleop, not
from the terminal: that device hears every key typed into the terminal too,
and the deployment's own terminal commands collide with the WASD bindings
(``s`` would be START and "move back" at once). So the deployment's terminal
reader is closed (``operator_keyboard=False``) and this script relays the keys
below to it.

Controls (panda-py / libfranka must be set up; FCI active, brakes released):

    i / o / x / k:            INIT (ramp home) / START / STOP / KILL
    SE3 translation/rotation: W/S, A/D, Q/E, arrows, Z/C (SE3Keyboard WASD bindings)
    Spacebar:                 toggle gripper open/close (BinaryJointController)

Usage:
    LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
        uv run examples/recipes/deployment/real_franka_teleoperation_keyboard.py
    # or with the OSC controller:
    LD_PRELOAD=... uv run examples/recipes/deployment/real_franka_teleoperation_keyboard.py --action osc
"""

import collections
from argparse import ArgumentParser

import genesis as gs

import eden as en
from eden.envs.base import RLEnvBase
from eden.extensions.deployment.franka_panda import FrankaPandaDeployment
from eden.managers.terms.actions.binary_actions import BinaryJointController
from eden.managers.terms.actions.task_space_actions import (
    DifferentialIKController,
    OperationalSpaceController,
)
from eden.options import ActionManagerOptions, EnvOptions
from eden.options.materials import RigidMaterialOptions
from eden.options.robots.franka_emika_panda import FrankaEmikaPanda
from eden.options.scenes.empty import EmptySceneOptions
from eden.utils.devices.keyboard import SE3Keyboard


CONTROL_FREQ = 30  # Hz - both sim and deployment

# Operator key -> what it asks the deployment, on keys the WASD teleop bindings leave free ("k" is KILL).
OPERATOR_KEYS = {
    "i": FrankaPandaDeployment.request_init,
    "o": FrankaPandaDeployment.request_start,
    "x": FrankaPandaDeployment.request_stop,
    "k": None,
}


def main():
    parser = ArgumentParser()
    parser.add_argument("-c", "--cpu", action="store_true", help="Run on CPU instead of the GPU")
    parser.add_argument("--no-vis", dest="vis", action="store_false", help="Run headless instead of opening the viewer")
    parser.add_argument(
        "--action",
        type=str,
        default="diffik",
        choices=["diffik", "osc"],
        help="End-effector controller to drive the arm with",
    )
    parser.add_argument("--ip", type=str, default="192.168.1.1", help="Franka Control box IP")
    args = parser.parse_args()

    en.init(backend=gs.cpu if args.cpu else gs.gpu, debug=True)

    control_dt = 1.0 / CONTROL_FREQ

    env = RLEnvBase(
        env_options=EnvOptions(
            num_envs=1,
            sim_dt=control_dt,
            decimation=1,
            enable_default_keybinds=False,
            show_world_frame=True,
        ),
        scene_options=EmptySceneOptions(
            robot=FrankaEmikaPanda(
                material=RigidMaterialOptions(
                    gravity_compensation=1.0,
                )
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
                position_scale=0.01,
                orientation_scale=0.01,
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
        show_viewer=args.vis and not en.is_smoke(),
    )
    env.build()

    dep = FrankaPandaDeployment(
        env,
        FrankaPandaDeployment.configure(
            entity_name="robot",
            control_freq=CONTROL_FREQ,
            sync=True,
            connect_timeout_s=10.0,
            robot_ip=args.ip,
            # The teleop keyboard below relays the operator keys; see the module docstring.
            operator_keyboard=False,
        ),
    )
    dep.connect()

    # ``env.reset()`` resets the *sim* scene to its initial state and runs
    # RESET-mode events. It does NOT invoke ``dep.reset()`` — see
    # ``eden/envs/base.py:reset()``. So we call them in sequence:
    #   1. env.reset()  -> sim scene back to defaults, RESET events fire.
    #   2. dep.reset()  -> reads the arm and boots the operator state machine
    #                       in STOP: each step holds the measured pose (the
    #                       first one starts the JointPosition controller
    #                       there) until START hands it to the keyboard.
    env.reset()
    dep.reset()

    en.logger.info(env.action_manager.summary())
    en.logger.info(
        "[teleop] Holding in STOP. Press o to START (i first to ramp home), then move the arm with the WASD "
        "bindings; spacebar toggles the gripper. x stops and holds, k or Ctrl-C exits."
    )

    keyboard = SE3Keyboard(key_bindings=SE3Keyboard.wasd_key_bindings())
    # Operator keys, on keys the teleop bindings leave free. The callbacks run on the keyboard's
    # listener thread, so they only queue the command; the control loop applies it between steps.
    pressed: collections.deque[str] = collections.deque()
    for key in OPERATOR_KEYS:
        keyboard.add_callback(key, lambda down, key=key: down and pressed.append(key))
    # Intentionally do NOT bind 'x' to env.reset(): on a real robot that
    # would reset the sim scene under a live arm; INIT (i+Enter) is the way home.

    try:
        while True:
            while pressed:
                key = pressed.popleft()
                if key == "k":
                    raise KeyboardInterrupt
                OPERATOR_KEYS[key](dep)
            action = keyboard.get_command()
            action_dict = {
                "ee_controller": action[:6].unsqueeze(0).repeat(env.num_envs, 1).to(env.device),
                "gripper_controller": action[6:].unsqueeze(0).repeat(env.num_envs, 1).to(env.device),
            }
            # ``FrankaPandaDeployment.step`` (a hold until START) resyncs the sim entity's qpos
            # from the wrapper's cached real state (no network call), then
            # runs action_manager.compute against that real pose and sends
            # the IK output to the JointPosition controller plus the
            # gripper width target.
            dep.step(action_dict)
            # Drive the sim with the same action so the viewer shows what
            # the policy commanded.  The sim drifts slightly from real
            # during scene.step inside this call, but the next dep.step's
            # resync undoes it before the next action computation.
            env.step(action_dict)
    except KeyboardInterrupt:
        en.logger.info("[teleop] KILL / Ctrl-C received, shutting down deployment.")
    finally:
        dep.close()


if __name__ == "__main__":
    main()
