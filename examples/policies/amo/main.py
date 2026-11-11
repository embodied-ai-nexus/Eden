"""Drive the AMO whole-body G1 control policy in Genesis with live keyboard commands.

---
tier: policy
capabilities: [task.locomotion, robot.g1, teleop.keyboard]
hardware: cpu
optional_deps: []
run: uv run python examples/policies/amo/main.py --vis
---

The 8-D command [vx, yaw, vy, height, torso_yaw, torso_pitch, torso_roll, arm_toggle]
is set live from the keyboard (pynput); see ``AMOKeyController`` for the bindings.
"""

from argparse import ArgumentParser
import torch
import genesis as gs

import eden as en
from eden.envs.base import RLEnvBase
from eden.options import (
    ActionManagerOptions,
    CommandManagerOptions,
    EnvOptions,
    ObservationManagerOptions,
)
from eden.extensions.policies.amo.policy import AMOPolicy
from eden.utils.devices.keyboard import Keyboard, KeyBinding


class AMOKeyController:
    def __init__(self, env):
        self.env = env
        self.cmd_term = env.command_manager.get_term("amo_command")

        # Key bindings to match play_amo.py logic
        # Command vector: [vx, yaw, vy, height, torso_yaw, torso_pitch, torso_roll, arm_toggle]
        # index 0: vx (W/S)
        # index 1: yaw (A/D)
        # index 2: vy (Q/E)
        # index 3: height (Z/X)
        # index 4: torso_yaw (J/U)
        # index 5: torso_pitch (K/I)
        # index 6: torso_roll (L/O)
        # index 7: arm_toggle (T)

        self.bindings = {
            "w": KeyBinding(0, 1, reset_on_release=False),
            "s": KeyBinding(0, -1, reset_on_release=False),
            "a": KeyBinding(1, 1, reset_on_release=False),
            "d": KeyBinding(1, -1, reset_on_release=False),
            "q": KeyBinding(2, 1, reset_on_release=False),
            "e": KeyBinding(2, -1, reset_on_release=False),
            "z": KeyBinding(3, 1, reset_on_release=False),
            "x": KeyBinding(3, -1, reset_on_release=False),
            "j": KeyBinding(4, 1, reset_on_release=False),
            "u": KeyBinding(4, -1, reset_on_release=False),
            "k": KeyBinding(5, 1, reset_on_release=False),
            "i": KeyBinding(5, -1, reset_on_release=False),
            "l": KeyBinding(6, 1, reset_on_release=False),
            "o": KeyBinding(6, -2, reset_on_release=False),
            "t": KeyBinding(7, 1, is_toggle=True),
        }

        self.keyboard = Keyboard(self.bindings)
        self.scales = torch.tensor([0.05, 0.1, 0.05, 0.05, 0.1, 0.05, 0.05, 1.0], device=env.device)

    def update(self):
        # Keyboard readings are on CPU
        readings = self.keyboard.get_command().to(self.env.device)

        cmds = readings * self.scales
        self.cmd_term.commands[0] = cmds
        return cmds


def main():
    parser = ArgumentParser()
    parser.add_argument("-c", "--cpu", action="store_true", help="Use CPU backend instead of GPU")
    parser.add_argument("-v", "--vis", action="store_true", help="Show viewer")
    parser.add_argument("-s", "--steps", type=int, default=-1, help="Number of steps to run (-1 for infinite)")

    args = parser.parse_args()

    en.init(backend=gs.cpu if args.cpu else gs.gpu, debug=True)

    scene_options = en.scenes.Empty(
        robot=AMOPolicy.robot_options(),
    )

    env = RLEnvBase(
        env_options=EnvOptions(num_envs=1, **AMOPolicy.env_options()),
        scene_options=scene_options,
        command_options=CommandManagerOptions(**AMOPolicy.command_options(resampling_time_range=(1e9, 1e9))),
        action_options=ActionManagerOptions(**AMOPolicy.action_options()),
        observation_options=ObservationManagerOptions(**AMOPolicy.observation_options()),
        show_viewer=args.vis,
        eval_mode=True,
    )
    env.build()
    env.config.save_as_json()

    policy = AMOPolicy(env=env, options=AMOPolicy.configure())

    # Setup keyboard controller
    controller = AMOKeyController(env)

    obs, _ = env.reset()

    print("Keyboard control enabled (via pynput).")
    print("Controls:")
    print("  W/S: Forward/Backward velocity")
    print("  A/D: Turn Left/Right")
    print("  Q/E: Left/Right velocity")
    print("  Z/X: Height Up/Down")
    print("  J/U: Torso Yaw")
    print("  K/I: Torso Pitch")
    print("  L/O: Torso Roll")
    print("  T: Toggle Arm Control")

    try:
        step = 0
        while True:
            if args.steps > 0 and step >= args.steps:
                break

            controller.update()

            raw_action = policy.step(obs)
            obs, _, _, _, _ = env.step(raw_action)
            step += 1

    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
