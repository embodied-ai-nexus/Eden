"""Drive the TWIST2 motion-tracking policy (Unitree G1) on an Eden npz motion.

---
tier: policy
capabilities: [task.mimic, robot.g1]
hardware: cpu
optional_deps: [onnx]
run: uv run python examples/policies/twist2/main.py --vis
---

Ports the inference pipeline of
`twist2_mjlab <https://github.com/lzyang2000/twist2_mjlab>`_ into Eden. The
kinematic reference is supplied by the shared beyondmimic ``MotionCommand``
(same path as the SONIC port); the policy builds the 1524-D actor observation
and runs the released ONNX actor, driving Genesis's built-in PD. The robot
spawns in TWIST2's knees-bent pose with the stock G1 gains (which match
mjlab's).

Convert a TWIST2 enriched PKL to the expected ``.npz`` first::

    python examples/policies/twist2/scripts/pkl_to_npz.py \
        --input-file examples/policies/twist2/resources/eyes_japan_greeting_01_hello_aita.pkl \
        -o examples/policies/twist2/resources/greeting.npz

Usage::

    python examples/policies/twist2/main.py --vis            # viewer
    python examples/policies/twist2/main.py --record            # write mp4
    python examples/policies/twist2/main.py --debug-vis      # add ghost ref
"""

from __future__ import annotations

import argparse
import os

from tqdm import tqdm

import eden as en
import genesis as gs
from eden.envs.base import RLEnvBase
from eden.extensions.policies.twist2.policy import Twist2Policy
from eden.options import (
    GHOST_DEFAULTS,
    ActionManagerOptions,
    CommandManagerOptions,
    EnvOptions,
    ObservationManagerOptions,
)
from eden.options.camera import CameraOptions, CamerasOptions

_DEFAULT_MOTION = os.path.join(os.path.dirname(__file__), "resources", "greeting.npz")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion-file", default=_DEFAULT_MOTION, help="Eden npz motion (Genesis link order).")
    parser.add_argument(
        "--onnx-path",
        default=None,
        help="Override the TWIST2 actor ONNX (defaults to the eden_policies HF asset).",
    )
    parser.add_argument("-c", "--cpu", action="store_true", help="Use CPU backend")
    parser.add_argument("-v", "--vis", action="store_true", help="Show viewer")
    parser.add_argument("-r", "--record", action="store_true", help="Record video")
    parser.add_argument("-d", "--debug-vis", action="store_true", help="Render a ghost of the reference motion")
    parser.add_argument("-s", "--steps", type=int, default=None, help="Number of steps to run")
    args = parser.parse_args()

    en.init(backend=gs.cpu if args.cpu else gs.gpu)

    if args.onnx_path is not None:
        Twist2Policy.use_local_checkpoint(args.onnx_path)

    if args.debug_vis:
        scene_options = en.scenes.Empty(
            robot=Twist2Policy.robot_options(),
            ghost_robot=en.robots.G1Dofs29(**GHOST_DEFAULTS),
        )
        ghost_entity_name = "ghost_robot"
    else:
        scene_options = en.scenes.Empty(robot=Twist2Policy.robot_options())
        ghost_entity_name = ""

    env = RLEnvBase(
        env_options=EnvOptions(
            num_envs=1,
            episode_length_s=1e9,
            **Twist2Policy.env_options(),
        ),
        scene_options=scene_options,
        command_options=CommandManagerOptions(
            **Twist2Policy.command_options(
                motion_file=args.motion_file,
                debug_vis=args.debug_vis,
                ghost_entity_name=ghost_entity_name,
                sampling_mode="start",
            )
        ),
        action_options=ActionManagerOptions(**Twist2Policy.action_options()),
        observation_options=ObservationManagerOptions(**Twist2Policy.observation_options()),
        cameras_options=(
            CamerasOptions(
                rec=CameraOptions(
                    cam_pos=(3.0, 0.0, 1.6),
                    cam_lookat=(0.0, 0.0, 0.9),
                    cam_fov=50.0,
                    follow_entity_name="robot",
                )
            )
            if args.record
            else None
        ),
        show_viewer=args.vis,
        eval_mode=True,
    )
    env.build()

    policy = Twist2Policy(env=env, options=Twist2Policy.configure())

    cmd = env.command_manager.get_term("motion")
    obs, _ = env.reset()
    policy.reset()

    n_steps = int(cmd.motion.time_step_total) - 1
    if args.steps is not None:
        n_steps = min(n_steps, args.steps)

    if args.record:
        out_path = os.path.join(os.path.dirname(__file__), "twist2_run.mp4")
        env.cameras["rec"].start_recording(out_path)

    for _ in tqdm(range(n_steps)):
        action = policy.step(obs)
        obs, _, _, _, _ = env.step(action)
        if args.record:
            env.cameras["rec"].render_rgb()

    if args.record:
        env.cameras["rec"].stop_recording()
        print(f"Recording written to {out_path}")


if __name__ == "__main__":
    main()
