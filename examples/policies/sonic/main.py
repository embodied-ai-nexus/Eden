"""Drive the GEAR-SONIC universal-token policy (g1 mode) on a kinematic reference.

---
tier: policy
capabilities: [task.mimic, robot.g1, deploy.real_robot]
hardware: cpu
optional_deps: [onnx]
run: uv run python examples/policies/sonic/main.py --vis
---

The actor is split into two ONNX graphs in the release bundle
(``model_encoder.onnx`` + ``model_decoder.onnx``); both are downloaded on first
use from the eden_policies HF mirror. Pass ``--encoder-path`` /
``--decoder-path`` to override with a local copy (e.g. the NVIDIA release).

The reference motion defaults to the BeyondMimic gangnam-style sample that
ships with Eden, since it is in the expected MJCF DOF order and includes a
``pelvis`` body — both required by ``MotionCommand`` with
``anchor_link_name="pelvis"``.

Usage::

    python examples/policies/sonic/main.py --vis            # viewer + tracking
    python examples/policies/sonic/main.py --record            # write mp4
    python examples/policies/sonic/main.py --debug-vis      # add ghost ref robot
    python examples/policies/sonic/main.py --deploy         # real Unitree G1
"""

from __future__ import annotations

import argparse
import os

import torch
from tqdm import tqdm

import eden as en
import genesis as gs
from eden.envs.base import RLEnvBase
from eden.extensions.policies.sonic.policy import SonicPolicy
from eden.options import (
    GHOST_DEFAULTS,
    ActionManagerOptions,
    CommandManagerOptions,
    EnvOptions,
    ObservationManagerOptions,
)
from eden.options.camera import CameraOptions, CamerasOptions


_DEFAULT_MOTION = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "projects",
    "mimic",
    "beyondmimic",
    "sample",
    "G1_gangnam_style_V01_bvh_60hz.npz",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--motion-file",
        default=_DEFAULT_MOTION,
        help="Genesis-BFS-ordered motion npz (joint_pos/joint_vel in MJCF order).",
    )
    parser.add_argument(
        "--encoder-path",
        default=None,
        help="Override the SONIC encoder ONNX (defaults to eden_policies asset).",
    )
    parser.add_argument(
        "--decoder-path",
        default=None,
        help="Override the SONIC decoder ONNX (defaults to eden_policies asset).",
    )
    parser.add_argument("-c", "--cpu", action="store_true", help="Use CPU backend")
    parser.add_argument("-v", "--vis", action="store_true", help="Show viewer")
    parser.add_argument("-r", "--record", action="store_true", help="Record video")
    parser.add_argument(
        "-d",
        "--debug-vis",
        action="store_true",
        help="Render a ghost robot of the reference motion",
    )
    parser.add_argument("-s", "--steps", type=int, default=None, help="Number of steps to run")
    parser.add_argument(
        "--report-every",
        type=int,
        default=200,
        help="Print mean / max body-position error vs reference every N steps",
    )
    parser.add_argument(
        "--deploy",
        action="store_true",
        help="Run the policy on a real Unitree G1 via UnitreeG1Deployment instead of in sim.",
    )
    parser.add_argument(
        "--deploy-kp-kd-scale",
        type=float,
        default=1.0,
        help="Scale applied to kp/kd before sending to the G1 SDK. Try 0.2 / 0.5 / 0.8 first.",
    )
    parser.add_argument(
        "--publish-rate",
        type=float,
        default=None,
        help="Hz at which the held wire command is re-sent between control ticks (default: the "
        "backend's 500 Hz, Unitree's own low-level rate). The G1's parallel ankle/waist joints "
        "close their PD once per received packet. Pass 0 for one packet per control tick -- the "
        "A/B against that default.",
    )
    args = parser.parse_args()

    en.init(backend=gs.cpu if args.cpu else gs.gpu)

    if args.deploy:
        deploy(args)
        return

    if args.debug_vis:
        scene_options = en.scenes.Empty(
            robot=SonicPolicy.robot_options(), ghost_robot=en.robots.G1Dofs29(**GHOST_DEFAULTS)
        )
        ghost_entity_name = "ghost_robot"
    else:
        scene_options = en.scenes.Empty(robot=SonicPolicy.robot_options())
        ghost_entity_name = ""

    if args.encoder_path is not None:
        SonicPolicy.use_local_checkpoint(args.encoder_path)
    if args.decoder_path is not None:
        SonicPolicy.use_local_checkpoint(args.decoder_path, checkpoint_attr="decoder_checkpoint_path")

    env = RLEnvBase(
        env_options=EnvOptions(
            num_envs=1,
            episode_length_s=1e9,
            **SonicPolicy.env_options(),
        ),
        scene_options=scene_options,
        command_options=CommandManagerOptions(
            **SonicPolicy.command_options(
                motion_file=args.motion_file,
                debug_vis=args.debug_vis,
                ghost_entity_name=ghost_entity_name,
                sampling_mode="start",
            )
        ),
        action_options=ActionManagerOptions(**SonicPolicy.action_options()),
        observation_options=ObservationManagerOptions(**SonicPolicy.observation_options()),
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

    policy = SonicPolicy(env=env, options=SonicPolicy.configure())

    cmd = env.command_manager.get_term("motion")
    obs, _ = env.reset()
    policy.reset()

    n_steps = int(cmd.motion.time_step_total) - 1
    if args.steps is not None:
        n_steps = min(n_steps, args.steps)

    if args.record:
        out_path = os.path.join(os.path.dirname(__file__), "sonic_run.mp4")
        env.cameras["rec"].start_recording(out_path)

    err_running_max = 0.0
    err_running_sum = 0.0
    err_count = 0
    body_idx = cmd.links_idx
    for step in tqdm(range(n_steps)):
        action = policy.step(obs)
        obs, _, _, _, _ = env.step(action)

        ref_pos = cmd.links_pos_relative_w[0]
        cur_pos = env.entities["robot"].get_links_pos(ls_idx_local=body_idx)[0]
        diff = (ref_pos - cur_pos).norm(dim=-1)
        err_running_max = max(err_running_max, float(diff.max().item()))
        err_running_sum += float(diff.mean().item())
        err_count += 1

        if args.record:
            env.cameras["rec"].render_rgb()
        if args.report_every and (step + 1) % args.report_every == 0:
            print(
                f"  step={step + 1:5d}  mean_link_err={err_running_sum / err_count:.4f}m  "
                f"max_link_err={err_running_max:.4f}m"
            )

    if args.record:
        env.cameras["rec"].stop_recording()
        print(f"Recording written to {out_path}")

    print(
        f"\nFinal: {err_count} steps  mean_link_err={err_running_sum / max(err_count, 1):.4f}m  "
        f"max_link_err={err_running_max:.4f}m"
    )


def deploy(args: argparse.Namespace) -> None:
    """Run SONIC on a real Unitree G1 via the deployment pipeline.

    The policy reads kinematics directly from ``env.entities["robot"]`` inside
    its ``step()``; ``UnitreeG1Deployment`` swaps that entity for a
    ``RobotStateEntity`` whose ``get_dofs_pos`` / ``get_dofs_vel`` /
    ``get_ang(frame="body")`` / ``get_quat()`` calls reflect the real-robot
    reading (IMU + motor encoders). SONIC's pelvis anchor maps directly to the
    IMU base, so no extra link-FK queries are needed during deploy.

    Notes
    -----
    - Headless: viewer / recording flags are ignored.
    - Ghost reference is dropped (no point next to a real robot).
    - ``auto_yaw_align`` is enabled; :meth:`SonicPolicy.step` still captures
      its own per-episode heading offset against the motion, so the two
      compose cleanly (motion-anchor obs ends up in a yaw-aligned world frame
      regardless of the robot's start heading).
    """
    from eden.extensions.deployment.unitree_g1 import UnitreeG1Deployment
    from eden.extensions.deployment.utils.republisher import publish_rate_configure_kwargs

    if args.encoder_path is not None:
        SonicPolicy.use_local_checkpoint(args.encoder_path)
    if args.decoder_path is not None:
        SonicPolicy.use_local_checkpoint(args.decoder_path, checkpoint_attr="decoder_checkpoint_path")

    scene_options = en.scenes.Empty(robot=SonicPolicy.robot_options())
    ghost_entity_name = ""

    env = RLEnvBase(
        env_options=EnvOptions(
            num_envs=1,
            episode_length_s=1e9,
            **SonicPolicy.env_options(),
        ),
        scene_options=scene_options,
        command_options=CommandManagerOptions(
            **SonicPolicy.command_options(
                motion_file=args.motion_file,
                debug_vis=False,
                ghost_entity_name=ghost_entity_name,
                sampling_mode="start",
            )
        ),
        action_options=ActionManagerOptions(**SonicPolicy.action_options()),
        observation_options=ObservationManagerOptions(**SonicPolicy.observation_options()),
        show_viewer=False,
        eval_mode=True,
    )
    env.build()

    # Construct the policy *before* the deployment swap so the policy's
    # __init__ resolves dof / pelvis link indices against the sim entity
    # (RobotStateEntity does not expose ``find_named_dofs_idx_local``).
    policy = SonicPolicy(env=env, options=SonicPolicy.configure())

    dep = UnitreeG1Deployment(
        env,
        UnitreeG1Deployment.configure(
            entity_name="robot",
            sync=True,
            connect_timeout_s=10.0,
            deploy_kp_kd_scale=args.deploy_kp_kd_scale,
            auto_yaw_align=True,
            **publish_rate_configure_kwargs(args.publish_rate),
        ),
    )
    dep.connect()

    cmd = env.command_manager.get_term("motion")
    obs = dep.reset()
    # Re-init history ring buffers + heading offset against the fresh
    # post-reset state. Critical: the robot then holds and INITs for many
    # seconds before the operator's START, so the history buffers (if not
    # cleared) would carry stale frames.
    policy.reset()

    n_steps = int(cmd.motion.time_step_total) - 1
    if args.steps is not None:
        n_steps = min(n_steps, args.steps)

    for _ in tqdm(range(n_steps)):
        with torch.no_grad():
            action = policy.step(obs)
        obs = dep.step(action)


if __name__ == "__main__":
    main()
