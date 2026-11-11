"""Run the Humanoid-GPT motion-tracker (MLP or Transformer) on a kinematic reference.

---
tier: policy
capabilities: [task.mimic, robot.g1, deploy.real_robot]
hardware: cpu
optional_deps: [onnx]
run: uv run python examples/policies/humanoid_gpt/main.py --vis
---

Reuses the bundled gangnam-style 50 Hz sample as the reference; the 136-dim obs +
action transform are reproduced in PyTorch from eden state + a ``MotionCommand``.
The checkpoint resolves from HF ``Kashu7100/eden_policies`` via :class:`PolicyBase`
— Transformer by default, MLP with ``--policy-type mlp``; ``--ckpt-path`` overrides
with a local file. See https://github.com/GalaxyGeneralRobotics/Humanoid-GPT.

Usage::

    python examples/policies/humanoid_gpt/main.py --vis            # viewer (Transformer)
    python examples/policies/humanoid_gpt/main.py --policy-type mlp --vis  # MLP
    python examples/policies/humanoid_gpt/main.py --debug-vis      # + ghost reference
    python examples/policies/humanoid_gpt/main.py --record         # write mp4
    python examples/policies/humanoid_gpt/main.py --deploy         # real Unitree G1
"""

from __future__ import annotations

import argparse
import os

import torch
from tqdm import tqdm

import eden as en
import genesis as gs
from eden.envs.base import RLEnvBase
from eden.extensions.policies.humanoid_gpt.policy import HumanoidGptPolicy
from eden.options import (
    GHOST_DEFAULTS,
    ActionManagerOptions,
    CommandManagerOptions,
    EnvOptions,
    ObservationManagerOptions,
)
from eden.options.camera import CameraOptions, CamerasOptions

_HERE = os.path.dirname(__file__)

_DEFAULT_MOTION = os.path.join(
    _HERE, "..", "..", "projects", "mimic", "beyondmimic", "sample", "G1_gangnam_style_V01_bvh_60hz.npz"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion-file", default=_DEFAULT_MOTION, help="Path to a 50 Hz motion npz")
    parser.add_argument(
        "--ckpt-path",
        default=None,
        help="Override the active ONNX checkpoint with a local file (defaults to the HF-resolved bundle)",
    )
    parser.add_argument(
        "--policy-type",
        default="transformer",
        choices=["mlp", "transformer"],
        help="Network architecture of the checkpoint (default: transformer)",
    )
    parser.add_argument("-c", "--cpu", action="store_true", help="Use CPU backend")
    parser.add_argument("-v", "--vis", action="store_true", help="Show viewer")
    parser.add_argument("-r", "--record", action="store_true", help="Record video")
    parser.add_argument("-d", "--debug-vis", action="store_true", help="Render a ghost robot of the reference motion")
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
        scene_options = en.scenes.Empty(robot=en.robots.G1Dofs29(), ghost_robot=en.robots.G1Dofs29(**GHOST_DEFAULTS))
        ghost_entity_name = "ghost_robot"
    else:
        scene_options = en.scenes.Empty(robot=en.robots.G1Dofs29())
        ghost_entity_name = ""

    if args.ckpt_path is not None:
        HumanoidGptPolicy.use_local_checkpoint(args.ckpt_path)

    env = RLEnvBase(
        env_options=EnvOptions(
            num_envs=1,
            episode_length_s=1e9,
            **HumanoidGptPolicy.env_options(),
        ),
        scene_options=scene_options,
        command_options=CommandManagerOptions(
            **HumanoidGptPolicy.command_options(
                motion_file=args.motion_file,
                debug_vis=args.debug_vis,
                ghost_entity_name=ghost_entity_name,
                sampling_mode="start",
            )
        ),
        action_options=ActionManagerOptions(**HumanoidGptPolicy.action_options()),
        observation_options=ObservationManagerOptions(
            **HumanoidGptPolicy.observation_options(policy_type=args.policy_type)
        ),
        cameras_options=(
            CamerasOptions(
                rec=CameraOptions(
                    cam_pos=(3.0, 1.5, 1.4),
                    cam_lookat=(0.0, 0.0, 0.8),
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

    policy = HumanoidGptPolicy(env=env, options=HumanoidGptPolicy.configure(policy_type=args.policy_type))

    cmd = env.command_manager.get_term("motion")
    obs, _ = env.reset()
    policy.reset()

    n_steps = int(cmd.motion.time_step_total) - 1
    if args.steps is not None:
        n_steps = min(n_steps, args.steps)

    if args.record:
        out_path = os.path.join(_HERE, "humanoid_gpt_run.mp4")
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
    """Run Humanoid-GPT on a real Unitree G1 via the deployment pipeline.

    ``UnitreeG1Deployment`` swaps ``env.entities["robot"]`` for a ``RobotStateEntity``
    backed by the real IMU + encoders, which the policy reads in ``step()``. Headless
    (viewer/ghost dropped). ``auto_yaw_align`` frames ``yaw_cmd`` in the robot's start
    heading; ``xy_cmd`` reads ``robot.get_pos()`` — open-loop on hardware without an
    external base-position estimate (joint + heading tracking unaffected).
    """
    from eden.extensions.deployment.unitree_g1 import UnitreeG1Deployment
    from eden.extensions.deployment.utils.republisher import publish_rate_configure_kwargs

    if args.ckpt_path is not None:
        HumanoidGptPolicy.use_local_checkpoint(args.ckpt_path)

    env = RLEnvBase(
        env_options=EnvOptions(
            num_envs=1,
            episode_length_s=1e9,
            **HumanoidGptPolicy.env_options(),
        ),
        scene_options=en.scenes.Empty(robot=en.robots.G1Dofs29()),
        command_options=CommandManagerOptions(
            **HumanoidGptPolicy.command_options(
                motion_file=args.motion_file,
                debug_vis=False,
                ghost_entity_name="",
                sampling_mode="start",
            )
        ),
        action_options=ActionManagerOptions(**HumanoidGptPolicy.action_options()),
        observation_options=ObservationManagerOptions(
            **HumanoidGptPolicy.observation_options(policy_type=args.policy_type)
        ),
        show_viewer=False,
        eval_mode=True,
    )
    env.build()

    # Construct the policy before the deployment swap so __init__ checks the robot's DOFs against
    # the sim entity (RobotStateEntity can't). dep.reset() restarts the env's observation history.
    policy = HumanoidGptPolicy(env=env, options=HumanoidGptPolicy.configure(policy_type=args.policy_type))

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

    n_steps = int(cmd.motion.time_step_total) - 1
    if args.steps is not None:
        n_steps = min(n_steps, args.steps)

    for _ in tqdm(range(n_steps)):
        with torch.no_grad():
            action = policy.step(obs)
        obs = dep.step(action)


if __name__ == "__main__":
    main()
