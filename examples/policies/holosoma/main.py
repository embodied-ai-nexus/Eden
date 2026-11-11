"""Drive the Holosoma velocity-command loco policy on a Unitree G1 in Genesis.

---
tier: policy
capabilities: [task.locomotion, robot.g1, deploy.real_robot]
hardware: cpu
optional_deps: [onnx]
run: uv run python examples/policies/holosoma/main.py --vis --vx 0.4
---

Builds a flat-ground G1 environment, sets a constant velocity command, and
rolls out the frozen Holosoma ONNX locomotion policy. The policy controls all
29 DOFs (legs + waist + arms).

``HolosomaLocomotionPolicy`` defaults its checkpoint to the hub path
``holosoma/ppo_g1_29dof.onnx`` (Kashu7100/eden_policies). Until that asset is
uploaded, point ``--model`` at a local ONNX — this script defaults to the model
bundled under ``references/holosoma`` when present.

Usage::

    python examples/policies/holosoma/main.py --vis --vx 0.4
    python examples/policies/holosoma/main.py --model /path/to/ppo_g1_29dof.onnx --vis --vx 0.4
    python examples/policies/holosoma/main.py --steps 50           # plumbing smoke test
    python examples/policies/holosoma/main.py --deploy --vx 0.4        # real Unitree G1
"""

from __future__ import annotations

import argparse
import os

import torch
from tqdm import tqdm

import eden as en
import genesis as gs
from eden.envs.base import RLEnvBase
from eden.extensions.policies.holosoma.policy import HolosomaLocomotionPolicy
from eden.options import (
    ActionManagerOptions,
    EnvOptions,
    ObservationManagerOptions,
)
from eden.options.camera import CameraOptions, CamerasOptions

# Model bundled under references/ (relative to the repo root). Used as the
# default when --model is not given and the file exists.
_BUNDLED_MODEL = os.path.normpath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
        "references",
        "holosoma",
        "src",
        "holosoma_inference",
        "holosoma_inference",
        "models",
        "loco",
        "g1_29dof",
        "ppo_g1_29dof.onnx",
    )
)


def main() -> None:
    """Parse args, build the env + policy, and roll out the loco policy."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default=_BUNDLED_MODEL if os.path.isfile(_BUNDLED_MODEL) else None,
        help="Path to a loco ONNX (ppo/fastsac). Defaults to the bundled references model if present.",
    )
    parser.add_argument("-c", "--cpu", action="store_true", help="Use CPU backend")
    parser.add_argument("-v", "--vis", action="store_true", help="Show viewer")
    parser.add_argument("-r", "--record", action="store_true", help="Record video")
    parser.add_argument("--vx", type=float, default=0.0, help="Forward velocity command [m/s]")
    parser.add_argument("--vy", type=float, default=0.0, help="Lateral velocity command [m/s]")
    parser.add_argument("--wz", type=float, default=0.0, help="Yaw-rate command [rad/s]")
    parser.add_argument("-s", "--steps", type=int, default=1000, help="Number of steps to run")
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

    # Resolve the checkpoint. By default HolosomaLocomotionPolicy downloads from
    # the eden_policies hub; --model (or the bundled references model) overrides
    # with a local path and drops the pinned md5.
    if args.model is not None:
        HolosomaLocomotionPolicy.use_local_checkpoint(args.model)

    scene_options = en.scenes.Empty(robot=HolosomaLocomotionPolicy.robot_options())

    env = RLEnvBase(
        env_options=EnvOptions(
            num_envs=1,
            episode_length_s=1e9,
            **HolosomaLocomotionPolicy.env_options(),
        ),
        scene_options=scene_options,
        action_options=ActionManagerOptions(**HolosomaLocomotionPolicy.action_options()),
        observation_options=ObservationManagerOptions(**HolosomaLocomotionPolicy.observation_options()),
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

    policy = HolosomaLocomotionPolicy(env=env, options=HolosomaLocomotionPolicy.configure())
    policy.set_command(torch.tensor([args.vx, args.vy, args.wz]))

    obs, _ = env.reset()
    policy.reset()

    if args.record:
        out_path = os.path.join(os.path.dirname(__file__), "holosoma_run.mp4")
        env.cameras["rec"].start_recording(out_path)

    robot = env.entities["robot"]
    base_z0 = float(robot.get_pos()[0, 2].item())
    for _ in tqdm(range(args.steps)):
        action = policy.step(obs)
        obs, _, _, _, _ = env.step(action)
        if args.record:
            env.cameras["rec"].render_rgb()

    base = robot.get_pos()[0]
    print(
        f"\nFinal base pos: x={base[0]:.3f} y={base[1]:.3f} z={base[2]:.3f}m "
        f"(z drift from start {base[2] - base_z0:+.3f}m)"
    )

    if args.record:
        env.cameras["rec"].stop_recording()
        print(f"Recording written to {out_path}")


def deploy(args: argparse.Namespace) -> None:
    """Run the Holosoma loco policy on a real Unitree G1 via the deployment pipeline.

    The policy reads proprioception directly from ``env.entities["robot"]``
    inside its ``step()``; ``UnitreeG1Deployment`` swaps that entity for a
    ``RobotStateEntity`` whose ``get_dofs_pos`` / ``get_dofs_vel`` /
    ``get_ang(frame="body")`` / ``get_quat()`` calls reflect the real-robot
    reading (IMU + motor encoders). The policy's observation is purely
    body-frame (projected gravity, body-rate gyro, velocity command), so it is
    insensitive to the world-frame yaw alignment.

    Notes
    -----
    - Headless: viewer / recording flags are ignored.
    - The velocity command (``--vx/--vy/--wz``) is held constant for the whole
      run; a ~zero command makes the policy stand.
    - PD gains come from ``HolosomaLocomotionPolicy.robot_options()`` (identical to the released
      ONNX kp/kd metadata), scaled by ``--deploy-kp-kd-scale`` before being
      sent to the SDK.
    """
    from eden.extensions.deployment.unitree_g1 import UnitreeG1Deployment
    from eden.extensions.deployment.utils.republisher import publish_rate_configure_kwargs

    if args.model is not None:
        HolosomaLocomotionPolicy.use_local_checkpoint(args.model)

    scene_options = en.scenes.Empty(robot=HolosomaLocomotionPolicy.robot_options())

    env = RLEnvBase(
        env_options=EnvOptions(
            num_envs=1,
            episode_length_s=1e9,
            **HolosomaLocomotionPolicy.env_options(),
        ),
        scene_options=scene_options,
        action_options=ActionManagerOptions(**HolosomaLocomotionPolicy.action_options()),
        observation_options=ObservationManagerOptions(**HolosomaLocomotionPolicy.observation_options()),
        show_viewer=False,
        eval_mode=True,
    )
    env.build()

    # Construct the policy *before* the deployment swap so the policy's
    # __init__ resolves dof indices against the sim entity
    # (RobotStateEntity does not expose ``find_named_dofs_idx_local``).
    policy = HolosomaLocomotionPolicy(env=env, options=HolosomaLocomotionPolicy.configure())
    policy.set_command(torch.tensor([args.vx, args.vy, args.wz]))

    dep = UnitreeG1Deployment(
        env,
        UnitreeG1Deployment.configure(
            entity_name="robot",
            sync=True,
            connect_timeout_s=10.0,
            deploy_kp_kd_scale=args.deploy_kp_kd_scale,
            **publish_rate_configure_kwargs(args.publish_rate),
        ),
    )
    dep.connect()

    obs = dep.reset()
    # Re-init prev-action and gait-phase state against the fresh post-reset
    # state. Critical: the robot then holds and INITs for many seconds before
    # the operator's START, so stale phase / last-action buffers would
    # otherwise leak into the first policy step.
    policy.reset()

    try:
        for _ in tqdm(range(args.steps)):
            with torch.no_grad():
                action = policy.step(obs)
            obs = dep.step(action)
    finally:
        # Soft shutdown (damping ramp) before dropping the channel.
        dep.close()


if __name__ == "__main__":
    main()
