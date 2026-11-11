"""Drive the HOMIE frozen walk/stand tracker on a Unitree G1 in Genesis.

---
tier: policy
capabilities: [task.locomotion, robot.g1]
hardware: cpu
optional_deps: []
run: uv run python examples/policies/homie/main.py --dummy --steps 50
---

HOMIE is a velocity- + height-command lower-body locomotion policy. This entry
point builds a flat-ground G1 environment, sets a constant locomotion command,
and rolls out the frozen tracker. The 14 arm DOFs are held at the HOMIE default
pose by the policy.

The released ``model_walk.pt`` / ``model_stand.pt`` checkpoints are *not*
mirrored on the eden_policies hub. Provide local paths::

    python examples/policies/homie/main.py --walk-model /path/model_walk.pt \
        --stand-model /path/model_stand.pt --vis --vx 0.5

Without trained checkpoints, ``--dummy`` generates architecture-correct random
weights so the full inference pipeline (observation construction, history,
walk/stand switching, action → target transform, sim step) can be smoke-tested.
A random-weight policy will NOT produce stable walking — it only validates the
plumbing.

With ``--estimate-lin-vel``, the deploy-time base-linear-velocity estimator
(contact-aided Kalman filter, see
``eden/extensions/deployment/utils/lin_vel_estimator.py``) runs alongside the
rollout on **hardware-like signals only** (synthetic IMU specific force, joint
q/dq/tau, FK) and is compared against the simulator's ground-truth base
velocity — per-axis RMSE is printed at the end and ``--plot`` writes an
est-vs-GT trace PNG next to this script.

Usage::

    python examples/policies/homie/main.py --dummy --steps 50         # plumbing smoke test
    python examples/policies/homie/main.py --walk-model W --stand-model S --vis --vx 0.5
    python examples/policies/homie/main.py --vx 0.5 --estimate-lin-vel --plot
"""

from __future__ import annotations

import argparse
import os
import tempfile

import numpy as np
import torch
from tqdm import tqdm

import eden as en
import genesis as gs
from genesis.utils.geom import quat_to_R
from eden.envs.base import RLEnvBase
from eden.extensions.deployment.utils.lin_vel_estimator import LinVelKalmanEstimator
from eden.extensions.policies.homie.modules import HOMIE_INIT_DICT, HIMActorCritic
from eden.extensions.policies.homie.policy import HomiePolicy
from eden.options import (
    ActionManagerOptions,
    EnvOptions,
    ObservationManagerOptions,
)
from eden.options.camera import CameraOptions, CamerasOptions

_GRAVITY = 9.81


class _LinVelEstProbe:
    """Run the deploy-time velocity KF next to a sim rollout and log est vs GT.

    The estimator only sees what a real deployment would: base quaternion,
    body-frame gyro, an accelerometer signal (synthesized here as the
    finite-difference of the ground-truth world velocity plus gravity, rotated
    into the body frame — i.e. the specific force a noise-free IMU would
    measure), joint velocities and actuator torques. Ground-truth base linear
    velocity is recorded only for comparison.
    """

    def __init__(self, robot, dt: float, feet_links) -> None:
        self._robot = robot
        self._dt = dt
        self._est = LinVelKalmanEstimator(robot, feet_links=feet_links, dt=dt)
        self._prev_v: np.ndarray | None = None
        self.gt: list[np.ndarray] = []
        self.est: list[np.ndarray] = []
        self.contacts: list[np.ndarray] = []

    def step(self) -> None:
        robot = self._robot
        quat = robot.get_quat()[0].double().cpu().numpy()
        R = quat_to_R(quat)
        gyro_body = R.T @ robot.get_ang(frame="world")[0].double().cpu().numpy()
        v_world = robot.get_vel()[0].double().cpu().numpy()
        dq = robot.get_dofs_vel()[0].double().cpu().numpy()
        tau = robot.get_dofs_control_force()[0].double().cpu().numpy()

        a_world = np.zeros(3) if self._prev_v is None else (v_world - self._prev_v) / self._dt
        self._prev_v = v_world
        accel_body = R.T @ (a_world + np.array([0.0, 0.0, _GRAVITY]))

        v_est = self._est.update(quat, gyro_body, accel_body, dq, tau, yaw_tag=0.0)
        self.gt.append(v_world)
        self.est.append(v_est)
        self.contacts.append(self._est.contacts.copy())

    def report(self, plot_path: str | None = None) -> None:
        gt, est = np.asarray(self.gt), np.asarray(self.est)
        err = est - gt
        rmse = np.sqrt((err**2).mean(axis=0))
        print(
            f"\n[lin_vel_est] {len(gt)} steps | GT mean v={gt.mean(axis=0).round(3)} m/s | "
            f"RMSE xyz={rmse.round(3)} m/s | max|err|={np.abs(err).max(axis=0).round(3)} m/s | "
            f"stance fraction={np.asarray(self.contacts).mean():.2f}"
        )
        if plot_path is None:
            return
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("[lin_vel_est] matplotlib not installed; skipping --plot")
            return
        t = np.arange(len(gt)) * self._dt
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        for i, (ax, label) in enumerate(zip(axes, ("vx", "vy", "vz"))):
            ax.plot(t, gt[:, i], label="ground truth", lw=1.5)
            ax.plot(t, est[:, i], label="estimated", lw=1.0, alpha=0.85)
            ax.set_ylabel(f"{label} [m/s]")
            ax.grid(alpha=0.3)
        axes[0].legend(loc="upper right")
        axes[0].set_title(f"Base linear velocity: KF estimate vs ground truth (RMSE {rmse.round(3)})")
        axes[-1].set_xlabel("time [s]")
        fig.tight_layout()
        fig.savefig(plot_path, dpi=120)
        print(f"[lin_vel_est] plot written to {plot_path}")


def _make_dummy_checkpoint(path: str, seed: int) -> None:
    """Write a random-weight HIMActorCritic state dict to ``path``.

    Used by ``--dummy`` to exercise the load + inference path without the
    released checkpoints. The output does not walk — it is for plumbing only.
    """
    torch.manual_seed(seed)
    model = HIMActorCritic(**HOMIE_INIT_DICT)
    torch.save(model.state_dict(), path)


def main() -> None:
    """Parse args, build the env + policy, and roll out the tracker."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--walk-model", default=None, help="Path to model_walk.pt")
    parser.add_argument("--stand-model", default=None, help="Path to model_stand.pt")
    parser.add_argument(
        "--dummy",
        action="store_true",
        help="Generate random-weight walk/stand checkpoints (plumbing smoke test only).",
    )
    parser.add_argument("-c", "--cpu", action="store_true", help="Use CPU backend")
    parser.add_argument("-v", "--vis", action="store_true", help="Show viewer")
    parser.add_argument("-r", "--record", action="store_true", help="Record video")
    parser.add_argument("--vx", type=float, default=0.0, help="Forward velocity command [m/s]")
    parser.add_argument("--vy", type=float, default=0.0, help="Lateral velocity command [m/s]")
    parser.add_argument("--wz", type=float, default=0.0, help="Yaw-rate command [rad/s]")
    parser.add_argument(
        "--height",
        type=float,
        default=0.0,
        help="Base-height delta about HOMIE default (0.74 m); negative crouches.",
    )
    parser.add_argument("-s", "--steps", type=int, default=1000, help="Number of steps to run")
    parser.add_argument(
        "--estimate-lin-vel",
        action="store_true",
        help="Run the deploy-time base-velocity Kalman estimator on hardware-like "
        "signals alongside the rollout and report estimated-vs-ground-truth error.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="With --estimate-lin-vel: save an est-vs-GT velocity trace PNG next to this script.",
    )
    args = parser.parse_args()

    en.init(backend=gs.cpu if args.cpu else gs.gpu)

    # Resolve checkpoints. By default HomiePolicy downloads the released
    # walk/stand models from the eden_policies hub; --walk-model/--stand-model
    # override with local paths, and --dummy generates random weights.
    tmpdir: tempfile.TemporaryDirectory | None = None
    if args.dummy:
        tmpdir = tempfile.TemporaryDirectory()
        walk_path = os.path.join(tmpdir.name, "model_walk.pt")
        stand_path = os.path.join(tmpdir.name, "model_stand.pt")
        _make_dummy_checkpoint(walk_path, seed=0)
        _make_dummy_checkpoint(stand_path, seed=1)
        HomiePolicy.walk_model_path = walk_path
        HomiePolicy.stand_model_path = stand_path
        HomiePolicy.walk_md5_hash = HomiePolicy.stand_md5_hash = None  # custom weights
        print(f"[dummy] random walk/stand checkpoints in {tmpdir.name} (plumbing only — will not walk)")
    else:
        if args.walk_model is not None:
            HomiePolicy.walk_model_path = args.walk_model
            HomiePolicy.walk_md5_hash = None  # custom checkpoint, drop the pinned hash
        if args.stand_model is not None:
            HomiePolicy.stand_model_path = args.stand_model
            HomiePolicy.stand_md5_hash = None

    scene_options = en.scenes.Empty(robot=HomiePolicy.robot_options())

    env = RLEnvBase(
        env_options=EnvOptions(
            num_envs=1,
            episode_length_s=1e9,
            **HomiePolicy.env_options(),
        ),
        scene_options=scene_options,
        action_options=ActionManagerOptions(**HomiePolicy.action_options()),
        observation_options=ObservationManagerOptions(**HomiePolicy.observation_options()),
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

    policy = HomiePolicy(env=env, options=HomiePolicy.configure())
    policy.set_command(torch.tensor([args.vx, args.vy, args.wz, args.height]))

    obs, _ = env.reset()
    policy.reset()

    if args.record:
        out_path = os.path.join(os.path.dirname(__file__), "homie_run.mp4")
        env.cameras["rec"].start_recording(out_path)

    robot = env.entities["robot"]
    probe = None
    if args.estimate_lin_vel:
        control_dt = env.env_options.sim_dt * env.env_options.decimation
        # The robot declares its own feet, exactly as the deployment reads them.
        probe = _LinVelEstProbe(robot, dt=control_dt, feet_links=scene_options.robot.feet_links_name)

    base_z0 = float(robot.get_pos()[0, 2].item())
    for step in tqdm(range(args.steps)):
        action = policy.step(obs)
        obs, _, _, _, _ = env.step(action)
        if probe is not None:
            probe.step()
        if args.record:
            env.cameras["rec"].render_rgb()

    base = robot.get_pos()[0]
    print(
        f"\nFinal base pos: x={base[0]:.3f} y={base[1]:.3f} z={base[2]:.3f}m "
        f"(z drift from start {base[2] - base_z0:+.3f}m)"
    )

    if probe is not None:
        plot_path = os.path.join(os.path.dirname(__file__), "homie_lin_vel_est.png") if args.plot else None
        probe.report(plot_path)

    if args.record:
        env.cameras["rec"].stop_recording()
        print(f"Recording written to {out_path}")

    if tmpdir is not None:
        tmpdir.cleanup()


if __name__ == "__main__":
    main()
