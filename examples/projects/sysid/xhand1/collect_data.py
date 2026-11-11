"""Collect real-XHand1 sysid trajectories via the deployment interface.

---
tier: project
capabilities: [task.sysid, robot.xhand, deploy.real_robot]
hardware: real-robot
optional_deps: []
run: uv run python examples/projects/sysid/xhand1/collect_data.py --side right --excitation chirp --seconds 30 --output data/xhand_chirp.npz
---

Usage
-----

Identification trace (chirp on all 12 joints, 30 s):

    uv run python examples/projects/sysid/xhand1/collect_data.py --side right \
        --excitation chirp --seconds 30 --output data/xhand_chirp.npz

Verification trace (held-out PRBS, 20 s):

    uv run python examples/projects/sysid/xhand1/collect_data.py --side right \
        --excitation prbs --seconds 20 --output data/xhand_prbs.npz

Safety
------
The excitation is centered on each joint's URDF midpoint, clamped to
``±safety_limit`` per joint (default 0.3 rad). Amplitude ratios are conservative
defaults tuned for the XHand1 so the fingers stay inside their workspace.
"""

from __future__ import annotations

import pathlib
import sys
from typing import Literal

import numpy as np
from sim_twin_config import make_argparser, make_xhand1_sim_config

import eden as en
from eden.extensions.deployment.base import DEPLOYMENT_REGISTRY
from eden.extensions.deployment.robotera_xhand import RoboTeraXHandDeployment
from eden.extensions.sysid import (
    ChirpExcitation,
    DeploymentRecorder,
    PRBSExcitation,
)
from eden.options.extensions.deployment import DeploymentOptions

DEFAULT_SAFETY_RAD = 0.30
DEFAULT_CHIRP_DOFS_RANGE_RATIO = 0.40
DEFAULT_PRBS_DOFS_RANGE_RATIO = 0.20


def _build_excitation(
    kind: Literal["chirp", "prbs"],
    num_dofs: int,
    duration: float,
    amplitude: np.ndarray,
    seed: int,
):
    dof_indices = list(range(num_dofs))
    if kind == "chirp":
        return ChirpExcitation(
            num_dofs=num_dofs,
            dof_indices=dof_indices,
            f_start=0.2,
            f_end=3.0,
            duration=duration,
            amplitude=amplitude,
            stagger_phase=np.pi,  # phase-shift joints so input isn't rank-1
        )
    if kind == "prbs":
        return PRBSExcitation(
            num_dofs=num_dofs,
            dof_indices=dof_indices,
            period=0.3,
            duration=duration,
            amplitude=amplitude,
            seed=seed,
        )
    raise ValueError(f"Unknown excitation: {kind!r}")


def _get_dof_limit_arrays(
    env, entity_name: str, dofs_name: list[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    entity = env.entities[entity_name]
    _, dofs_idx_local = entity.find_named_dofs_idx_local(dofs_name, name_scope=entity.dofs_name, preserve_order=True)
    # Both bounds carry the env axis; env 0 is representative (limits are a model constant).
    lower, upper = entity.get_dofs_limit(dofs_idx_local=dofs_idx_local)
    lower, upper = lower[0], upper[0]
    lower_np = lower.detach().cpu().numpy().astype(np.float64)
    upper_np = upper.detach().cpu().numpy().astype(np.float64)
    raw_range = upper_np - lower_np
    dof_range = np.where(np.isfinite(raw_range), raw_range, 2.0 * np.pi).clip(min=1e-6)
    center = np.where(np.isfinite(lower_np) & np.isfinite(upper_np), 0.5 * (lower_np + upper_np), 0.0)
    return lower_np, upper_np, center, dof_range


def main() -> int:
    parser = make_argparser(description=__doc__)
    parser.add_argument(
        "--excitation",
        choices=["chirp", "prbs"],
        default="chirp",
        help="Excitation signal type.",
    )
    parser.add_argument("-t", "--seconds", type=float, default=30.0, help="Seconds to drive the excitation trajectory")
    parser.add_argument(
        "--dofs-range-ratio",
        type=float,
        default=None,
        help=(
            "Total excitation span as a fraction of each joint's URDF position range. "
            "1.0 sweeps from lower to upper around the midpoint. Defaults depend on excitation."
        ),
    )
    parser.add_argument(
        "--safety-limit",
        type=float,
        default=DEFAULT_SAFETY_RAD,
        help="Per-joint absolute clamp on the excitation offset (rad).",
    )
    parser.add_argument(
        "--control-freq",
        type=float,
        default=250.0,
        help="Control-loop frequency in Hz.",
    )
    parser.add_argument(
        "--protocol", choices=["RS485", "EtherCAT"], default="RS485", help="Hardware bus the hand is wired on"
    )
    parser.add_argument("--serial-port", default="/dev/ttyUSB0", help="Serial device the hand is attached to")
    parser.add_argument("--hand-id", type=int, default=0, help="Hardware id of the hand on the bus")
    parser.add_argument(
        "--kp-scale",
        type=float,
        default=1.0,
        help="Scale on the hand's firmware proportional gain (RoboTeraXHandDeployment.dofs_kp), rounded to an int",
    )
    parser.add_argument(
        "--kd-scale",
        type=float,
        default=1.0,
        help="Scale on the hand's firmware derivative gain (RoboTeraXHandDeployment.dofs_kd), rounded to an int",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "-o",
        "--output",
        type=pathlib.Path,
        required=True,
        help="Output npz path for the recorded Trajectory.",
    )
    args = parser.parse_args()

    dofs_range_ratio = args.dofs_range_ratio
    if dofs_range_ratio is None:
        dofs_range_ratio = (
            DEFAULT_CHIRP_DOFS_RANGE_RATIO if args.excitation == "chirp" else DEFAULT_PRBS_DOFS_RANGE_RATIO
        )
    if not 0.0 <= dofs_range_ratio <= 1.0:
        raise SystemExit(f"dofs_range_ratio {dofs_range_ratio} must be in [0.0, 1.0]. Aborting.")
    if args.kp_scale < 0.0 or args.kd_scale < 0.0:
        raise SystemExit(f"--kp-scale {args.kp_scale} and --kd-scale {args.kd_scale} must be >= 0. Aborting.")

    # A lightweight sim twin is needed because DeploymentBase.__init__ pulls
    # dofs_name / default_dofs_{pos,kp,kd} off the configured entity in the scene.
    en.init(performance_mode=False)
    config = make_xhand1_sim_config(
        side=args.side,
        num_envs=1,
        sim_dt=args.sim_dt,
        decimation=args.decimation,
    )
    from eden.envs.base import RLEnvBase

    env = RLEnvBase.from_config(config)
    env.build()
    dofs_name = list(config.scene_options.robot.dofs_name)
    dof_lower, dof_upper, dof_center, dof_ranges = _get_dof_limit_arrays(env, "robot", dofs_name)
    amplitude = 0.5 * dofs_range_ratio * dof_ranges
    finite_limits = np.isfinite(dof_lower) & np.isfinite(dof_upper)
    limit_headroom = np.minimum(dof_center - dof_lower, dof_upper - dof_center)
    too_large_for_limits = finite_limits & (amplitude > limit_headroom + 1e-9)
    if np.any(too_large_for_limits):
        worst = int(np.argmax(np.where(too_large_for_limits, amplitude - limit_headroom, -np.inf)))
        raise SystemExit(
            f"dofs_range_ratio {dofs_range_ratio} exceeds centered joint-limit headroom on {dofs_name[worst]}: "
            f"requested +/-{amplitude[worst]:.3f} rad, available +/-{limit_headroom[worst]:.3f} rad. Aborting."
        )
    if np.any(amplitude > args.safety_limit):
        raise SystemExit(
            f"dofs_range_ratio {dofs_range_ratio} produces a max per-side amplitude of {amplitude.max():.3f} rad, "
            f"which exceeds safety limit {args.safety_limit:.3f} rad. Aborting."
        )

    # Confirm the backend is registered (also checks SDK availability).
    if "robo_tera_x_hand_deployment" not in DEPLOYMENT_REGISTRY:
        raise SystemExit("RoboTeraXHandDeployment is not registered.")

    deployer = RoboTeraXHandDeployment(
        env=env,
        options=DeploymentOptions(
            entity_name="robot",
            control_freq=args.control_freq,
            sync=True,
        ),
    )
    # Per-hand fields live on the deployer subclass, not on DeploymentOptions.
    deployer.protocol = args.protocol
    deployer.serial_port = args.serial_port
    deployer.hand_id = args.hand_id
    # The hand runs its own PD on these firmware gains, programmed at connect(); the gains in each
    # RobotCommand never reach it. So the scales go here, not only into the recorder's commands.
    deployer.dofs_kp = round(deployer.dofs_kp * args.kp_scale)
    deployer.dofs_kd = round(deployer.dofs_kd * args.kd_scale)

    en.logger.info(f"Connecting to XHand1 via {args.protocol} ({args.serial_port}) …")
    deployer.connect()
    try:
        # The script exists to move the hand, so it presses INIT itself (request_init) rather than
        # waiting for an operator: a ramp from the measured pose to the default over init_steps.
        en.logger.info("INIT: ramping the hand to its default pose.")
        deployer.reset()
        deployer.request_init()
        while not deployer.init_done:
            deployer.step(None)

        excitation = _build_excitation(
            args.excitation,
            num_dofs=deployer.num_dofs,
            duration=args.seconds,
            amplitude=amplitude,
            seed=args.seed,
        )
        recorder = DeploymentRecorder(
            deployer=deployer,
            excitation=excitation,
            kp_scale=args.kp_scale,
            kd_scale=args.kd_scale,
            safety_limits=np.full(deployer.num_dofs, args.safety_limit, dtype=np.float64),
            center_dofs_pos=dof_center,
            dofs_limit=(dof_lower, dof_upper),
        )

        en.logger.info(
            f"Recording {args.seconds:.1f} s of {args.excitation} data "
            f"(dofs_range_ratio={dofs_range_ratio:.3f}, amp={amplitude.min():.3f}–{amplitude.max():.3f} rad, "
            f"center={dof_center.min():.3f}–{dof_center.max():.3f} rad, clamp=±{args.safety_limit:.2f} rad)."
        )
        trajectory = recorder.run()
    finally:
        deployer.close()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    trajectory.save(args.output)
    en.logger.info(f"Saved {len(trajectory)} samples to {args.output} (nominal dt={trajectory.dt * 1000:.2f} ms).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
