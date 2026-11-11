"""Identify XHand1 DOF parameters from recorded real-hand trajectories.

---
tier: project
capabilities: [task.sysid, robot.xhand]
hardware: gpu
optional_deps: []
run: uv run python examples/projects/sysid/xhand1/identify.py --trajectory data/xhand_chirp.npz --output results/xhand_sysid/
---

Usage
-----

Basic run on one chirp trace:

    uv run python examples/projects/sysid/xhand1/identify.py \
        --trajectory data/xhand_chirp.npz \
        --output results/xhand_sysid/

Multiple traces (jointly fit):

    uv run python examples/projects/sysid/xhand1/identify.py \
        --trajectory data/xhand_chirp.npz data/xhand_prbs_train.npz \
        --output results/xhand_sysid/

What is identified
------------------
Per-joint ``damping``, ``armature``, and ``frictionloss`` — the three
parameters most responsible for the XHand1's stiction + linkage drag.
``stiffness`` is left frozen at the URDF value; add it with ``--include
stiffness`` for a harder fit if residuals remain large.

The residual uses ``dofs_pos`` only, because it is the one signal the XHand1
SDK measures directly. ``dofs_vel`` is a finite difference of consecutive
position reads (see ``RoboTeraXHandDeployment.read_state``) -- a
control-period average, noisier and a period behind the sim's instantaneous
velocity -- and trajectories recorded before that estimate existed hold
zeros there. ``dofs_torque`` is the firmware's own unsigned reading in
firmware units, not N·m. Either would bias the normalised residual.

Parallelism
-----------
- ``--optimizer scipy`` (default): serial Trust-Region Reflective, one
  rollout per FD column per iteration. ``--num-envs 1`` is enough.
- ``--optimizer scipy_parallel_fd``: compute all FD columns in one batched
  replay. Requires ``--num-envs (n_params + 1)`` (61 for the default 60
  parameters: 5 properties x 12 joints).
- ``--optimizer cmaes``: CMA-ES with batched candidate evaluation.
  ``--num-envs`` sets the population size — tune to your GPU budget
  (16–4096 are all reasonable, larger pops trade compute for fewer
  generations).
"""

from __future__ import annotations

import pathlib
import sys
from argparse import BooleanOptionalAction
from typing import Sequence

import genesis as gs
import numpy as np
from sim_twin_config import make_argparser, make_xhand1_sim_config

import eden as en
from eden.extensions.sysid import (
    CMAES,
    Parameter,
    ParameterSet,
    SciPyLeastSquares,
    Trajectory,
    make_parameter_from_default,
)
from eden.extensions.sysid.report import signal_plots, write_summary
from eden.options.extensions.sysid import SystemIdentificationOptions

# Physically reasonable bounds for a small geared dex-hand joint.
# kp / kd are included because the XHand SDK's internal PD gains are in
# firmware-specific units, not N·m/rad — RobotCommand.dofs_kp / .dofs_kd
# are never forwarded to the hand (see RoboTeraXHandDeployment.send_payload).
# The sim therefore sees an effective gain ~100× too stiff; identifying
# kp/kd alongside the solver fields lets the optimiser find the closed-loop
# bandwidth that actually matches the real response.
_BOUNDS = {
    "damping": (0.0, 20.0),  # N·m·s/rad
    "armature": (5e-4, 0.05),  # kg·m^2 reflected
    "frictionloss": (1e-3, 2.0),  # N·m Coulomb torque
    "stiffness": (0.0, 20.0),  # N·m/rad passive
    "kp": (0.1, 100.0),  # N·m/rad (PD proportional)
    "kd": (0.01, 20.0),  # N·m·s/rad (PD derivative)
}


def _build_parameters(
    env,
    dof_names: Sequence[str],
    properties: Sequence[str],
    entity_name: str = "robot",
    per_dof: bool = True,
) -> ParameterSet:
    """Build a per-joint Parameter per requested property with nominal = URDF value.

    For each property the parameter is constructed with the current entity
    value as ``nominal`` and absolute bounds from ``_BOUNDS`` — not scale-
    relative, since ``frictionloss`` often defaults to 0 which degenerates
    multiplicative bounds.
    """
    params: list[Parameter] = []
    for prop in properties:
        p = make_parameter_from_default(
            env=env,
            name=prop,
            property=prop,
            dof_names=dof_names,
            entity_name=entity_name,
            per_dof=per_dof,
        )
        lo, hi = _BOUNDS[prop]
        p.min_value[:] = lo
        p.max_value[:] = np.maximum(hi, p.nominal * 1.001)  # keep nominal feasible
        p.nominal = np.clip(p.nominal, p.min_value, p.max_value)
        p.value = p.nominal.copy()
        params.append(p)
    return ParameterSet(params)


def _warm_start_from_yaml(params: ParameterSet, yaml_path: pathlib.Path) -> None:
    """Copy ``value`` from a saved ParameterSet YAML into an existing set.

    Bounds and nominal in ``params`` are preserved — only values are overwritten.
    Missing params, shape mismatches, and out-of-bounds values are each logged
    once and fall back to the current nominal (or a clipped value).
    """
    if not yaml_path.exists():
        raise SystemExit(f"--resume path {yaml_path} does not exist.")
    prev = ParameterSet.load_yaml(yaml_path)
    en.logger.info(f"Resuming from {yaml_path} ({len(prev)} params in file)")

    copied, skipped = 0, 0
    for p in params:
        if p.name not in prev:
            en.logger.warning(f"--resume: '{p.name}' not in {yaml_path.name}; using nominal.")
            skipped += 1
            continue
        loaded = prev[p.name]
        if loaded.value.shape != p.value.shape:
            en.logger.warning(
                f"--resume: '{p.name}' shape {loaded.value.shape} in file "
                f"doesn't match current {p.value.shape}; using nominal."
            )
            skipped += 1
            continue
        clipped = np.clip(loaded.value, p.min_value, p.max_value)
        if not np.allclose(clipped, loaded.value):
            en.logger.warning(
                f"--resume: '{p.name}' value clipped to current bounds [{p.min_value.min():g}, {p.max_value.max():g}]."
            )
        p.value = clipped.copy()
        copied += 1
    en.logger.info(f"--resume: warm-started {copied} params, skipped {skipped}.")


def main() -> int:
    parser = make_argparser(description=__doc__)
    parser.add_argument(
        "--trajectory",
        type=pathlib.Path,
        nargs="+",
        required=True,
        help="One or more .npz files written by collect_data.py (or SysIDRecorder).",
    )
    parser.add_argument(
        "--include",
        choices=list(_BOUNDS.keys()),
        nargs="+",
        default=["damping", "armature", "frictionloss", "kp", "kd"],
        help="Which DOF properties to identify.",
    )
    parser.add_argument(
        "--per-joint",
        action=BooleanOptionalAction,
        default=True,
        help="If set, identify a separate scalar per joint (recommended).",
    )
    parser.add_argument(
        "--signals",
        nargs="+",
        default=["dofs_pos"],
        choices=["dofs_pos", "dofs_vel", "dofs_torque"],
        help="Signals used in the residual. XHand1 SDK reports only dofs_pos.",
    )
    parser.add_argument(
        "--optimizer",
        choices=["scipy", "scipy_parallel_fd", "cmaes"],
        default="scipy",
        help="Optimizer backend to run the fit with",
    )
    parser.add_argument(
        "-b",
        "--num-envs",
        type=int,
        default=1,
        help="Sim-twin env count. For scipy_parallel_fd use n_params + 1; for cmaes use population size.",
    )
    parser.add_argument(
        "--diff-step",
        type=float,
        default=1e-2,
        help="Relative FD step in unit-hypercube parameter space (scipy backends). "
        "Default 1e-2 = 1%% of each bound range.",
    )
    parser.add_argument("--max-iters", type=int, default=100, help="Maximum optimizer iterations")
    parser.add_argument("--kp", type=float, default=50.0, help="PD kp (must match data-collection kp).")
    parser.add_argument("--kd", type=float, default=25.0, help="Derivative gain used to track the trajectory")
    parser.add_argument(
        "--sigma0",
        type=float,
        default=None,
        help="CMA-ES initial step size in unit-hypercube parameter space. "
        "Defaults: 0.25 from scratch, 0.05 when --resume is set (exploit a warm-start). "
        "Higher = more exploration, lower = tighter local search.",
    )
    parser.add_argument(
        "--resume",
        type=pathlib.Path,
        default=None,
        help="Path to a previously saved params_identified.yaml to warm-start from. "
        "Bounds are taken from the current _BOUNDS (so bound widening takes effect); "
        "only the fitted values are copied. Mismatched shapes or missing params fall "
        "back to the fresh nominal.",
    )
    parser.add_argument("-o", "--output", type=pathlib.Path, required=True, help="Path to write the result to")
    args = parser.parse_args()

    en.init(backend=gs.gpu, performance_mode=False)
    config = make_xhand1_sim_config(
        side=args.side,
        num_envs=args.num_envs,
        sim_dt=args.sim_dt,
        decimation=args.decimation,
        kp=args.kp,
        kd=args.kd,
    )
    from eden.envs.base import RLEnvBase

    env = RLEnvBase.from_config(config)
    env.build()

    entity = env.entities["robot"]
    dof_names = list(entity.dofs_name)

    trajectories = [Trajectory.load(p) for p in args.trajectory]
    en.logger.info(
        f"Loaded {len(trajectories)} trajectories "
        f"(total {sum(len(t) for t in trajectories)} samples across "
        f"{sum(len(t) * t.dt for t in trajectories):.1f} s)."
    )

    # The rollout steps at env.dt per trajectory sample. If the real loop ran
    # at a different cadence than the sim twin, the replayed chirp will be
    # frequency-shifted and the residual will carry a systematic bias no
    # parameter fit can remove.
    sim_step_dt = config.env_options.sim_dt * config.env_options.decimation
    for i, t in enumerate(trajectories):
        rel_err = abs(t.dt - sim_step_dt) / sim_step_dt if sim_step_dt > 0 else 0.0
        if rel_err > 0.02:
            en.logger.warning(
                f"trajectory {i} dt={t.dt * 1000:.2f} ms but sim step "
                f"dt={sim_step_dt * 1000:.2f} ms ({rel_err * 100:.1f} % mismatch). "
                f"Either re-record at --control-freq {1 / sim_step_dt:.0f} or "
                f"retune the sim's sim_dt * decimation to match the recorded dt."
            )

    params = _build_parameters(env, dof_names, args.include, per_dof=args.per_joint)
    if args.resume is not None:
        _warm_start_from_yaml(params, args.resume)
    en.logger.info(
        f"Optimising {params.size} parameters "
        f"({len(args.include)} properties × {len(dof_names) if args.per_joint else 1} joints)."
    )

    base_kwargs: dict = dict(
        entity_name="robot",
        signals=tuple(args.signals),
        signal_weights={},
        normalize=True,
        max_iters=args.max_iters,
        verbose=True,
    )
    if args.optimizer == "cmaes":
        if args.num_envs > 1:
            base_kwargs["population_size"] = args.num_envs
            base_kwargs["batched"] = True
        sigma0 = args.sigma0
        if sigma0 is None:
            sigma0 = 0.05 if args.resume is not None else 0.25
        base_kwargs["sigma0"] = sigma0
        en.logger.info(f"CMA-ES sigma0={sigma0:g} ({'warm-start' if args.resume is not None else 'cold-start'}).")
        sysid = CMAES(env=env, options=SystemIdentificationOptions(**base_kwargs))
    else:
        parallel_fd = args.optimizer == "scipy_parallel_fd"
        if parallel_fd and args.num_envs != params.size + 1:
            raise SystemExit(
                f"scipy_parallel_fd needs --num-envs == n_params + 1 (got {args.num_envs}, need {params.size + 1})."
            )
        base_kwargs["diff_step"] = args.diff_step
        base_kwargs["parallel_fd"] = parallel_fd
        sysid = SciPyLeastSquares(env=env, options=SystemIdentificationOptions(**base_kwargs))

    initial = params.copy()
    identified, result = sysid.fit(params, trajectories)
    en.logger.info(f"Optimisation done. cost={result.cost:.6g} nfev={result.nfev} msg={result.message!r}")

    args.output.mkdir(parents=True, exist_ok=True)
    write_summary(args.output, initial, identified, result)

    # Per-signal plots on the first trajectory with both the pre-fit (URDF
    # defaults) and post-fit predictions overlaid on the measurement so the
    # improvement from identification is visible at a glance.
    from eden.extensions.sysid.rollout import single_candidate_rollout

    traj0 = trajectories[0]
    initial_pred = single_candidate_rollout(env, initial, traj0, entity_name="robot", signals=tuple(args.signals))
    identified_pred = single_candidate_rollout(env, identified, traj0, entity_name="robot", signals=tuple(args.signals))
    signal_plots(
        predicted={"initial": initial_pred, "identified": identified_pred},
        measured=traj0,
        signals=tuple(args.signals),
        save_dir=args.output / "plots",
        dof_names=dof_names,
    )
    en.logger.info(f"Wrote results to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
