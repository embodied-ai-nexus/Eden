"""Verify identified XHand1 parameters against a held-out trajectory.

---
tier: project
capabilities: [task.sysid, robot.xhand]
hardware: gpu
optional_deps: []
run: uv run python examples/projects/sysid/xhand1/verify.py --trajectory data/xhand_prbs_verify.npz --params results/xhand_sysid/params_identified.yaml --output results/xhand_sysid/verify/
---

Applies the YAML produced by ``identify.py`` to the sim twin, replays the
held-out action trace, and reports per-signal RMSE against the measured
response — both *before* and *after* applying the identified parameters
— so it is immediately visible whether identification improved the fit.

Usage
-----

    uv run python examples/projects/sysid/xhand1/verify.py \
        --trajectory data/xhand_prbs_verify.npz \
        --params results/xhand_sysid/params_identified.yaml \
        --output results/xhand_sysid/verify/
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
from sim_twin_config import make_argparser, make_xhand1_sim_config

import eden as en
from eden.extensions.sysid import ParameterSet, Trajectory
from eden.extensions.sysid.report import signal_plots
from eden.extensions.sysid.rollout import single_candidate_rollout


def _rmse(pred: np.ndarray, meas: np.ndarray) -> float:
    n = min(pred.shape[0], meas.shape[0])
    return float(np.sqrt(np.mean((pred[:n] - meas[:n]) ** 2)))


def _per_signal_rmse(
    env,
    params: ParameterSet,
    trajectory: Trajectory,
    signals: tuple[str, ...],
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    pred = single_candidate_rollout(env, params, trajectory, entity_name="robot", signals=signals)
    out: dict[str, float] = {}
    for s in signals:
        meas = trajectory.signal(s)
        if meas is None or s not in pred:
            continue
        out[s] = _rmse(pred[s], meas)
    return out, pred


def main() -> int:
    parser = make_argparser(description=__doc__)
    parser.add_argument("--trajectory", type=pathlib.Path, required=True, help="Recorded trajectory to fit against")
    parser.add_argument("--params", type=pathlib.Path, required=True, help="Identified parameter file to verify")
    parser.add_argument(
        "--signals",
        nargs="+",
        default=["dofs_pos", "dofs_vel"],
        choices=["dofs_pos", "dofs_vel", "dofs_torque"],
        help="Signals to score the fit on",
    )
    parser.add_argument("--kp", type=float, default=50.0, help="Proportional gain used to track the trajectory")
    parser.add_argument("--kd", type=float, default=5.0, help="Derivative gain used to track the trajectory")
    parser.add_argument("-o", "--output", type=pathlib.Path, required=True, help="Path to write the result to")
    args = parser.parse_args()

    signals = tuple(args.signals)

    en.init(performance_mode=False)
    config = make_xhand1_sim_config(
        side=args.side,
        num_envs=1,
        sim_dt=args.sim_dt,
        decimation=args.decimation,
        kp=args.kp,
        kd=args.kd,
    )
    from eden.envs.base import RLEnvBase

    env = RLEnvBase.from_config(config)
    env.build()
    trajectory = Trajectory.load(args.trajectory)

    identified = ParameterSet.load_yaml(args.params)

    # Baseline = URDF-default values, read straight from the freshly-built sim
    # twin. Same shape / bounds as the identified set so the rollout applies
    # them through the same code path.
    from eden.extensions.sysid.modifier import DOF_PROPERTY_GETTERS

    baseline = identified.copy()
    entity = env.entities["robot"]
    for p in baseline:
        _, dof_indices = entity.find_named_dofs_idx_local(list(p.dof_names), preserve_order=True)
        vals = getattr(entity, DOF_PROPERTY_GETTERS[p.property])(dofs_idx_local=dof_indices)
        if vals.ndim == 2:
            vals = vals[0]
        vals_np = vals.detach().cpu().numpy().astype(np.float64)
        p.value = vals_np if p.per_dof else np.array([float(vals_np.mean())])

    baseline_rmse, baseline_pred = _per_signal_rmse(env, baseline, trajectory, signals)
    identified_rmse, identified_pred = _per_signal_rmse(env, identified, trajectory, signals)

    args.output.mkdir(parents=True, exist_ok=True)
    lines = ["signal             baseline RMSE   identified RMSE   improvement"]
    summary = {}
    for s in signals:
        b = baseline_rmse.get(s, float("nan"))
        i = identified_rmse.get(s, float("nan"))
        imp = 100.0 * (b - i) / b if (b and np.isfinite(b)) else float("nan")
        lines.append(f"{s:<18} {b:>14.6g} {i:>17.6g} {imp:>12.1f}%")
        summary[s] = {"baseline": b, "identified": i, "improvement_pct": imp}
    report = "\n".join(lines)
    en.logger.info("\n" + report)
    (args.output / "verify.txt").write_text(report + "\n")

    signal_plots(
        predicted={"baseline": baseline_pred, "identified": identified_pred},
        measured=trajectory,
        signals=signals,
        save_dir=args.output / "plots",
        dof_names=list(entity.dofs_name),
    )

    # Non-zero exit if identification didn't meaningfully improve any signal.
    any_improved = any(
        np.isfinite(summary[s]["improvement_pct"]) and summary[s]["improvement_pct"] > 1.0 for s in signals
    )
    return 0 if any_improved else 2


if __name__ == "__main__":
    sys.exit(main())
