"""Analyze an `eden deploy --debug_record` .npz: per-joint tracking RMS, torque
saturation %, and dominant oscillation frequency.

---
tier: policy
capabilities: [deploy.real_robot, robot.g1]
hardware: cpu
optional_deps: []
run: uv run python examples/policies/humanoid_gpt/analyze_record.py rec.npz
---

The recording stores, per control step, the command actually sent
(`cmd_pos`/`cmd_kp`/`cmd_kd`) and the measured robot state
(`q`/`qd`/`tau`/`base_*`), plus `stamp` and `dofs_name`. This script turns that
into the three numbers you want when chasing a deploy instability:

* tracking RMS  — RMS of (cmd_pos - q) per joint: how far the joint lags its
  target. A buzzing joint shows large RMS even though the *target* is smooth.
* torque sat %  — fraction of steps with |tau| >= sat_frac * effort_limit: a
  motor pinned at its torque ceiling (the 5010->4010 hip hypothesis).
* dominant freq — peak frequency of the tracking residual (and of the base
  angular velocity): the limit-cycle / "monotone whine" frequency.

Effort limits are resolved from eden's `G1Dofs29` (4010-spec) when available, so
they match the deployed robot's motors.

Usage::

    python examples/policies/humanoid_gpt/analyze_record.py logs/<...>/deploy_record.npz
    python examples/policies/humanoid_gpt/analyze_record.py rec.npz --sat-frac 0.8 --fmin 1.0 --top 8
"""

from __future__ import annotations

import argparse

import numpy as np


def _effort_limits(dofs_name: list[str]) -> np.ndarray | None:
    """Per-joint effort limits aligned to ``dofs_name`` from eden's G1Dofs29, or None."""
    try:
        from eden.extensions.deployment.base import resolve_dofs_values
        from eden.options.robots.unitree_g1 import G1Dofs29

        return resolve_dofs_values(G1Dofs29().default_dofs_force_limits, list(dofs_name))
    except Exception as exc:  # noqa: BLE001 — analysis must run without a full eden env
        print(f"[warn] could not resolve effort limits from eden ({exc}); torque sat % disabled.")
        return None


def _dominant_freq(sig: np.ndarray, dt: float, fmin: float) -> tuple[float, float]:
    """Return (peak_freq_hz, single-sided_amplitude) of ``sig`` above ``fmin`` (mean removed)."""
    n = sig.shape[0]
    if n < 4 or dt <= 0:
        return 0.0, 0.0
    spec = np.abs(np.fft.rfft(sig - sig.mean()))
    freqs = np.fft.rfftfreq(n, d=dt)
    band = freqs >= fmin
    if not band.any():
        band = freqs > 0
    if not band.any():
        return 0.0, 0.0
    idx = np.flatnonzero(band)[np.argmax(spec[band])]
    return float(freqs[idx]), float(2.0 * spec[idx] / n)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="deploy_record.npz from `eden deploy --debug_record`")
    parser.add_argument("--sat-frac", type=float, default=0.9, help="|tau| >= sat_frac*limit counts as saturated")
    parser.add_argument("--fmin", type=float, default=1.0, help="ignore oscillations below this Hz (drift)")
    parser.add_argument("--top", type=int, default=6, help="how many worst joints to summarize")
    args = parser.parse_args()

    d = np.load(args.path)
    names = [str(n) for n in d["dofs_name"]]
    cmd, q, tau = d["cmd_pos"], d["q"], d["tau"]
    kp = d["cmd_kp"] if "cmd_kp" in d else np.ones_like(cmd)
    stamp = d["stamp"]
    n_steps = len(stamp)
    dt = float(np.median(np.diff(stamp))) if n_steps > 1 else 0.0
    fs = 1.0 / dt if dt > 0 else float("nan")

    print(f"\n{args.path}: {n_steps} steps, {stamp[-1] - stamp[0]:.1f}s, ~{fs:.1f} Hz\n")

    limits = _effort_limits(names)
    err_deg = np.degrees(cmd - q)  # (T, N)
    rms = np.sqrt((err_deg**2).mean(axis=0))
    peak = np.abs(err_deg).max(axis=0)
    tau_rms = np.sqrt((tau**2).mean(axis=0))
    tau_max = np.abs(tau).max(axis=0)
    disabled = np.abs(kp).max(axis=0) == 0.0
    if limits is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            sat = 100.0 * (np.abs(tau) >= args.sat_frac * limits[None, :]).mean(axis=0)
            sat = np.where(limits > 0, sat, np.nan)
    else:
        sat = np.full(len(names), np.nan)
    fdom = np.array([_dominant_freq(err_deg[:, j], dt, args.fmin) for j in range(len(names))])

    w = max(len(n) for n in names)
    print(f"{'joint':<{w}}  {'RMS°':>6} {'peak°':>6} {'τrms':>6} {'τmax':>6} {'sat%':>5} {'fdom':>6} {'amp°':>6}")
    print("-" * (w + 46))
    for j, name in enumerate(names):
        tag = "  [free]" if disabled[j] else ""
        satj = f"{sat[j]:5.0f}" if np.isfinite(sat[j]) else "  n/a"
        print(
            f"{name:<{w}}  {rms[j]:6.2f} {peak[j]:6.1f} {tau_rms[j]:6.1f} {tau_max[j]:6.1f} "
            f"{satj} {fdom[j, 0]:6.1f} {fdom[j, 1]:6.2f}{tag}"
        )

    # --- summaries ---
    order = np.argsort(rms)[::-1][: args.top]
    print(f"\nworst {args.top} by tracking RMS:")
    for j in order:
        print(f"  {names[j]:<{w}}  RMS={rms[j]:.2f}°  peak={peak[j]:.1f}°  fdom={fdom[j, 0]:.1f}Hz")
    if limits is not None:
        sat_order = np.argsort(np.nan_to_num(sat))[::-1][: args.top]
        if sat[sat_order[0]] > 0:
            print(f"\ntorque saturation (|τ| >= {args.sat_frac:.0%} of effort limit):")
            for j in sat_order:
                if sat[j] > 0:
                    print(f"  {names[j]:<{w}}  sat={sat[j]:.0f}%  τmax={tau_max[j]:.1f}/{limits[j]:.0f}N·m")

    if "base_ang_vel" in d:
        bav = d["base_ang_vel"]
        bf = [_dominant_freq(bav[:, k], dt, args.fmin) for k in range(3)]
        print(
            f"\nbase angular velocity: RMS={np.sqrt((bav**2).sum(axis=1).mean()):.3f} rad/s  "
            f"dominant freq (x/y/z) = {bf[0][0]:.1f}/{bf[1][0]:.1f}/{bf[2][0]:.1f} Hz"
        )


if __name__ == "__main__":
    main()
