"""Speed and accuracy benchmark comparing ``nlopt`` (``LD_SLSQP``) vs ``scipy`` (``SLSQP``)
across ``PositionOptimizer``, ``VectorOptimizer``, and ``DexPilotOptimizer`` on a fixed-base ``XHand1_R``
scene in Genesis.

Run standalone::

    uv run python tests/extensions/retargeting/dex_retargeter/benchmark/bench_slsqp_backends.py
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import numpy as np
import torch

import eden as en
from eden.envs.base import EnvBase
from eden.extensions.retargeting.dex_retargeter import DexRetargeter
from eden.extensions.retargeting.dex_retargeter.optimizer import DexPilotOptimizer, SolverBackend
from eden.options import EnvOptions, SceneOptions
from eden.options.entities import PlaneOptions
from eden.utils.configs import EdenConfig

FINGER_TIPS = [
    "right_hand_thumb_rota_tip",
    "right_hand_index_rota_tip",
    "right_hand_mid_tip",
    "right_hand_ring_tip",
    "right_hand_pinky_tip",
]
WRIST = "right_hand_link"


@dataclass(frozen=True)
class BenchmarkRow:
    mode: str
    backend: str
    n_frames: int
    mean_ms: float
    median_ms: float
    p95_ms: float
    fps: float
    evals_per_frame: float
    tip_rmse_mm: float
    tip_max_mm: float
    qpos_rmse_rad: float
    solved_traj: np.ndarray


def _build_xhand_env() -> EnvBase:
    en.init(debug=False)
    cfg = EdenConfig(
        env_options=EnvOptions(num_envs=1),
        scene_options=SceneOptions(
            robot=en.robots.XHand1_R(is_fixed_base=True, default_root_pos=(0.0, 0.0, 0.1)),
            floor=PlaneOptions(),
        ),
    )
    env = EnvBase.from_config(cfg, show_viewer=False)
    env.build()
    env.reset()
    return env


def _make_smooth_trajectory(limits: np.ndarray, n_frames: int, seed: int = 0) -> np.ndarray:
    """Generate a smooth multi-frequency reachable 12-DOF grasp/pinch trajectory within joint limits."""
    rng = np.random.default_rng(seed)
    span = limits[:, 1] - limits[:, 0]
    lo = limits[:, 0] + 0.05 * span
    hi = limits[:, 1] - 0.05 * span
    mid = 0.5 * (lo + hi)
    amp = 0.40 * (hi - lo)
    t = np.linspace(0.0, 2.0 * np.pi, n_frames, dtype=np.float32)[:, None]
    phases = rng.uniform(0.0, 2.0 * np.pi, size=(1, len(lo))).astype(np.float32)
    freqs = rng.choice([1.0, 2.0, 3.0], size=(1, len(lo))).astype(np.float32)
    q = mid[None, :] + amp[None, :] * (0.65 * np.sin(freqs * t + phases) + 0.35 * np.cos(t))
    return np.clip(q, lo, hi).astype(np.float32)


def _make_retargeter(env: EnvBase, mode: str, backend: SolverBackend) -> DexRetargeter:
    common = {"entity_name": "robot", "low_pass_alpha": 1.0, "solver_backend": backend}
    if mode == "position":
        opts = DexRetargeter.configure(
            type="position",
            target_link_names=FINGER_TIPS,
            target_link_human_indices=np.array([4, 8, 12, 16, 20]),
            **common,
        )
    elif mode == "vector":
        opts = DexRetargeter.configure(
            type="vector",
            target_origin_link_names=[WRIST] * 5,
            target_task_link_names=FINGER_TIPS,
            target_link_human_indices=np.array([[0] * 5, [4, 8, 12, 16, 20]]),
            **common,
        )
    elif mode == "dexpilot":
        opts = DexRetargeter.configure(
            type="dexpilot",
            finger_tip_link_names=FINGER_TIPS,
            wrist_link_name=WRIST,
            **common,
        )
    else:
        raise ValueError(f"Unknown mode {mode!r}")
    return DexRetargeter(env=env, options=opts)


def run_benchmark(n_frames: int = 200, warmup: int = 10, seed: int = 0) -> list[BenchmarkRow]:
    env = _build_xhand_env()
    robot = env.entities["robot"]
    limits = torch.stack(robot.get_dofs_limit(), dim=-1)[0].cpu().numpy()
    gt_qpos = _make_smooth_trajectory(limits, n_frames + warmup, seed=seed)

    tip_links = [robot.get_link(name) for name in FINGER_TIPS]
    wrist_link = robot.get_link(WRIST)
    orig_idx, task_idx = DexPilotOptimizer._generate_link_indices(len(FINGER_TIPS))

    def _tip_positions(q: np.ndarray) -> np.ndarray:
        robot.set_qpos(torch.as_tensor(q, dtype=torch.float32, device=robot.device).unsqueeze(0), zero_velocity=True)
        return np.stack([lk.get_pos(relative=False).squeeze(0).cpu().numpy() for lk in tip_links], axis=0).astype(
            np.float32
        )

    robot.set_qpos(
        torch.as_tensor(gt_qpos[0], dtype=torch.float32, device=robot.device).unsqueeze(0), zero_velocity=True
    )
    wrist_pos = wrist_link.get_pos(relative=False).squeeze(0).cpu().numpy().astype(np.float32)
    gt_tips = np.stack([_tip_positions(q) for q in gt_qpos], axis=0)

    rows: list[BenchmarkRow] = []
    for mode in ("position", "vector", "dexpilot"):
        for backend in ("nlopt", "scipy"):
            retargeter = _make_retargeter(env, mode, backend)  # type: ignore[arg-type]
            fk_counter = [0]
            orig_set_qpos = retargeter.optimizer._set_qpos

            def _counted_set_qpos(qpos: np.ndarray, _orig=orig_set_qpos, _cnt=fk_counter) -> None:
                _cnt[0] += 1
                _orig(qpos)

            retargeter.optimizer._set_qpos = _counted_set_qpos  # type: ignore[method-assign]

            frame_ms: list[float] = []
            solved_list: list[np.ndarray] = []
            tip_errs_m: list[np.ndarray] = []
            eval_counts: list[int] = []

            for f in range(n_frames + warmup):
                tips_f = gt_tips[f]
                if mode == "position":
                    ref_val = tips_f
                elif mode == "vector":
                    ref_val = tips_f - wrist_pos[None, :]
                else:
                    all_pts = np.vstack([wrist_pos[None, :], tips_f])
                    ref_val = all_pts[task_idx] - all_pts[orig_idx]

                fk_counter[0] = 0
                t0 = time.perf_counter()
                q_sol = retargeter.retarget(ref_val)
                dt_ms = (time.perf_counter() - t0) * 1e3

                if f >= warmup:
                    frame_ms.append(dt_ms)
                    eval_counts.append(fk_counter[0])
                    solved_list.append(q_sol.copy())
                    ach_tips = _tip_positions(q_sol)
                    tip_errs_m.append(np.linalg.norm(ach_tips - tips_f, axis=-1))

            solved_arr = np.stack(solved_list, axis=0)
            tip_err_mm = np.concatenate(tip_errs_m) * 1e3
            q_rmse = float(np.sqrt(np.mean((solved_arr - gt_qpos[warmup:]) ** 2)))
            ms_arr = np.asarray(frame_ms, dtype=np.float64)

            rows.append(
                BenchmarkRow(
                    mode=mode,
                    backend=backend,
                    n_frames=n_frames,
                    mean_ms=float(ms_arr.mean()),
                    median_ms=float(np.median(ms_arr)),
                    p95_ms=float(np.percentile(ms_arr, 95)),
                    fps=float(1000.0 / ms_arr.mean()),
                    evals_per_frame=float(np.mean(eval_counts)),
                    tip_rmse_mm=float(np.sqrt(np.mean(tip_err_mm**2))),
                    tip_max_mm=float(np.max(tip_err_mm)),
                    qpos_rmse_rad=q_rmse,
                    solved_traj=solved_arr,
                )
            )
    return rows


def format_markdown_table(rows: list[BenchmarkRow]) -> str:
    by_key = {(r.mode, r.backend): r for r in rows}
    lines = [
        "| Optimizer | Backend | Mean (ms/frame) | Median (ms) | p95 (ms) | Throughput (Hz) | FK evals/frame | Tip RMSE (mm) | Tip Max (mm) | `qpos` diff vs `nlopt` RMSE (rad) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode in ("position", "vector", "dexpilot"):
        nl = by_key[(mode, "nlopt")]
        for backend in ("nlopt", "scipy"):
            r = by_key[(mode, backend)]
            q_diff = 0.0 if backend == "nlopt" else float(np.sqrt(np.mean((r.solved_traj - nl.solved_traj) ** 2)))
            lines.append(
                f"| `{mode}` | `{backend}` | {r.mean_ms:.3f} | {r.median_ms:.3f} | {r.p95_ms:.3f} | "
                f"{r.fps:.1f} | {r.evals_per_frame:.1f} | {r.tip_rmse_mm:.3f} | {r.tip_max_mm:.3f} | {q_diff:.5f} |"
            )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark nlopt vs scipy SLSQP in DexRetargeter on XHand1_R.")
    parser.add_argument("--frames", type=int, default=200, help="Measured trajectory frames after warmup.")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup frames before timing.")
    parser.add_argument("--seed", type=int, default=0, help="Random trajectory seed.")
    args = parser.parse_args()

    rows = run_benchmark(n_frames=args.frames, warmup=args.warmup, seed=args.seed)
    print(format_markdown_table(rows))


if __name__ == "__main__":
    main()
