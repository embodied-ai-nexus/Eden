"""Tests for ``eden.extensions.retargeting.dex_retargeter.optimizer`` SLSQP backends."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from eden.extensions.retargeting.dex_retargeter import DexRetargeter
from eden.extensions.retargeting.dex_retargeter.optimizer import (
    DexPilotOptimizer,
    _make_slsqp_backend,
    _ScipySlsqpOpt,
)

FINGER_TIPS = [
    "right_hand_thumb_rota_tip",
    "right_hand_index_rota_tip",
    "right_hand_mid_tip",
    "right_hand_ring_tip",
    "right_hand_pinky_tip",
]
WRIST = "right_hand_link"


def test_scipy_slsqp_adapter_respects_bounds_and_converges() -> None:
    opt = _ScipySlsqpOpt(2)
    opt.set_lower_bounds([-0.5, -0.5])
    opt.set_upper_bounds([0.5, 0.5])
    opt.set_ftol_abs(1e-8)

    target = np.array([0.3, -1.2], dtype=np.float64)

    def obj(x: np.ndarray, grad: np.ndarray) -> float:
        diff = x.astype(np.float64) - target
        if grad.size > 0:
            grad[:] = 2.0 * diff
        return float(np.sum(diff**2))

    opt.set_min_objective(obj)
    sol = opt.optimize(np.zeros(2, dtype=np.float32))
    np.testing.assert_allclose(sol, np.array([0.3, -0.5], dtype=np.float32), atol=1e-4)


def test_unknown_solver_backend_raises_actionable_error() -> None:
    with pytest.raises(ValueError, match="solver_backend='invalid'"):
        _make_slsqp_backend(2, solver_backend="invalid")  # type: ignore[arg-type]


@pytest.mark.parametrize("mode", ["position", "vector", "dexpilot"])
def test_scipy_and_nlopt_backends_match_across_trajectory(env, mode: str) -> None:
    pytest.importorskip("nlopt")
    robot = env.entities["robot"]
    limits = torch.stack(robot.get_dofs_limit(), dim=-1)[0].cpu().numpy()
    lo = limits[:, 0] + 0.05 * (limits[:, 1] - limits[:, 0])
    hi = limits[:, 1] - 0.05 * (limits[:, 1] - limits[:, 0])
    mid = 0.5 * (lo + hi)
    amp = 0.35 * (hi - lo)

    t = np.linspace(0.0, np.pi, 16, dtype=np.float32)[:, None]
    gt_traj = np.clip(mid[None, :] + amp[None, :] * np.sin(t), lo, hi).astype(np.float32)

    tip_links = [robot.get_link(name) for name in FINGER_TIPS]
    wrist_link = robot.get_link(WRIST)
    orig_idx, task_idx = DexPilotOptimizer._generate_link_indices(len(FINGER_TIPS))

    refs: list[np.ndarray] = []
    gt_tips: list[np.ndarray] = []
    try:
        for q in gt_traj:
            robot.set_qpos(torch.as_tensor(q, device=robot.device).unsqueeze(0), zero_velocity=True)
            tips = np.stack([lk.get_pos(relative=False).squeeze(0).cpu().numpy() for lk in tip_links], axis=0)
            wrist = wrist_link.get_pos(relative=False).squeeze(0).cpu().numpy()
            gt_tips.append(tips)
            if mode == "position":
                refs.append(tips.astype(np.float32))
            elif mode == "vector":
                refs.append((tips - wrist[None, :]).astype(np.float32))
            else:
                pts = np.vstack([wrist[None, :], tips])
                refs.append((pts[task_idx] - pts[orig_idx]).astype(np.float32))

        ref_seq = np.stack(refs, axis=0)
        gt_tips_arr = np.stack(gt_tips, axis=0)

        def _make(backend: str) -> DexRetargeter:
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
            else:
                opts = DexRetargeter.configure(
                    type="dexpilot",
                    finger_tip_link_names=FINGER_TIPS,
                    wrist_link_name=WRIST,
                    **common,
                )
            return DexRetargeter(env=env, options=opts)

        q_nlopt = _make("nlopt").retarget_motion(ref_seq)
        q_scipy = _make("scipy").retarget_motion(ref_seq)

        assert np.isfinite(q_scipy).all()
        q_rmse = float(np.sqrt(np.mean((q_scipy - q_nlopt) ** 2)))
        assert q_rmse < 0.05, f"{mode}: scipy vs nlopt trajectory RMSE {q_rmse:.4f} rad exceeds 0.05 rad"

        # Verify SciPy fingertip tracking accuracy directly on the final frame.
        robot.set_qpos(torch.as_tensor(q_scipy[-1], device=robot.device).unsqueeze(0), zero_velocity=True)
        ach_tips = np.stack([lk.get_pos(relative=False).squeeze(0).cpu().numpy() for lk in tip_links], axis=0)
        tip_err_m = np.linalg.norm(ach_tips - gt_tips_arr[-1], axis=-1)
        bound_m = 0.05 if mode != "dexpilot" else 0.08
        assert tip_err_m.max() < bound_m, f"{mode}: max tip error {tip_err_m.max() * 1e3:.2f} mm"
    finally:
        env.reset()
