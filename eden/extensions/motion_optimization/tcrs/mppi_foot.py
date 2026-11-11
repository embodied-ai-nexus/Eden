"""
MPPI (Model Predictive Path Integral) foot swing trajectory optimizer.

Sampling-based foot swing planner. Uses cubic-spline-parameterized
control points with a pre-computed basis matrix for fast batch
expansion.

Usage:
    from eden.extensions.motion_optimization.tcrs.mppi_foot import MppiFootOptimizer, MppiFootParams
    opt = MppiFootOptimizer(MppiFootParams())
    traj_x, traj_y, traj_z = opt.optimize_swing(
        liftoff_pos, landing_pos, swing_duration, ref_traj, ref_times, terrain
    )

Also exposes ``slerp_quaternion``, used throughout the planner stack.

Ported from Mondo-Robotics/PMT ``TCRS/stair_mppi/mppi_foot.py`` @ 88838ca.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation, Slerp

if TYPE_CHECKING:
    from eden.extensions.motion_optimization.tcrs.terrain_source import TerrainSource


# ---------------------------------------------------------------------------
# Quaternion utilities (used by the planner in ``pipeline.py``)
# ---------------------------------------------------------------------------


# Kept over genesis.utils.geom.slerp: this vectorizes over t for a single quat pair, not over quat batches.
def slerp_quaternion(q0_wxyz: np.ndarray, q1_wxyz: np.ndarray, t: np.ndarray) -> np.ndarray:
    """SLERP between two [w,x,y,z] quaternions. Returns (N, 4) in [w,x,y,z].

    Aligns q1 to the same hemisphere as q0 (flips if dot < 0) to ensure
    shortest-arc interpolation — avoids unexpected 360 degree foot rotations.
    """
    t = np.asarray(t, dtype=np.float64)
    q0 = np.asarray(q0_wxyz, dtype=np.float64)
    q1 = np.asarray(q1_wxyz, dtype=np.float64)
    if np.dot(q0, q1) < 0.0:
        q1 = -q1
    q0_xyzw = np.array([q0[1], q0[2], q0[3], q0[0]])
    q1_xyzw = np.array([q1[1], q1[2], q1[3], q1[0]])
    rots = Rotation.from_quat(np.stack([q0_xyzw, q1_xyzw]))
    slerp_obj = Slerp([0.0, 1.0], rots)
    interp = slerp_obj(np.clip(t, 0.0, 1.0))
    q_xyzw = interp.as_quat()
    return np.column_stack([q_xyzw[:, 3], q_xyzw[:, 0], q_xyzw[:, 1], q_xyzw[:, 2]])


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class MppiFootParams:
    """Parameters for the MPPI foot planner."""

    n_knots: int = 0  # Number of interior control points (excluding the two boundary points)
    n_samples: int = 128  # Number of samples per iteration
    n_iterations: int = 10  # Number of MPPI iterations
    n_dense: int = 100  # Number of dense evaluation points
    temperature: float = 0.1  # Softmax temperature (smaller = greedier)
    noise_std_xy: float = 0.02  # Noise standard deviation for the x/y channels (m)
    noise_std_z: float = 0.04  # Noise standard deviation for the z channel (m)
    w_track: float = 10.0  # Reference trajectory tracking weight
    w_smooth: float = 10.0  # Smoothness weight (second-order differences)
    w_terrain: float = 10000.0  # Terrain penetration penalty weight
    w_ceiling: float = 500.0  # Height upper-bound penalty weight
    h_clearance: float = 0.03  # Terrain clearance margin (m)
    h_contact_z: float = 0.0  # Midfoot z height when supported on flat ground (m)
    ceiling_margin: float = 0.3  # Maximum allowed height above liftoff/landing points (m)
    seed: int = 42  # Random seed
    # Phase-aware cost terms (default 0 = disabled, no behavior change)
    w_stance_ground: float = 0.0  # Penalty for the foot leaving the ground during stance
    w_swing_clearance_reward: float = 0.0  # Reward for ground clearance during swing


# ---------------------------------------------------------------------------
# Lightweight trajectory wrapper (.eval(t) sampler over dense values)
# ---------------------------------------------------------------------------


class EvalWrapper:
    """Wraps dense trajectory arrays, providing an .eval(t) interface with linear interpolation at arbitrary times.

    Internally stores (n_dense,) time and value arrays; eval() linearly interpolates the value at any time.
    """

    def __init__(self, times: np.ndarray, values: np.ndarray):
        self.times = np.asarray(times, dtype=np.float64)
        self.values = np.asarray(values, dtype=np.float64)

    def eval(self, t: np.ndarray) -> np.ndarray:
        """Evaluate at arbitrary time points (linear interpolation)."""
        return np.interp(t, self.times, self.values)


# ---------------------------------------------------------------------------
# Pre-computed spline basis matrix
# ---------------------------------------------------------------------------


def _build_basis_matrix(knot_times: np.ndarray, dense_times: np.ndarray) -> np.ndarray:
    """Pre-compute the basis matrix of a clamped cubic spline.

    For fixed knot time positions, a clamped cubic spline is a linear function
    of the knot values:
        dense_values = B @ knot_values
    where B: (n_dense, n_knots_total).

    Each column is extracted by setting unit knot values one at a time.

    Parameters
    ----------
    knot_times : np.ndarray
        (n_knots_total,) knot times including the two boundary points.
    dense_times : np.ndarray
        (n_dense,) dense evaluation times.

    Returns
    -------
    B : np.ndarray
        (n_dense, n_knots_total) basis matrix.
    """
    n_knots = len(knot_times)
    n_dense = len(dense_times)
    B = np.zeros((n_dense, n_knots), dtype=np.float64)
    for k in range(n_knots):
        unit = np.zeros(n_knots, dtype=np.float64)
        unit[k] = 1.0
        cs = CubicSpline(knot_times, unit, bc_type="clamped")
        B[:, k] = cs(dense_times)
    return B


# ---------------------------------------------------------------------------
# MPPI Foot Swing Optimizer
# ---------------------------------------------------------------------------


class MppiFootOptimizer:
    """Optimize a single-foot swing trajectory via MPPI sampling.

    Returns three EvalWrapper objects (x/y/z), as consumed by
    :func:`~eden.extensions.motion_optimization.tcrs.pipeline.precompute_mppi_foot`.
    """

    def __init__(self, params: MppiFootParams | None = None):
        self.params = params or MppiFootParams()
        self.rng = np.random.default_rng(self.params.seed)

    def optimize_swing(
        self,
        liftoff_pos: np.ndarray,
        landing_pos: np.ndarray,
        swing_duration: float,
        ref_traj: np.ndarray,
        ref_times: np.ndarray,
        terrain: TerrainSource,
        v_start: np.ndarray | None = None,
        v_end: np.ndarray | None = None,
        phase_stance_mask: np.ndarray | None = None,
        phase_swing_envelope: np.ndarray | None = None,
    ) -> tuple[EvalWrapper, EvalWrapper, EvalWrapper]:
        """Optimize the xyz trajectory of one swing.

        ``v_start``/``v_end`` are ignored (the clamped spline boundary conditions
        fix zero endpoint velocity). The optional phase-aware inputs (None = disabled)
        are ``phase_stance_mask``, an ``(n_dense,)`` bool mask of stance samples along
        the dense times, and ``phase_swing_envelope``, an ``(n_dense,)`` swing
        intensity in [0, 1].

        Returns
        -------
        tuple[EvalWrapper, EvalWrapper, EvalWrapper]
            ``(traj_x, traj_y, traj_z)``, each with an ``.eval(t)`` method over ``[0, swing_duration]``.
        """
        p = self.params
        n_knots_inner = p.n_knots
        n_knots_total = n_knots_inner + 2  # Including the two boundary points

        # ---- Time setup ----
        dense_times = np.linspace(0.0, swing_duration, p.n_dense)
        knot_times = np.linspace(0.0, swing_duration, n_knots_total)

        # ---- Pre-compute the basis matrix ----
        B_full = _build_basis_matrix(knot_times, dense_times)
        # Separate boundary columns from interior columns
        B_start = B_full[:, 0]  # (n_dense,)
        B_end = B_full[:, -1]  # (n_dense,)
        B_inner = B_full[:, 1:-1]  # (n_dense, n_knots_inner)

        # Boundary contribution: (n_dense, 3) — contribution of the fixed endpoints to the trajectory
        boundary_contrib = B_start[:, None] * liftoff_pos[None, :] + B_end[:, None] * landing_pos[None, :]

        # ---- Interpolate the reference trajectory onto dense_times ----
        ref_interp = np.column_stack(
            [np.interp(dense_times, ref_times, ref_traj[:, d]) for d in range(3)]
        )  # (n_dense, 3)

        # ---- Clearance envelope: sin²(πφ), zero at endpoints, maximal in the middle ----
        phi = dense_times / max(swing_duration, 1e-12)
        swing_envelope = np.sin(np.pi * phi) ** 2
        clearance = p.h_clearance * 2.0 * swing_envelope  # (n_dense,)

        # ---- Height upper bound ----
        z_ceiling = max(liftoff_pos[2], landing_pos[2]) + p.ceiling_margin

        # ---- Initialize mu: linearly interpolate the interior points ----
        mu = np.zeros((n_knots_inner, 3), dtype=np.float64)
        for i in range(n_knots_inner):
            alpha = (i + 1) / (n_knots_inner + 1)
            mu[i] = (1 - alpha) * liftoff_pos + alpha * landing_pos

        # ---- Noise standard deviation per channel ----
        noise_std = np.array([p.noise_std_xy, p.noise_std_xy, p.noise_std_z], dtype=np.float64)

        # ---- MPPI iterations ----
        for _iter in range(p.n_iterations):
            # Sampling: (n_samples, n_knots_inner, 3)
            noise = self.rng.normal(size=(p.n_samples, n_knots_inner, 3))
            noise *= noise_std[None, None, :]
            samples = mu[None, :, :] + noise
            samples[0] = mu  # Sample 0 = current mean (noise-free)

            # Expand to dense trajectories: (n_samples, n_dense, 3)
            # trajs = boundary_contrib + B_inner @ samples_per_channel
            trajs = boundary_contrib[None, :, :] + np.einsum("hk,ikd->ihd", B_inner, samples)

            # Terrain query: (n_samples, n_dense)
            terrain_z = np.empty((p.n_samples, p.n_dense), dtype=np.float64)
            for i in range(p.n_samples):
                terrain_z[i] = terrain.height_batch(trajs[i, :, 0], trajs[i, :, 1])

            # Compute costs
            costs = self._cost_batch(
                trajs,
                terrain_z,
                ref_interp,
                clearance,
                z_ceiling,
                phase_stance_mask=phase_stance_mask,
                phase_swing_envelope=phase_swing_envelope,
            )

            # Softmax weighting
            beta = costs.min()
            weights = np.exp(-(costs - beta) / p.temperature)
            w_sum = weights.sum()
            if w_sum < 1e-30:
                weights = np.ones(p.n_samples) / p.n_samples
            else:
                weights /= w_sum

            # Update mu
            mu = np.einsum("i,ijk->jk", weights, samples)

        # ---- Final trajectory ----
        final_traj = boundary_contrib + B_inner @ mu  # (n_dense, 3)

        return (
            EvalWrapper(dense_times, final_traj[:, 0]),
            EvalWrapper(dense_times, final_traj[:, 1]),
            EvalWrapper(dense_times, final_traj[:, 2]),
        )

    def _cost_batch(
        self,
        trajs: np.ndarray,  # (K, N, 3)
        terrain_z: np.ndarray,  # (K, N)
        ref_interp: np.ndarray,  # (N, 3)
        clearance: np.ndarray,  # (N,)
        z_ceiling: float,
        phase_stance_mask: np.ndarray | None = None,  # (N,) bool
        phase_swing_envelope: np.ndarray | None = None,  # (N,) float [0,1]
    ) -> np.ndarray:
        """Compute costs for all samples in batch.

        Parameters
        ----------
        trajs : np.ndarray
            (n_samples, n_dense, 3) sampled trajectories.
        terrain_z : np.ndarray
            (n_samples, n_dense) terrain height at each trajectory's xy.
        ref_interp : np.ndarray
            (n_dense, 3) reference trajectory (interpolated onto dense_times).
        clearance : np.ndarray
            (n_dense,) clearance envelope.
        z_ceiling : float
            upper bound on height in z.
        phase_stance_mask : np.ndarray | None
            (n_dense,) bool — True = stance phase.
        phase_swing_envelope : np.ndarray | None
            (n_dense,) float — swing phase intensity [0,1].

        Returns
        -------
        costs : np.ndarray
            (n_samples,) total cost per trajectory.
        """
        p = self.params
        K = trajs.shape[0]
        costs = np.zeros(K, dtype=np.float64)

        # 1. Tracking cost: ||traj - ref||²
        track_err = trajs - ref_interp[None, :, :]  # (K, N, 3)
        costs += p.w_track * np.sum(track_err**2, axis=(1, 2))

        # 2. Smoothness cost: second-order differences
        if trajs.shape[1] > 2:
            d2 = np.diff(trajs, n=2, axis=1)  # (K, N-2, 3)
            costs += p.w_smooth * np.sum(d2**2, axis=(1, 2))

        # 3. Terrain penetration in z
        floor_z = terrain_z + p.h_contact_z + clearance[None, :]  # (K, N)
        pen_z = np.maximum(floor_z - trajs[:, :, 2], 0.0)  # (K, N)
        costs += p.w_terrain * np.sum(pen_z**2, axis=1)

        # 4. Height upper bound
        over_z = np.maximum(trajs[:, :, 2] - z_ceiling, 0.0)  # (K, N)
        costs += p.w_ceiling * np.sum(over_z**2, axis=1)

        # 5. Stance ground penalty: penalize foot leaving ground during stance
        if p.w_stance_ground > 0.0 and phase_stance_mask is not None:
            ground_z = terrain_z + p.h_contact_z  # (K, N)
            stance_err = (trajs[:, :, 2] - ground_z) * phase_stance_mask[None, :]  # (K, N)
            costs += p.w_stance_ground * np.sum(stance_err**2, axis=1)

        # 6. Swing clearance reward: reward clearance above terrain during swing
        if p.w_swing_clearance_reward > 0.0 and phase_swing_envelope is not None:
            ground_z = terrain_z + p.h_contact_z  # (K, N)
            swing_clearance = (trajs[:, :, 2] - ground_z) * phase_swing_envelope[None, :]  # (K, N)
            # Negative cost = reward for being above ground
            costs -= p.w_swing_clearance_reward * np.sum(np.maximum(swing_clearance, 0.0), axis=1)

        return costs
