"""Bipedal start-transition IK for motion-tracking preprocessing.

Ports OpenTrack's :file:`start_end_transition_handler.py` to Eden as a
robot-agnostic helper. The output is a per-frame ``qpos`` array
``(T, 7 + nDOF)`` representing a "stand still → step → smooth merge" segment
that lands on the first frame of an arbitrary mocap clip.

Math sketch (per IK iteration)::

    err = [pos_swing_target - pos_swing_curr,
           pos_stance_target - pos_stance_curr,
           rot_err_swing, rot_err_stance,
           com_xy_target - com_xy_curr]
    J   = stack([Jp_swing, Jp_stance, Jr_swing, Jr_stance, J_com])
    H   = Jᵀ W J + λ I + α W_qpos
    dq  = clip(H⁻¹ Jᵀ W err, ±dq_bound)
    qpos = integrate_tangent(qpos, dq, dt=1)

Differences from OpenTrack's reference solver
---------------------------------------------

* **FK / Jacobian via Genesis, not MuJoCo.** Uses
  ``RigidEntity.get_jacobian(link, local_point=...)``
  (https://github.com/Genesis-Embodied-AI/Genesis/pull/1353) for the spatial
  Jacobian at the foot's contact point. ``set_qpos`` triggers Genesis's forward
  kinematics under the hood, so the per-iteration ``mj_forward`` loop becomes a
  single ``robot.set_qpos`` call, and the module needs no MuJoCo runtime.
* Damped least squares + per-DOF clip in place of OpenTrack's OSQP
  box-constrained QP. DLS + post-clip matches the inactive-bound case
  numerically and avoids a new ``osqp`` dependency.
* Robot-agnostic: every bipedal-specific bit (foot link names, contact-point
  offsets, upper-body joints, stance ``qpos``, qpos-bias weights) is read from
  a :class:`BipedalIKProfile`. The G1 instance lives in
  :mod:`eden.options.robots.unitree_g1`. The profile carries the foot
  contact-point offset explicitly (e.g. ``(0.04, 0, -0.037)`` for G1),
  removing the dependence on MJCF site definitions.

Convention
----------

* Quaternions are wxyz, Hamilton, unit-norm (Eden / MuJoCo / Genesis default).
* Free-joint angular velocity is body-frame; quaternion integration is
  right-multiplication ``q_new = q ⊗ exp(½·ω·dt)``. Verified to match
  ``mj_integratePos`` exactly in unit tests.
* DOF order is ``dofs_name`` everywhere — both ``set_qpos`` and
  ``get_jacobian`` use it (the Jacobian column reorder happens in Eden's
  wrapper).
"""

from __future__ import annotations

import fnmatch
from typing import TYPE_CHECKING

import numpy as np
import torch
from genesis.utils.geom import (
    R_to_quat,
    quat_to_R,
    slerp,
    transform_by_quat,
    transform_quat_by_quat,
    xyz_to_quat,
)

from eden.options.entities import RobotOptions
from eden.options.options import ConfigurableOptions
from eden.utils.geom import yaw_from_quat

if TYPE_CHECKING:
    from eden.entities.rigid import RigidEntity


# ---------------------------------------------------------------------------
# Tuning constants — IK stopping criteria, weight balance, numeric epsilons
# ---------------------------------------------------------------------------

# Per-iteration stopping criteria for the swing-foot DLS IK. Each tuple is
# ``(minimum iter, error tolerance)``: once iter >= n and err_norm < tol, we
# bail. Walked through in order, so tighter tolerances win when they fire.
# Plus a plateau check ``|prev_err - err| < _IK_PLATEAU_EPS`` (handled at the
# call site).
_IK_STOPS: tuple[tuple[int, float], ...] = ((0, 1e-3), (50, 5e-3), (100, 1e-2))
_IK_PLATEAU_EPS: float = 1e-6

# Inside ``_solve_swing_foot_ik``'s weight matrix we down-scale rotation
# error vs translation error (rotation residuals are radians, much larger
# than centimetre-scale translations) and down-scale the CoM residual (it's
# a soft balance hint, not a hard target).
_ROT_WEIGHT_BALANCE: float = 0.1
_COM_WEIGHT_BALANCE: float = 0.5

# Minimum frames per IK step for the 5th-order swing-foot bump to look like
# a step rather than a single jump; below this the smoothstep collapses.
_MIN_STEP_FRAMES: int = 4

# Below this magnitude the SE(3) tangent angular component is treated as a
# zero rotation (identity quat delta) — guards the ``sin(half)/omega_norm``
# division in ``integrate_qpos_tangent``.
_OMEGA_ZERO_EPS: float = 1e-12


# ---------------------------------------------------------------------------
# Robot-agnostic IK profile
# ---------------------------------------------------------------------------


class BipedalIKProfile(ConfigurableOptions):
    """Robot-specific knobs required to synthesize a start transition.

    The profile is the single source of truth for everything the IK needs to
    know about a humanoid biped: which robot to build (Eden ``RobotOptions``),
    which links carry the feet, where the contact point is on each foot, and
    the per-DOF weighting that controls how "loud" the IK is allowed to be.

    The stance pose is derived from ``robot_options.default_root_pos`` and
    ``robot_options.default_dofs_pos`` (the canonical Eden idiom) — the
    profile doesn't carry a redundant ``default_qpos`` array. Per-DOF bias
    weights use fnmatch patterns against ``robot_options.dofs_name``, with
    explicit fields for the 6 free-joint base DOFs.

    Precondition for consumers
    --------------------------
    Any function that takes a ``BipedalIKProfile`` and then builds a Genesis
    scene from ``robot_options`` (e.g. :func:`generate_start_transition`,
    :class:`~eden.datasets.modifiers.SmoothStartTransition`) requires
    :func:`eden.init` (which drives ``genesis.init``) to have run first.
    Callers may pass a profile around freely without Genesis init — only the
    scene-building entry points need it. See
    :func:`eden.utils.genesis_runtime.transient_robot_env` for the shared
    precondition probe.

    Parameters
    ----------
    robot_options:
        Eden ``RobotOptions`` for the robot. ``default_root_pos`` +
        ``default_dofs_pos`` (fnmatch dict) determine the stance qpos that
        seeds the IK.
    left_foot_link, right_foot_link:
        Genesis link names whose Jacobian is taken w.r.t. each foot contact
        point. For G1, these are the ankle-roll links.
    left_foot_local_offset, right_foot_local_offset:
        Contact-point offset in the foot link's local frame, ``(x, y, z)`` in
        metres. Mirrors the ``pos="…"`` attribute of an MJCF foot site without
        depending on the MJCF site being declared. For G1, both feet are at
        ``(0.04, 0, -0.037)`` w.r.t. the ankle-roll link.
    upper_body_joint_patterns:
        Fnmatch patterns (e.g. ``"*_shoulder_*_joint"``). Joints matching any
        pattern have their qpos overridden with a cosine-eased blend from
        stance to clip-frame-0 across the transition — keeps the arms /
        torso from drifting into IK-feasible-but-ugly poses.
    base_lin_bias_weights, base_ang_bias_weights:
        Per-axis qpos-bias weight for the floating-base translation (X/Y/Z)
        and angular velocity (roll/pitch/yaw). Defaults to uniform 1.0;
        biped users typically bump yaw to ~2.5 to keep the stance facing
        forward through the synthesized step.
    joint_bias_patterns:
        Fnmatch dict mapping a joint-name pattern to a qpos-bias weight.
        Unmatched joints default to 1.0. Set ``"*_hip_roll_joint": 0.0`` etc.
        to free those joints from the bias regularizer.
    qpos_bias_scale:
        Scalar multiplier ``α`` applied to the full qpos-bias diagonal in
        the QP regularizer. OpenTrack uses 0.5.
    ee_weights:
        Five-entry vector ``(w_swing_foot, w_stance_foot, w_swing_rot,
        w_stance_rot, w_com)``. Rotation weights are internally further scaled
        by 0.1 to balance against position; CoM by 0.5. Defaults to all ones.
    damping:
        Levenberg-Marquardt damping ``λ`` on the normal-equation Hessian.
    dq_bound:
        Per-DOF clip on the IK step magnitude (rad / frame). Matches the
        OpenTrack OSQP box-bound.
    """

    robot_options: RobotOptions
    left_foot_link: str = "left_ankle_roll_link"
    right_foot_link: str = "right_ankle_roll_link"
    left_foot_local_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    right_foot_local_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    upper_body_joint_patterns: tuple[str, ...] = ()
    base_lin_bias_weights: tuple[float, float, float] = (1.0, 1.0, 1.0)
    base_ang_bias_weights: tuple[float, float, float] = (1.0, 1.0, 1.0)
    joint_bias_patterns: dict[str, float] = {}
    qpos_bias_scale: float = 0.5
    ee_weights: tuple[float, float, float, float, float] = (1.0, 1.0, 1.0, 1.0, 1.0)
    damping: float = 1e-3
    dq_bound: float = 10.0


# ---------------------------------------------------------------------------
# Profile resolvers — derive the runtime numpy arrays from the options
# ---------------------------------------------------------------------------


def _resolve_default_qpos(profile: BipedalIKProfile) -> np.ndarray:
    """Build the stand-still qpos ``(7 + nDOF,)`` from the profile's robot options.

    Layout: ``[root_pos (3), wxyz identity (4), joint angles in dofs_name order]``.
    ``default_dofs_pos`` is fnmatch-expanded against ``dofs_name``; unmatched
    joints default to ``0.0``. Exact name overrides win over wildcard patterns.
    """
    robot = profile.robot_options
    dofs_name = robot.dofs_name
    spec = robot.default_dofs_pos
    dof_q = np.zeros(len(dofs_name), dtype=np.float64)
    for i, name in enumerate(dofs_name):
        if name in spec:
            dof_q[i] = float(spec[name])
            continue
        for pattern, value in spec.items():
            if any(ch in pattern for ch in "*?[") and fnmatch.fnmatch(name, pattern):
                dof_q[i] = float(value)
                break
    root = np.asarray(robot.default_root_pos, dtype=np.float64)
    quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return np.concatenate([root, quat, dof_q])


def _resolve_qpos_bias_weights(profile: BipedalIKProfile) -> np.ndarray:
    """Build the per-velocity-DOF (``nv = 6 + nDOF``) bias weight array.

    First 6 entries are the floating-base twist (``base_lin_bias_weights`` ⊕
    ``base_ang_bias_weights``). Joint entries default to 1.0 and are
    overridden by the first matching ``joint_bias_patterns`` entry, with
    exact name overrides winning over wildcards.
    """
    robot = profile.robot_options
    dofs_name = robot.dofs_name
    weights = np.ones(6 + len(dofs_name), dtype=np.float64)
    weights[0:3] = profile.base_lin_bias_weights
    weights[3:6] = profile.base_ang_bias_weights
    patterns = profile.joint_bias_patterns or {}
    for i, name in enumerate(dofs_name):
        if name in patterns:
            weights[6 + i] = float(patterns[name])
            continue
        for pattern, value in patterns.items():
            if any(ch in pattern for ch in "*?[") and fnmatch.fnmatch(name, pattern):
                weights[6 + i] = float(value)
                break
    return weights


def _resolve_upper_body_addr(profile: BipedalIKProfile, dofs_name: list[str]) -> np.ndarray:
    """Qpos indices for joints whose name matches any ``upper_body_joint_patterns``.

    Returned indices are into the full qpos layout (``7 + position in
    dofs_name``), so they index 1-DOF hinge / slide joints directly.
    """
    patterns = profile.upper_body_joint_patterns
    if not patterns:
        return np.zeros(0, dtype=np.int64)
    addrs = []
    for i, name in enumerate(dofs_name):
        for pattern in patterns:
            if name == pattern or (any(ch in pattern for ch in "*?[") and fnmatch.fnmatch(name, pattern)):
                addrs.append(7 + i)
                break
    return np.array(addrs, dtype=np.int64)


# ---------------------------------------------------------------------------
# Quaternion + SE(3) helpers
# ---------------------------------------------------------------------------
#
# Wherever possible we route through ``genesis.utils.geom`` so the IK math
# uses the same convention as Eden's training-loop kinematics. The helpers
# below are the few operations Genesis doesn't expose directly:
#
# * ``_slerp_rotation_matrix`` — sandwich SLERP between R→quat→R conversions.
# * ``_small_angle_rot_error`` — first-order axis-angle error for the IK
#   residual (vex of the skew-symmetric part of ``R_err - I``).
# * ``integrate_qpos_tangent`` — SE(3) tangent → qpos integration matching
#   ``mj_integratePos`` for floating-base + hinge robots.
# * ``_interp_qpos`` — element-wise qpos interpolation that mixes linear
#   blending for translation/joints and SLERP for the base quat.


def _slerp_rotation_matrix(R1: np.ndarray, R2: np.ndarray, alpha: float) -> np.ndarray:
    """SLERP between two rotation matrices via the quaternion path."""
    return quat_to_R(slerp(R_to_quat(R1), R_to_quat(R2), np.array(alpha)))


def _cosine_easing(alpha: float) -> float:
    """Cosine smoothstep ``0.5·(1 − cos(π·α))`` — zero velocity at α=0 and α=1.

    Used for both the upper-body blend and the merge alpha so easing in
    and out of the synthesized prefix reads smoothly to a human observer.
    """
    return 0.5 * (1.0 - np.cos(np.pi * alpha))


def _smoothstep5(s: float) -> float:
    """5th-order smoothstep ``10s³ − 15s⁴ + 6s⁵`` (Perlin's smootherstep).

    ``f(0) = 0``, ``f(1) = 1``, ``f'(0) = f'(1) = f''(0) = f''(1) = 0``.
    Used for the swing-foot's vertical bump trajectory.
    """
    return 10 * s**3 - 15 * s**4 + 6 * s**5


def _small_angle_rot_error(R_target: np.ndarray, R_current: np.ndarray) -> np.ndarray:
    """``vex(A - Aᵀ) / 2`` with ``A = R_targetᵀ·R_current`` — first-order axis-angle error."""
    R_err = R_target.T @ R_current
    return 0.5 * np.array(
        [
            R_err[2, 1] - R_err[1, 2],
            R_err[0, 2] - R_err[2, 0],
            R_err[1, 0] - R_err[0, 1],
        ]
    )


def integrate_qpos_tangent(qpos: np.ndarray, dq: np.ndarray, dt: float = 1.0) -> np.ndarray:
    """Integrate a tangent-space velocity ``dq`` into a floating-base ``qpos``.

    Matches ``mujoco.mj_integratePos`` exactly for the floating-base + hinge
    joint case (which is what every bipedal humanoid uses):

    * ``qpos[0:3] += dq[0:3] * dt`` — base translation in world frame.
    * ``qpos[3:7] = qpos[3:7] ⊗ exp(½ · dq[3:6] · dt)`` — base orientation
      right-multiplied by the quaternion exponential of the **body-frame**
      angular velocity. Convention pinned by a unit test against
      ``mj_integratePos``.
    * ``qpos[7:] += dq[6:] * dt`` — 1-DOF joints integrate linearly.

    Parameters
    ----------
    qpos:
        Shape ``(7 + nDOF,)``. Layout ``[xyz, wxyz, joints]``.
    dq:
        Shape ``(6 + nDOF,)``. Layout ``[lin, ang, joint_vels]``.
    dt:
        Integration step (typically 1.0 since the IK's ``dq`` is the desired
        per-iteration displacement, not a per-second velocity).
    """
    # One allocation (a float64 copy): the integrator mutates the buffer in place.
    out = qpos.astype(np.float64, copy=True)
    out[0:3] += dq[0:3] * dt
    omega = dq[3:6] * dt
    omega_norm = float(np.linalg.norm(omega))
    if omega_norm > _OMEGA_ZERO_EPS:
        half = omega_norm / 2.0
        s = np.sin(half) / omega_norm
        dq_quat = np.array([np.cos(half), omega[0] * s, omega[1] * s, omega[2] * s])
    else:
        dq_quat = np.array([1.0, 0.0, 0.0, 0.0])
    # ``transform_quat_by_quat(v, u)`` returns ``u ⊗ v`` — i.e. left-multiplies
    # ``v`` by ``u`` (see its docstring). We want ``q_new = q_old ⊗ dq_quat``
    # (right-multiply by the body-frame angular exponential), so pass
    # ``v=dq_quat, u=q_old``. The genesis impl allocates its own output,
    # so reading ``out[3:7]`` as ``u`` doesn't alias.
    out[3:7] = transform_quat_by_quat(dq_quat, out[3:7])
    out[3:7] /= np.linalg.norm(out[3:7])
    out[7:] += dq[6:] * dt
    return out


def _interp_qpos(q_start: np.ndarray, q_end: np.ndarray, alpha: float) -> np.ndarray:
    """Element-wise interp on qpos: linear for translation/joints, SLERP for the base quat."""
    out = np.zeros_like(q_start)
    out[0:3] = (1 - alpha) * q_start[0:3] + alpha * q_end[0:3]
    out[3:7] = slerp(q_start[3:7], q_end[3:7], np.array(alpha))
    out[7:] = (1 - alpha) * q_start[7:] + alpha * q_end[7:]
    return out


# ---------------------------------------------------------------------------
# Genesis FK helpers — read site pose at a contact point on a link
# ---------------------------------------------------------------------------


def _set_qpos(robot: "RigidEntity", qpos: np.ndarray) -> None:
    """``robot.set_qpos`` with a numpy qpos. Triggers FK as a side effect."""
    t = torch.as_tensor(qpos, dtype=torch.float32, device=robot.device).unsqueeze(0)
    # ``zero_velocity=False`` matches MuJoCo's ``mj_forward`` behaviour: we
    # only want kinematics updated, not the velocity buffer cleared on each
    # IK iteration.
    robot.set_qpos(t, zero_velocity=False, skip_forward=False)


def _site_pos_rot(
    robot: "RigidEntity",
    link_idx: int,
    local_offset: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the world-frame ``(position, rotation matrix)`` of a contact site.

    Assumes ``robot.set_qpos`` has already been called for the current qpos.
    The site is a point fixed in the link's local frame at ``local_offset``;
    site orientation is taken to equal the link's orientation (matches MJCF
    sites declared without an explicit ``quat``, which is the G1 convention).

    Index on-device before the host sync so we transfer 3 / 4 floats instead
    of the full ``(n_links, 3/4)`` per-link tensor. This is the IK's inner
    hot path — for G1 (30 links) the per-call data transfer drops 30×.
    """
    link_pos = robot.get_links_pos(envs_idx=0)[0, link_idx].cpu().numpy().astype(np.float64)
    link_quat = robot.get_links_quat(envs_idx=0)[0, link_idx].cpu().numpy().astype(np.float64)
    offset_world = transform_by_quat(np.asarray(local_offset, dtype=np.float64), link_quat)
    site_pos = link_pos + offset_world
    site_rot = quat_to_R(link_quat)
    return site_pos, site_rot


def _foot_link_indices(robot: "RigidEntity", profile: BipedalIKProfile) -> tuple[int, int]:
    """Resolve ``(left_idx, right_idx)`` into ``get_links_pos``'s link axis."""
    _, idx = robot.find_named_links_idx_local([profile.left_foot_link, profile.right_foot_link], preserve_order=True)
    return int(idx[0]), int(idx[1])


def _blend_upper_body_in_place(
    transition: np.ndarray,
    upper_addr: np.ndarray,
    q_default: np.ndarray,
    q_first: np.ndarray,
) -> None:
    """Overwrite upper-body DOFs across the transition with a cosine-eased blend.

    The IK leaves them physically feasible but visually unmotivated; cosine
    easing means arms accelerate gradually instead of snapping into motion
    when the default-hold ends, and decelerate smoothly into clip-frame-0.

    Mutates ``transition`` in place. No-op when ``upper_addr`` is empty.
    """
    if upper_addr.size == 0:
        return
    n_trans = transition.shape[0]
    delta = q_first[upper_addr] - q_default[upper_addr]
    for t in range(n_trans):
        a = _cosine_easing(t / max(n_trans - 1, 1))
        transition[t, upper_addr] = q_default[upper_addr] + delta * a


# ---------------------------------------------------------------------------
# Single-step IK + multi-step assembler
# ---------------------------------------------------------------------------


def _solve_swing_foot_ik(
    robot: "RigidEntity",
    profile: BipedalIKProfile,
    qpos_bias_weights: np.ndarray,
    *,
    swing_link,  # Genesis RigidLink
    stance_link,
    swing_link_idx: int,
    stance_link_idx: int,
    swing_local_offset: np.ndarray,
    stance_local_offset: np.ndarray,
    fixed_foot_pos: np.ndarray,
    fixed_foot_rot: np.ndarray,
    swing_foot_pos_start: np.ndarray,
    swing_foot_rot_start: np.ndarray,
    swing_foot_pos_end: np.ndarray,
    swing_foot_rot_end: np.ndarray,
    q_ref: np.ndarray,
    traj_len: int,
    foot_h: float,
    com_off: float,
    max_iter: int,
) -> np.ndarray:
    """Run per-frame DLS IK for one foot step, return qpos trajectory ``(traj_len+1, nq)``.

    ``qpos_bias_weights`` is the pre-resolved per-velocity-DOF bias array
    (``_resolve_qpos_bias_weights(profile)``) — passed in instead of read off
    the profile so the resolution happens once per ``apply()`` and not once
    per IK iteration.
    """
    n_dofs = robot.n_dofs  # 6 + n_actuated (free joint contributes 6)
    eye_n = np.eye(n_dofs)
    Jcom = np.zeros((2, n_dofs))
    Jcom[:, :2] = np.eye(2)  # ∂(base XY)/∂(twist XYZ) = I on the first two rows

    w_swing, w_stance, w_swing_rot, w_stance_rot, w_com = profile.ee_weights
    W_diag = np.concatenate(
        [
            w_swing * np.ones(3),
            w_stance * np.ones(3),
            w_swing_rot * _ROT_WEIGHT_BALANCE * np.ones(3),
            w_stance_rot * _ROT_WEIGHT_BALANCE * np.ones(3),
            w_com * _COM_WEIGHT_BALANCE * np.ones(2),
        ]
    )
    W = np.diag(W_diag)

    if qpos_bias_weights.shape != (n_dofs,):
        raise ValueError(f"qpos_bias_weights has shape {qpos_bias_weights.shape}, expected ({n_dofs},)")
    W_qpos = np.diag(qpos_bias_weights)

    swing_local_t = torch.as_tensor(swing_local_offset, dtype=torch.float32, device=robot.device)
    stance_local_t = torch.as_tensor(stance_local_offset, dtype=torch.float32, device=robot.device)

    q_cur = q_ref.copy().astype(np.float64)
    out_traj = np.zeros((traj_len + 1, q_cur.shape[0]), dtype=np.float64)

    for t in range(traj_len + 1):
        alpha = t / max(traj_len, 1)
        xy_des = swing_foot_pos_start[0:2] + (swing_foot_pos_end[0:2] - swing_foot_pos_start[0:2]) * _cosine_easing(
            alpha
        )
        # 5th-order vertical bump mirrored around the midpoint — peaks at alpha=0.5.
        z_des = swing_foot_pos_start[2] + foot_h * _smoothstep5(min(alpha, 1 - alpha) * 2)

        swing_pos_target = np.array([xy_des[0], xy_des[1], z_des])
        swing_rot_target = _slerp_rotation_matrix(swing_foot_rot_start, swing_foot_rot_end, alpha)
        com_target = 0.5 * (xy_des + fixed_foot_pos[0:2]) + np.array([com_off, 0.0])

        prev_err = float("inf")
        for it in range(max_iter):
            _set_qpos(robot, q_cur)
            swing_pos, swing_rot = _site_pos_rot(robot, swing_link_idx, swing_local_offset)
            stance_pos, stance_rot = _site_pos_rot(robot, stance_link_idx, stance_local_offset)

            err = np.concatenate(
                [
                    swing_pos_target - swing_pos,
                    fixed_foot_pos - stance_pos,
                    _small_angle_rot_error(swing_rot, swing_rot_target),
                    _small_angle_rot_error(stance_rot, fixed_foot_rot),
                    com_target - q_cur[0:2],
                ]
            )
            err_norm = float(np.linalg.norm(err))
            if any(it >= n and err_norm < tol for n, tol in _IK_STOPS):
                break
            if abs(prev_err - err_norm) < _IK_PLATEAU_EPS:
                break
            prev_err = err_norm

            # Spatial Jacobian at the foot contact point. Genesis returns
            # ``(n_envs, 6, n_dofs)`` or ``(6, n_dofs)`` for n_envs=0; we run
            # with n_envs=1, so squeeze the env axis. Columns are in
            # dofs_name order (with the free joint's 6 tangent DOFs prepended),
            # which is what our integrator expects.
            J_swing = robot.get_jacobian(swing_link, local_point=swing_local_t).cpu().numpy()
            J_stance = robot.get_jacobian(stance_link, local_point=stance_local_t).cpu().numpy()
            if J_swing.ndim == 3:
                J_swing = J_swing[0]
                J_stance = J_stance[0]
            Jp_swing, Jr_swing = J_swing[:3], J_swing[3:]
            Jp_stance, Jr_stance = J_stance[:3], J_stance[3:]

            J = np.vstack([Jp_swing, Jp_stance, Jr_swing, Jr_stance, Jcom])
            H = J.T @ W @ J + profile.damping * eye_n + profile.qpos_bias_scale * W_qpos
            b = J.T @ W @ err
            try:
                dq = np.linalg.solve(H, b)
            except np.linalg.LinAlgError:
                # H is SPD by construction, but be defensive for pathological inputs.
                dq, *_ = np.linalg.lstsq(H, b, rcond=None)
            np.clip(dq, -profile.dq_bound, profile.dq_bound, out=dq)

            q_cur = integrate_qpos_tangent(q_cur, dq, 1.0)

        out_traj[t] = q_cur

    return out_traj


def _align_stance_foot(
    robot: "RigidEntity",
    profile: BipedalIKProfile,
    q_start: np.ndarray,
    q_end: np.ndarray,
    swing_side: str,
    stance_link_idx: int,
    stance_local_offset: np.ndarray,
) -> np.ndarray:
    """Offset ``q_end`` base XY so the stance foot is co-located with its start position."""
    _set_qpos(robot, q_start)
    p0, _ = _site_pos_rot(robot, stance_link_idx, stance_local_offset)
    _set_qpos(robot, q_end)
    p1, _ = _site_pos_rot(robot, stance_link_idx, stance_local_offset)

    out = q_end.copy()
    out[:2] += p0[:2] - p1[:2]
    return out


def _compute_step_distance(
    robot: "RigidEntity",
    profile: BipedalIKProfile,
    q_start: np.ndarray,
    q_end: np.ndarray,
) -> float:
    """Absolute difference in inter-foot XY distance between two qpos states."""
    left_idx, right_idx = _foot_link_indices(robot, profile)
    left_off = np.asarray(profile.left_foot_local_offset)
    right_off = np.asarray(profile.right_foot_local_offset)

    _set_qpos(robot, q_start)
    l0, _ = _site_pos_rot(robot, left_idx, left_off)
    r0, _ = _site_pos_rot(robot, right_idx, right_off)
    _set_qpos(robot, q_end)
    l1, _ = _site_pos_rot(robot, left_idx, left_off)
    r1, _ = _site_pos_rot(robot, right_idx, right_off)
    return float(abs(np.linalg.norm(l1[:2] - r1[:2]) - np.linalg.norm(l0[:2] - r0[:2])))


def _yaw_align_trajectory(qpos_traj: np.ndarray) -> np.ndarray:
    """Rotate every frame so the first frame faces world-X."""
    out = qpos_traj.copy()
    q0 = out[0, 3:7] / np.linalg.norm(out[0, 3:7])
    # `yaw_from_quat`, not `quat_to_xyz(q0)[2]`: the genesis default is an
    # intrinsic Rx*Ry*Rz decomposition whose third angle is a body-z angle, not
    # the world-Z heading (they agree only when roll/pitch are ~0).
    yaw0 = float(yaw_from_quat(q0))
    q_inv = xyz_to_quat(np.array([0.0, 0.0, -yaw0]))
    c, s = np.cos(-yaw0), np.sin(-yaw0)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    for i in range(out.shape[0]):
        # ``transform_quat_by_quat(v, u)`` returns ``u ⊗ v``; passing
        # ``v=current_quat, u=q_inv`` left-multiplies by the inverse-yaw delta.
        q_new = transform_quat_by_quat(out[i, 3:7], q_inv)
        out[i, 3:7] = q_new / np.linalg.norm(q_new)
        out[i, :3] = Rz @ out[i, :3]
    return out


def generate_start_transition(
    robot: "RigidEntity",
    profile: BipedalIKProfile,
    qpos_traj: np.ndarray,
    *,
    framerate: float,
    default_pose_sec: float = 1.0,
    single_step_threshold_m: float = 0.1,
    transition_sec: float = 0.3,
    foot_height_m: float = 0.05,
    com_xy_offset_m: float = 0.03,
    step_sec: float = 0.3,
    double_stance_frames: int = 3,
    start_foot: str = "right",
    max_ik_iters: int = 200,
) -> np.ndarray:
    """Generate ``stand → step(s) → SLERP merge → clip`` and return the full qpos series.

    Parameters
    ----------
    robot:
        Pre-built Eden ``RigidEntity`` for the robot in ``profile.robot_options``.
        Must live in a single-env scene.
    profile:
        Robot-specific IK spec, including the contact-point offsets and stance
        qpos.
    qpos_traj:
        Source clip ``(T, 7 + nDOF)`` in the same DOF order as the robot's
        ``dofs_name``.
    framerate:
        Hz. Used to translate ``*_sec`` parameters into integer frame counts.
    default_pose_sec:
        Seconds to hold the default stance pose before stepping.
    single_step_threshold_m:
        Change in inter-foot XY distance (m) between the stance pose and clip frame 0 up to
        which a single step is used instead of two.
    transition_sec:
        Duration (s) of the SLERP merge between the generated start and the source clip.
    foot_height_m:
        Peak swing-foot height (m) during a step.
    com_xy_offset_m:
        Offset (m) along world +x (the clip's facing after yaw alignment) of the base-XY
        balance target from the midpoint between the feet during a step.
    step_sec:
        Duration (s) of a single step.
    double_stance_frames:
        Number of double-stance frames inserted between steps.
    start_foot:
        Which foot lifts first — ``"right"`` matches OpenTrack's default.
    max_ik_iters:
        Maximum IK iterations per frame.

    Returns
    -------
    np.ndarray
        ``(T_default + T_steps + T_merge + T_clip, 7 + nDOF)``: the default-pose hold, the IK
        step(s), the merge into clip frame 0, then the XY-shifted clip.
    """
    # Resolve once per call — these are O(nDOF) fnmatch loops, cheap, but
    # we don't want to pay them per IK iteration.
    q_default = _resolve_default_qpos(profile)
    qpos_bias = _resolve_qpos_bias_weights(profile)
    upper_addr = _resolve_upper_body_addr(profile, list(profile.robot_options.dofs_name))

    if qpos_traj.shape[1] != q_default.shape[0]:
        raise ValueError(f"qpos_traj has nq={qpos_traj.shape[1]} but the profile's robot has nq={q_default.shape[0]}")

    default_pose_len = max(int(round(default_pose_sec * framerate)), 0)
    transition_len = max(int(round(transition_sec * framerate)), 1)
    step_len = max(int(round(step_sec * framerate)), _MIN_STEP_FRAMES)

    # 1. Yaw-align: rotate the whole clip so frame 0 faces the same direction
    #    as the stance qpos. Without this, the IK has to handle a discontinuous
    #    yaw between stance and clip-frame-0, which it can't bridge in a few
    #    frames.
    qpos_traj = _yaw_align_trajectory(qpos_traj)

    # 2. Decide one vs two steps based on inter-foot distance change.
    #    ``q_default`` was resolved at the top of the function.
    q_first = qpos_traj[0].copy()
    step_dist = _compute_step_distance(robot, profile, q_default, q_first)
    num_steps = 2 if step_dist > single_step_threshold_m else 1

    # Resolve foot link handles + offsets once — they don't change across steps.
    left_link = robot.get_link(profile.left_foot_link)
    right_link = robot.get_link(profile.right_foot_link)
    left_off = np.asarray(profile.left_foot_local_offset)
    right_off = np.asarray(profile.right_foot_local_offset)
    left_idx, right_idx = _foot_link_indices(robot, profile)

    # 3. IK each step in turn. Step k swings whichever foot is the "current"
    #    start_foot; we then flip to the other for step k+1.
    step_traj_chunks: list[np.ndarray] = []
    q_step_start = q_default.copy()
    swing_side = start_foot
    last_step_end = q_default.copy()
    for k in range(num_steps):
        step_ratio = (k + 1) / num_steps

        if swing_side == "right":
            swing_link, swing_off, swing_idx = right_link, right_off, right_idx
            stance_link, stance_off, stance_idx = left_link, left_off, left_idx
        else:
            swing_link, swing_off, swing_idx = left_link, left_off, left_idx
            stance_link, stance_off, stance_idx = right_link, right_off, right_idx

        q_end_aligned = _align_stance_foot(robot, profile, q_step_start, q_first, swing_side, stance_idx, stance_off)

        # Foot start/end world poses at the step boundaries.
        _set_qpos(robot, q_step_start)
        swing_pos_start, swing_rot_start = _site_pos_rot(robot, swing_idx, swing_off)
        stance_pos, stance_rot = _site_pos_rot(robot, stance_idx, stance_off)

        _set_qpos(robot, q_end_aligned)
        swing_pos_end_full, swing_rot_end = _site_pos_rot(robot, swing_idx, swing_off)
        # Partial step toward final pose so multi-step plans split the displacement.
        swing_pos_end = swing_pos_start + (swing_pos_end_full - swing_pos_start) * step_ratio

        q_ref = q_step_start if k == 0 else step_traj_chunks[-1][-1]
        step_traj = _solve_swing_foot_ik(
            robot,
            profile,
            qpos_bias,
            swing_link=swing_link,
            stance_link=stance_link,
            swing_link_idx=swing_idx,
            stance_link_idx=stance_idx,
            swing_local_offset=swing_off,
            stance_local_offset=stance_off,
            fixed_foot_pos=stance_pos,
            fixed_foot_rot=stance_rot,
            swing_foot_pos_start=swing_pos_start,
            swing_foot_rot_start=swing_rot_start,
            swing_foot_pos_end=swing_pos_end,
            swing_foot_rot_end=swing_rot_end,
            q_ref=q_ref,
            traj_len=step_len,
            foot_h=foot_height_m,
            com_off=com_xy_offset_m,
            max_iter=max_ik_iters,
        )
        step_traj_chunks.append(step_traj)
        last_step_end = step_traj[-1].copy()

        # Add double-stance dwell between steps (but not after the last step).
        if k < num_steps - 1 and double_stance_frames > 0:
            dwell = np.tile(step_traj[-1], (double_stance_frames, 1))
            step_traj_chunks.append(dwell)

        q_step_start = step_traj[-1]
        swing_side = "left" if swing_side == "right" else "right"

    transition = np.concatenate(step_traj_chunks, axis=0)

    # 4. Overwrite upper-body DOFs during the transition with a cosine-eased
    #    blend from the stance pose to clip-frame-0's upper body. ``upper_addr``
    #    was resolved at the top of the function.
    _blend_upper_body_in_place(transition, upper_addr, q_default, q_first)

    # The IK-end qpos picked up in step 3 (``last_step_end``) is the pre-override
    # state — arms at default, feet placed by the IK. The override above just
    # rewrote the upper-body slice. We need to bridge from the **post-override**
    # end state into the clip, otherwise step 6's merge would jump arms from
    # T-pose → default → T-pose at the transition→merge seam.
    last_step_end = transition[-1].copy()

    # 5. Compute XY offset to bridge IK endpoint and clip frame 0 (stance foot
    #    must stay at the same world position across the transition→clip seam).
    _set_qpos(robot, last_step_end)
    l_end, _ = _site_pos_rot(robot, left_idx, left_off)
    _set_qpos(robot, qpos_traj[0])
    l_clip, _ = _site_pos_rot(robot, left_idx, left_off)
    xy_offset = l_end[:2] - l_clip[:2]

    # 6. SLERP merge from IK endpoint to clip-frame-0. Cosine easing matches
    #    step 4. Using ``t / (N-1)`` (not ``t/N``) makes the last frame land
    #    exactly on ``q_merge_end``, so the merge→clip seam has two identical
    #    frames in a row instead of a 1/N jump.
    q_merge_start = last_step_end
    q_merge_end = qpos_traj[0].copy()
    q_merge_end[:2] += xy_offset
    merge = np.zeros((transition_len, q_merge_start.shape[0]), dtype=np.float64)
    for t in range(transition_len):
        a = _cosine_easing(t / max(transition_len - 1, 1))
        merge[t] = _interp_qpos(q_merge_start, q_merge_end, a)

    # 7. Apply XY offset to the rest of the clip.
    clip_shifted = qpos_traj.copy()
    clip_shifted[:, :2] += xy_offset

    # 8. Prepend default-pose hold.
    default_hold = np.tile(q_default, (default_pose_len, 1))

    return np.concatenate([default_hold, transition, merge, clip_shifted], axis=0)
