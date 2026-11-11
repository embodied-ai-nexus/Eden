"""
Offline TCRS pipeline: terrain-adaptive whole-body motion optimization.

Pipeline stages (per clip):
  1. Build a :class:`~eden.extensions.motion_optimization.tcrs.motion.MotionClip` from raw arrays
     (contact masks + gait-phase clock are derived on construction).
  2. Apply an SE(2) placement transform (yaw about the frame-0 pelvis, then xy translation).
  3. Build :class:`~eden.extensions.motion_optimization.tcrs.motion.TerrainReference` +
     ``FootstepResolver`` (snap footholds onto safe terrain support surfaces).
  4. MPPI-optimize every swing-foot trajectory on the mid-foot virtual point
     (:func:`precompute_mppi_foot`), then post-process: lateral separation clamps,
     side-wall pushes, capsule-based leg/foot collision resolution, stance latching.
  5. Filter the pelvis z with :class:`~eden.extensions.motion_optimization.tcrs.terrain_warp.SupportAwareRootZFilter`
     against a reach-bounded terrain-driven target.
  6. Solve per-frame whole-body Jacobian IK (with the lower-body stance stabilizer post-pass)
     and record raw / ghost-FK / optimized motion sets.
  7. Zero-phase lowpass the optimized output's lower body for export.

Ported from Mondo-Robotics/PMT ``TCRS/stair_mppi/mppi_foot_planner_smooth.py`` @ 88838ca
(offline pipeline only; viewer, plum mode, window/trajectory IK, curobo/drake backends and
batch CLI not ported). The small joint/body reorder helpers come from the same repo's
``TCRS/stair_mppi/minimal_mppi_demo.py``; the mapping builders were rewritten to resolve
joints by name via ``jnt_qposadr`` instead of actuator ids.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
from genesis.utils.geom import quat_to_R, transform_quat_by_quat, xyz_to_quat

import eden as en
from eden.options.extensions.motion_optimization import TCRSPlannerOptions, TCRSRobotOptions, merged_ik_solver_config
from eden.extensions.motion_optimization.tcrs.ik import FrameIKTargets, JacobianIK
from eden.extensions.motion_optimization.tcrs.motion import MotionClip, TerrainReference
from eden.extensions.motion_optimization.tcrs.mppi_foot import (
    MppiFootOptimizer,
    MppiFootParams,
    slerp_quaternion,
)
from eden.extensions.motion_optimization.tcrs.terrain_source import TerrainSource
from eden.extensions.motion_optimization.tcrs.terrain_warp import SupportAwareRootZFilter
from eden.utils.geom import segment_closest_points, yaw_from_quat

# Support-aware foot IK (:func:`_contact_point_scale_for_pose`): a stance toe/heel whose terrain
# height differs from the ankle's by more than IK_EDGE_HEIGHT_TOL (m) hangs over a step edge, and
# its IK point weight is scaled by IK_UNSUPPORTED_POINT_SCALE.
IK_EDGE_HEIGHT_TOL = 0.02
IK_UNSUPPORTED_POINT_SCALE = 0.20


# ---------------------------------------------------------------------------
# Dofs/links <-> MuJoCo joint/body reorder helpers (from minimal_mppi_demo.py)
# ---------------------------------------------------------------------------


def build_dofs_mapping(model: mujoco.MjModel, dofs_name: list[str]) -> list[int]:
    """Clip joint column i -> index into qpos[7:], resolved by joint name via jnt_qposadr."""
    mapping = []
    for name in dofs_name:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid == -1:
            raise ValueError(f"Joint '{name}' not found in the MuJoCo model")
        # int() coercion: `x in (enum, ...)` compares with the pybind enum on the left, and
        # mujoco >= 3.12's enums answer False to a numpy scalar there (plain `x == enum` is fine).
        if int(model.jnt_type[jid]) not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            raise ValueError(f"Joint '{name}' is not a 1-DOF joint")
        mapping.append(int(model.jnt_qposadr[jid]) - 7)
    return mapping


def build_links_mapping(model: mujoco.MjModel, links_name: list[str]) -> list[int]:
    """Build mapping from clip link index to MuJoCo body id: ``mapping[clip_idx] = mujoco_body_id``."""
    mapping = []
    for name in links_name:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid == -1:
            raise ValueError(f"Body '{name}' not found in MuJoCo model")
        mapping.append(bid)
    return mapping


def reorder_to_mujoco(values, mapping):
    """Reorder a (n_dofs,) clip-order vector into MuJoCo qpos[7:] order."""
    out = np.zeros_like(values)
    for policy_idx, mujoco_idx in enumerate(mapping):
        out[mujoco_idx] = values[policy_idx]
    return out


def reorder_from_mujoco(values, mapping):
    """Reorder a (n_dofs,) MuJoCo qpos[7:] vector into clip order."""
    out = np.zeros_like(values)
    for policy_idx, mujoco_idx in enumerate(mapping):
        out[policy_idx] = values[mujoco_idx]
    return out


def update_robot_pose(data, root_pos, root_quat, dofs_qpos):
    """Write a root pose + joint vector into a MuJoCo data (velocities/controls zeroed)."""
    data.qpos[:3] = np.asarray(root_pos, dtype=np.float64)
    data.qpos[3:7] = np.asarray(root_quat, dtype=np.float64)
    data.qpos[7:] = np.asarray(dofs_qpos, dtype=np.float64)
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0


# ---------------------------------------------------------------------------
# MPPI swing planner: precompute foot trajectories for all swing phases
# ---------------------------------------------------------------------------


def _push_side(terrain, foot_pos, fwd_x, fwd_y):
    """Check side-wall penetration for one foot and return push vector."""
    return terrain.foot_side_penetration(
        float(foot_pos[0]),
        float(foot_pos[1]),
        float(foot_pos[2]),
        fwd_x,
        fwd_y,
        foot_half_width=0.05,
        foot_half_length=0.10,
    )


def _find_swing_phases(contact_mask):
    """Find (start, end) index pairs for all swing phases.

    A swing phase is a run of consecutive False entries in contact_mask.
    start = first swing frame, end = first re-contact frame.
    Liftoff frame = start - 1 (last contact frame before the swing).
    Landing frame = end (first contact frame after the swing).
    """
    phases = []
    n = len(contact_mask)
    i = 0
    while i < n:
        if not contact_mask[i]:
            j = i
            while j < n and not contact_mask[j]:
                j += 1
            phases.append((i, j))
            i = j
        else:
            i += 1
    return phases


def _find_stance_phases(contact_mask):
    """Find (start, end) index pairs for all stance phases.

    A stance phase is a run of consecutive True entries in contact_mask.
    Returns a list of (start, end) with contact_mask[start:end] all True.
    """
    phases = []
    n = len(contact_mask)
    i = 0
    while i < n:
        if contact_mask[i]:
            j = i
            while j < n and contact_mask[j]:
                j += 1
            phases.append((i, j))
            i = j
        else:
            i += 1
    return phases


def _ankle_to_midfoot(ankle_pos, quat, robot_spec):
    """Convert an ankle position to the mid-foot virtual center position.

    Formula: midfoot_world = ankle_world + R(quat) @ mid_foot_offset, where
    ``robot_spec.mid_foot_offset`` is the average of the toe and heel offsets in
    the ankle-roll-link frame.
    """
    R = quat_to_R(quat)
    return ankle_pos + R @ np.asarray(robot_spec.mid_foot_offset)


def _midfoot_to_ankle(midfoot_pos, quat, robot_spec):
    """Convert a mid-foot virtual center position back to the ankle position.

    Formula: ankle_world = midfoot_world - R(quat) @ mid_foot_offset,
    i.e. the inverse of :func:`_ankle_to_midfoot`.
    """
    R = quat_to_R(quat)
    return midfoot_pos - R @ np.asarray(robot_spec.mid_foot_offset)


def _contact_point_scale_for_pose(
    foot_pos,
    foot_quat,
    terrain,
    robot_spec,
    is_contact,
    support_height_tol=IK_EDGE_HEIGHT_TOL,
    unsupported_scale=IK_UNSUPPORTED_POINT_SCALE,
):
    """Return [ankle, toe, heel] scales for support-aware foot IK."""
    scales = np.ones(3, dtype=np.float64)
    if not is_contact:
        return scales

    R = quat_to_R(foot_quat)
    ankle_surface = float(terrain.height_at(float(foot_pos[0]), float(foot_pos[1])))
    toe_world = foot_pos + R @ np.asarray(robot_spec.toe_offset)
    heel_world = foot_pos + R @ np.asarray(robot_spec.heel_offset)
    toe_surface = float(terrain.height_at(float(toe_world[0]), float(toe_world[1])))
    heel_surface = float(terrain.height_at(float(heel_world[0]), float(heel_world[1])))

    if abs(toe_surface - ankle_surface) > support_height_tol:
        scales[1] = unsupported_scale
    if abs(heel_surface - ankle_surface) > support_height_tol:
        scales[2] = unsupported_scale
    return scales


def _precompute_contact_point_scales(
    foot_positions,
    foot_quats,
    contact_mask,
    terrain,
    robot_spec,
    support_height_tol=IK_EDGE_HEIGHT_TOL,
    unsupported_scale=IK_UNSUPPORTED_POINT_SCALE,
):
    """Precompute per-frame [ankle, toe, heel] IK scales."""
    scales = np.ones((len(contact_mask), 3), dtype=np.float64)
    for i in np.flatnonzero(contact_mask):
        scales[i] = _contact_point_scale_for_pose(
            foot_positions[i],
            foot_quats[i],
            terrain,
            robot_spec,
            is_contact=True,
            support_height_tol=support_height_tol,
            unsupported_scale=unsupported_scale,
        )
    return scales


def _safe_normalize(vec, fallback):
    vec = np.asarray(vec, dtype=np.float64)
    norm = np.linalg.norm(vec)
    if norm < 1e-8:
        return np.asarray(fallback, dtype=np.float64).copy()
    return vec / norm


# ---------------------------------------------------------------------------
# Continuous optimisation modules (no discrete branching)
# ---------------------------------------------------------------------------


def _should_trust_phase_clock(phase_params, fit_quality_threshold=0.72):
    """Only use phase-clock outputs when the periodic fit is clearly reliable."""
    if phase_params is None:
        return False
    gait_type = getattr(phase_params, "gait_type", "")
    fit_quality = float(getattr(phase_params, "fit_quality", 0.0))
    if gait_type not in {"walk", "run", "hop", "gallop", "stand"}:
        return False
    return fit_quality >= fit_quality_threshold


def _support_state(motion, frame_idx, phase_clock_trusted):
    """Return (left_weight, right_weight, is_flight) for one frame."""
    if phase_clock_trusted and motion.per_frame_phase is not None:
        return (
            float(motion.per_frame_phase.support_weight_left[frame_idx]),
            float(motion.per_frame_phase.support_weight_right[frame_idx]),
            bool(motion.per_frame_phase.flight_mask[frame_idx]),
        )
    return (
        float(motion.left_contact[frame_idx]),
        float(motion.right_contact[frame_idx]),
        False,
    )


def _pelvis_reach_z_upper_bound(pelvis_xy, foot_pos, support_weight, max_reach):
    """Maximum pelvis z allowed by one pelvis-ankle pair at the current xy."""
    pelvis_xy = np.asarray(pelvis_xy, dtype=np.float64)
    foot_pos = np.asarray(foot_pos, dtype=np.float64)
    support = float(np.clip(support_weight, 0.0, 1.0))
    xy_dist = float(np.linalg.norm(foot_pos[:2] - pelvis_xy[:2]))
    xy_dist = min(xy_dist, float(max_reach))
    vertical_allow = np.sqrt(max(float(max_reach) ** 2 - xy_dist**2, 0.0))
    return float(foot_pos[2] + vertical_allow + (1.0 - support) * 0.04)


def _continuous_ik_weights(
    base_root_xy_weight,
    base_root_z_weight,
    left_point_scale,
    right_point_scale,
    support_weight_left,
    support_weight_right,
):
    """Derive IK weights by smooth interpolation from continuous support weights.

    No if/else state machine — every output is a smooth function of the two
    scalar support weights in [0,1].

    When total support is high (double stance), weights stay at nominal.
    When total support drops (swing / flight), root xy loosens and foot
    point scales are attenuated proportionally.
    """
    lw = float(support_weight_left)
    rw = float(support_weight_right)
    total = lw + rw  # 0 (flight) … 2 (double stance)

    # confidence ∈ [0, 1]: 0 = flight, 1 = solid double stance
    confidence = float(np.clip(total / 2.0, 0.0, 1.0))

    # root xy: full weight at confidence=1, 55% at confidence=0
    xy_scale = 0.55 + 0.45 * confidence
    # root z: full weight at confidence=1, 65% at confidence=0
    z_scale = 0.65 + 0.35 * confidence

    root_xy_weight = float(base_root_xy_weight) * xy_scale
    root_z_weight = float(base_root_z_weight) * z_scale

    # foot point scales: attenuate toe/heel when that foot's own weight is low
    left_scale = np.asarray(left_point_scale, dtype=np.float64).copy()
    right_scale = np.asarray(right_point_scale, dtype=np.float64).copy()
    # ankle always keeps its scale; toe/heel soften with (1 - foot_support_weight)
    l_attenuation = 0.45 + 0.55 * lw  # 0.45 when fully swinging, 1.0 when stance
    r_attenuation = 0.45 + 0.55 * rw
    left_scale[1:] *= l_attenuation
    right_scale[1:] *= r_attenuation

    return root_xy_weight, root_z_weight, left_scale, right_scale


def _is_locked_double_stance(
    support_weight_left,
    support_weight_right,
    cur_pelvis,
    cur_lfoot,
    cur_rfoot,
    prev_pelvis=None,
    prev_lfoot=None,
    prev_rfoot=None,
    foot_tol=5e-4,
    pelvis_tol=1.5e-2,
):
    """Detect steady double-stance frames where multiseed branch-hopping hurts continuity.

    When both feet are in solid support and the latched foot targets are
    effectively unchanged from the previous frame, the best behavior is to stay
    on the previous IK branch. Re-running the multiseed bank there can flip
    between nearly equivalent local minima and create visible lower-body sway.
    """
    if prev_pelvis is None or prev_lfoot is None or prev_rfoot is None:
        return False
    if float(support_weight_left) < 0.95 or float(support_weight_right) < 0.95:
        return False

    foot_delta_l = float(np.linalg.norm(np.asarray(cur_lfoot) - np.asarray(prev_lfoot)))
    foot_delta_r = float(np.linalg.norm(np.asarray(cur_rfoot) - np.asarray(prev_rfoot)))
    pelvis_delta = float(np.linalg.norm(np.asarray(cur_pelvis) - np.asarray(prev_pelvis)))
    return foot_delta_l <= float(foot_tol) and foot_delta_r <= float(foot_tol) and pelvis_delta <= float(pelvis_tol)


def _project_pelvis_foot_to_reach(pelvis, foot, support_weight, max_reach):
    """Continuously pull one pelvis-foot pair back inside the reach envelope.

    The IK solver becomes poorly conditioned near full leg extension.  A soft
    least-squares penalty is not enough there: it can still hand IK targets
    that require the leg to be longer than the model permits.  This projection
    is intentionally one-dimensional along the pelvis-foot line, so it does not
    introduce contact-state jumps.

    Stance feet move very little; swing feet can absorb more of the correction.
    """
    pelvis = np.asarray(pelvis, dtype=np.float64).copy()
    foot = np.asarray(foot, dtype=np.float64).copy()
    rel = foot - pelvis
    dist = float(np.linalg.norm(rel))
    if dist <= max_reach or dist < 1e-8:
        return pelvis, foot

    direction = rel / dist
    excess = dist - float(max_reach)
    support = float(np.clip(support_weight, 0.0, 1.0))

    # support=1: keep the foot almost fixed and move pelvis toward it.
    # support=0: let swing foot absorb most of the correction.
    foot_fraction = 0.03 + 0.77 * (1.0 - support)
    pelvis_fraction = 1.0 - foot_fraction

    pelvis += pelvis_fraction * excess * direction
    foot -= foot_fraction * excess * direction
    return pelvis, foot


def _smooth_target_shaper(
    cur_pelvis,
    cur_lfoot,
    cur_rfoot,
    support_weight_left,
    support_weight_right,
    max_reach=None,
):
    """Conservative target shaper: only fix genuine reach violations.

    The targets are returned unchanged unless a pelvis-foot distance exceeds the
    reach envelope; pulling pelvis/feet in ordinary double-stance frames would
    change valid startup targets and create bad IK seeds.

    ``max_reach`` must be provided (upstream defaulted to a module-level G1
    constant; the port threads ``robot_spec.max_reach`` from the caller).
    """
    if max_reach is None:
        raise ValueError("_smooth_target_shaper requires max_reach (use robot_spec.max_reach)")

    p0 = np.asarray(cur_pelvis, dtype=np.float64)
    l0 = np.asarray(cur_lfoot, dtype=np.float64)
    r0 = np.asarray(cur_rfoot, dtype=np.float64)
    lw = float(support_weight_left)
    rw = float(support_weight_right)
    reach_limit = max(float(max_reach), 0.1)
    over_l = float(np.linalg.norm(l0 - p0) - reach_limit)
    over_r = float(np.linalg.norm(r0 - p0) - reach_limit)
    if max(over_l, over_r) <= 1e-6:
        return p0, l0, r0

    shaped_p = p0.copy()
    shaped_l = l0.copy()
    shaped_r = r0.copy()
    for _ in range(3):
        shaped_p, shaped_l = _project_pelvis_foot_to_reach(shaped_p, shaped_l, lw, reach_limit)
        shaped_p, shaped_r = _project_pelvis_foot_to_reach(shaped_p, shaped_r, rw, reach_limit)

    return shaped_p, shaped_l, shaped_r


def _build_ik_solver(model, robot_spec, log_config=False):
    """Build the Jacobian IK solver from ``robot_spec.ik_config``.

    All IK solver settings — including root_xy/z_weight — come from the solver
    section of ``robot_spec.ik_config``, merged over ``DEFAULT_IK_CONFIG["solver"]``
    (the single source of defaults; no per-call fallbacks). The jacobian
    backend is the only one ported (curobo/drake branches dropped).
    """
    ik_solver = JacobianIK.from_config(model, robot_spec)
    _solver_cfg = merged_ik_solver_config(robot_spec.ik_config.get("solver", {}))
    root_xy_weight = float(_solver_cfg["root_xy_weight"])
    root_z_weight = float(_solver_cfg["root_z_weight"])
    if log_config:
        en.logger.info(
            f"  IK weights: ankle={ik_solver.point_weights[0]:.2f} "
            f"toe={ik_solver.point_weights[3]:.2f} "
            f"heel={ik_solver.point_weights[6]:.2f} "
            f"penetration={ik_solver.penetration_weight:.1f} "
            f"root_xy={root_xy_weight:.1f} root_z={root_z_weight:.1f}"
        )

    return ik_solver, root_xy_weight, root_z_weight


def _build_ghost_fk_model(model):
    """Ghost FK state: share the (read-only) model, allocate a dedicated MjData.

    Upstream loaded a second MjModel from the XML so ghost FK never disturbs
    the solver state; sharing the model with a fresh MjData is equivalent.
    """
    ghost_data = mujoco.MjData(model)
    return model, ghost_data


@dataclass
class LowerBodyStabilizerState:
    prev_leg_qpos: np.ndarray = None
    prev_root_pos: np.ndarray = None
    prev_left_contact: bool = False
    prev_right_contact: bool = False
    left_lock_pos: np.ndarray = None
    left_lock_quat: np.ndarray = None
    right_lock_pos: np.ndarray = None
    right_lock_quat: np.ndarray = None


def _scale_point_scale(point_scale, factor):
    scale = np.asarray(point_scale, dtype=np.float64).copy()
    return scale * float(max(factor, 0.0))


def _lower_body_stabilizer_enabled(params):
    # Jacobian is the only IK backend in the port, so the upstream backend check is gone.
    return bool(params.ik_lower_body_stabilizer)


def _stabilizer_target_from_base(
    base_target,
    left_contact,
    right_contact,
    stab_state,
    params,
):
    left_pos = base_target.left_pos.copy()
    left_quat = base_target.left_quat.copy()
    right_pos = base_target.right_pos.copy()
    right_quat = base_target.right_quat.copy()

    if left_contact and stab_state.left_lock_pos is not None:
        left_pos = stab_state.left_lock_pos.copy()
        left_quat = stab_state.left_lock_quat.copy()
    if right_contact and stab_state.right_lock_pos is not None:
        right_pos = stab_state.right_lock_pos.copy()
        right_quat = stab_state.right_lock_quat.copy()

    left_scale = _scale_point_scale(
        base_target.left_point_scale,
        params.ik_stabilizer_stance_scale if left_contact else params.ik_stabilizer_swing_scale,
    )
    right_scale = _scale_point_scale(
        base_target.right_point_scale,
        params.ik_stabilizer_stance_scale if right_contact else params.ik_stabilizer_swing_scale,
    )

    root_xy_scale = 1.0
    root_z_scale = 1.0
    if left_contact or right_contact:
        root_xy_scale = float(params.ik_stabilizer_root_xy_scale)
        root_z_scale = float(params.ik_stabilizer_root_z_scale)
        if left_contact and right_contact:
            root_xy_scale *= 0.85
            root_z_scale *= 0.90

    return FrameIKTargets(
        root_pos=base_target.root_pos.copy(),
        root_quat=base_target.root_quat.copy(),
        left_pos=left_pos,
        left_quat=left_quat,
        right_pos=right_pos,
        right_quat=right_quat,
        fixed_upper_body_qpos=base_target.fixed_upper_body_qpos.copy(),
        ref_leg_qpos=base_target.ref_leg_qpos.copy() if base_target.ref_leg_qpos is not None else None,
        left_point_scale=left_scale,
        right_point_scale=right_scale,
        root_xy_weight=float(base_target.root_xy_weight) * root_xy_scale,
        root_z_weight=float(base_target.root_z_weight) * root_z_scale,
    )


def _run_lower_body_stabilizer_frame(
    data,
    ik_solver,
    base_target,
    base_state,
    left_contact,
    right_contact,
    stab_state,
    params,
):
    left_contact = bool(left_contact)
    right_contact = bool(right_contact)

    if not left_contact:
        stab_state.left_lock_pos = None
        stab_state.left_lock_quat = None
    if not right_contact:
        stab_state.right_lock_pos = None
        stab_state.right_lock_quat = None

    target = _stabilizer_target_from_base(base_target, left_contact, right_contact, stab_state, params)

    base_root, base_leg = ik_solver.unpack_state(base_state)
    if not left_contact and not right_contact:
        stab_state.prev_leg_qpos = base_leg.copy()
        stab_state.prev_root_pos = base_root.copy()
        stab_state.prev_left_contact = False
        stab_state.prev_right_contact = False
        return np.asarray(base_state, dtype=np.float64).copy()

    seed_leg = base_leg.copy()
    seed_root = base_root.copy()
    if stab_state.prev_leg_qpos is not None:
        seed_leg = 0.65 * base_leg + 0.35 * stab_state.prev_leg_qpos
    if stab_state.prev_root_pos is not None:
        seed_root = 0.65 * base_root + 0.35 * stab_state.prev_root_pos

    root_continuity_weight = float(params.ik_stabilizer_root_continuity_weight)
    if left_contact and right_contact:
        root_continuity_weight *= 1.5

    ik_result = ik_solver.solve_with_root(
        data,
        target_root_pos=target.root_pos,
        target_root_quat=target.root_quat,
        target_lf_pos=target.left_pos,
        target_lf_quat=target.left_quat,
        target_rf_pos=target.right_pos,
        target_rf_quat=target.right_quat,
        fixed_upper_body_qpos=target.fixed_upper_body_qpos,
        initial_leg_qpos=seed_leg,
        ref_leg_qpos=base_leg,
        root_xy_weight=target.root_xy_weight,
        root_z_weight=target.root_z_weight,
        left_point_scale=target.left_point_scale,
        right_point_scale=target.right_point_scale,
        allow_multiseed=False,
        force_multiseed=False,
        initial_root_pos=seed_root,
        root_continuity_weight=root_continuity_weight,
    )

    data.qpos[:] = ik_result.qpos
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    mujoco.mj_forward(ik_solver.model, data)

    entering_left = left_contact and not stab_state.prev_left_contact
    entering_right = right_contact and not stab_state.prev_right_contact
    if entering_left:
        stab_state.left_lock_pos = data.xpos[ik_solver.left_foot_body_id].copy()
        stab_state.left_lock_quat = data.xquat[ik_solver.left_foot_body_id].copy()
    if entering_right:
        stab_state.right_lock_pos = data.xpos[ik_solver.right_foot_body_id].copy()
        stab_state.right_lock_quat = data.xquat[ik_solver.right_foot_body_id].copy()

    stab_state.prev_leg_qpos = ik_result.leg_qpos.copy()
    stab_state.prev_root_pos = data.qpos[:3].copy()
    stab_state.prev_left_contact = left_contact
    stab_state.prev_right_contact = right_contact
    return ik_solver.pack_state(data.qpos[:3], ik_result.leg_qpos)


def _update_lower_body_stabilizer_memory(
    data,
    ik_solver,
    target,
    state_vec,
    left_contact,
    right_contact,
    stab_state,
):
    left_contact = bool(left_contact)
    right_contact = bool(right_contact)
    _apply_state_to_model(data, ik_solver, target, state_vec)

    if not left_contact:
        stab_state.left_lock_pos = None
        stab_state.left_lock_quat = None
    elif not stab_state.prev_left_contact:
        stab_state.left_lock_pos = data.xpos[ik_solver.left_foot_body_id].copy()
        stab_state.left_lock_quat = data.xquat[ik_solver.left_foot_body_id].copy()

    if not right_contact:
        stab_state.right_lock_pos = None
        stab_state.right_lock_quat = None
    elif not stab_state.prev_right_contact:
        stab_state.right_lock_pos = data.xpos[ik_solver.right_foot_body_id].copy()
        stab_state.right_lock_quat = data.xquat[ik_solver.right_foot_body_id].copy()

    _, leg_q = ik_solver.unpack_state(state_vec)
    stab_state.prev_leg_qpos = leg_q.copy()
    stab_state.prev_root_pos = data.qpos[:3].copy()
    stab_state.prev_left_contact = left_contact
    stab_state.prev_right_contact = right_contact


def _run_lower_body_stabilizer_sequence(
    data,
    ik_solver,
    frame_targets,
    left_contact,
    right_contact,
    base_states,
    params,
    base_frame_errors=None,
    progress_prefix=None,
    progress_every=200,
):
    stabilized = np.asarray(base_states, dtype=np.float64).copy()
    base_frame_errors = None if base_frame_errors is None else np.asarray(base_frame_errors, dtype=np.float64)
    stab_state = LowerBodyStabilizerState()
    n_frames = stabilized.shape[0]
    active_until = -1
    trigger_err = float(params.ik_stabilizer_trigger_mm) / 1000.0
    cooldown = max(int(params.ik_stabilizer_cooldown_frames), 0)

    for t in range(n_frames):
        need_stabilize = bool(left_contact[t] or right_contact[t])
        if base_frame_errors is not None:
            if base_frame_errors[t] > trigger_err:
                active_until = max(active_until, t + cooldown)
            need_stabilize = need_stabilize and (base_frame_errors[t] > trigger_err or t <= active_until)

        if need_stabilize:
            stabilized[t] = _run_lower_body_stabilizer_frame(
                data,
                ik_solver,
                frame_targets[t],
                stabilized[t],
                left_contact[t],
                right_contact[t],
                stab_state,
                params,
            )
        else:
            _update_lower_body_stabilizer_memory(
                data,
                ik_solver,
                frame_targets[t],
                stabilized[t],
                left_contact[t],
                right_contact[t],
                stab_state,
            )
        if progress_prefix and (t % progress_every == 0 or t == n_frames - 1):
            en.logger.debug(f"{progress_prefix} frame {t}/{n_frames}")

    return stabilized


def _build_frame_ik_target_at(
    motion,
    pelvis_positions,
    mppi_lfoot,
    mppi_rfoot,
    mppi_lfoot_quats,
    mppi_rfoot_quats,
    dofs_mapping,
    ik_solver,
    left_point_scales,
    right_point_scales,
    root_xy_weight,
    root_z_weight,
    phase_clock_trusted,
    t,
    prev_shaped_pelvis=None,
    prev_shaped_lfoot=None,
    prev_shaped_rfoot=None,
):
    cur_pelvis = pelvis_positions[t]
    cur_lfoot = mppi_lfoot[t]
    cur_rfoot = mppi_rfoot[t]
    cur_root_quat = motion.links_quat[t, motion.robot_spec.pelvis_index]
    cur_lf_quat = mppi_lfoot_quats[t]
    cur_rf_quat = mppi_rfoot_quats[t]

    lw, rw, _ = _support_state(motion, t, phase_clock_trusted)
    cur_pelvis, cur_lfoot, cur_rfoot = _smooth_target_shaper(
        cur_pelvis,
        cur_lfoot,
        cur_rfoot,
        lw,
        rw,
        max_reach=motion.robot_spec.max_reach,
    )
    locked_double_stance = _is_locked_double_stance(
        lw,
        rw,
        cur_pelvis,
        cur_lfoot,
        cur_rfoot,
        prev_shaped_pelvis,
        prev_shaped_lfoot,
        prev_shaped_rfoot,
    )
    root_xy_w, root_z_w, left_scale, right_scale = _continuous_ik_weights(
        root_xy_weight,
        root_z_weight,
        left_point_scales[t],
        right_point_scales[t],
        lw,
        rw,
    )

    raw_dofs_qpos = reorder_to_mujoco(motion.dofs_pos[t], dofs_mapping)
    ref_leg = ik_solver.extract_leg_qpos(np.concatenate([np.zeros(7, dtype=np.float64), raw_dofs_qpos]))
    target = FrameIKTargets(
        root_pos=cur_pelvis.copy(),
        root_quat=cur_root_quat.copy(),
        left_pos=cur_lfoot.copy(),
        left_quat=cur_lf_quat.copy(),
        right_pos=cur_rfoot.copy(),
        right_quat=cur_rf_quat.copy(),
        fixed_upper_body_qpos=raw_dofs_qpos.copy(),
        ref_leg_qpos=ref_leg.copy(),
        left_point_scale=np.asarray(left_scale, dtype=np.float64).copy(),
        right_point_scale=np.asarray(right_scale, dtype=np.float64).copy(),
        root_xy_weight=float(root_xy_w),
        root_z_weight=float(root_z_w),
    )
    return target, locked_double_stance, lw, rw


def _build_frame_ik_targets(
    motion,
    pelvis_positions,
    mppi_lfoot,
    mppi_rfoot,
    mppi_lfoot_quats,
    mppi_rfoot_quats,
    dofs_mapping,
    ik_solver,
    left_point_scales,
    right_point_scales,
    root_xy_weight,
    root_z_weight,
    phase_clock_trusted,
):
    """Build the exact per-frame IK targets used by the frame IK solve."""
    N = motion.n_frames
    frame_targets = []
    locked_double_stance = np.zeros(N, dtype=bool)

    prev_shaped_pelvis = None
    prev_shaped_lfoot = None
    prev_shaped_rfoot = None
    for t in range(N):
        target, locked_double_stance[t], _, _ = _build_frame_ik_target_at(
            motion,
            pelvis_positions,
            mppi_lfoot,
            mppi_rfoot,
            mppi_lfoot_quats,
            mppi_rfoot_quats,
            dofs_mapping,
            ik_solver,
            left_point_scales,
            right_point_scales,
            root_xy_weight,
            root_z_weight,
            phase_clock_trusted,
            t,
            prev_shaped_pelvis=prev_shaped_pelvis,
            prev_shaped_lfoot=prev_shaped_lfoot,
            prev_shaped_rfoot=prev_shaped_rfoot,
        )
        prev_shaped_pelvis = target.root_pos.copy()
        prev_shaped_lfoot = target.left_pos.copy()
        prev_shaped_rfoot = target.right_pos.copy()
        frame_targets.append(target)

    return frame_targets, locked_double_stance


def _solve_single_frame_ik(
    data,
    ik_solver,
    target,
    prev_leg_qpos=None,
    prev_root_pos=None,
    locked_double_stance=False,
    default_initial_leg_qpos=None,
):
    force_multiseed = bool(locked_double_stance and prev_root_pos is not None)
    root_continuity_w = 120.0 if locked_double_stance else 0.0
    ik_result = ik_solver.solve_with_root(
        data,
        target_root_pos=target.root_pos,
        target_root_quat=target.root_quat,
        target_lf_pos=target.left_pos,
        target_lf_quat=target.left_quat,
        target_rf_pos=target.right_pos,
        target_rf_quat=target.right_quat,
        fixed_upper_body_qpos=target.fixed_upper_body_qpos,
        initial_leg_qpos=prev_leg_qpos if prev_leg_qpos is not None else default_initial_leg_qpos,
        ref_leg_qpos=target.ref_leg_qpos,
        root_xy_weight=target.root_xy_weight,
        root_z_weight=target.root_z_weight,
        left_point_scale=target.left_point_scale,
        right_point_scale=target.right_point_scale,
        force_multiseed=force_multiseed,
        initial_root_pos=prev_root_pos,
        root_continuity_weight=root_continuity_w,
    )
    state_vec = ik_solver.pack_state(ik_result.qpos[:3], ik_result.leg_qpos)
    root_err = 0.0 if ik_result.root_pos_offset is None else float(np.linalg.norm(ik_result.root_pos_offset))
    frame_err = max(root_err, float(ik_result.left_pos_err), float(ik_result.right_pos_err))
    return ik_result, state_vec, frame_err


def _compute_single_frame_init_states(
    data,
    ik_solver,
    frame_targets,
    locked_double_stance,
    progress_prefix=None,
    progress_every=100,
):
    """Run the single-frame IK over every frame; returns the packed states and per-frame errors (m)."""
    N = len(frame_targets)
    # Packed-state width = root_xyz (3) + leg DOFs; resolve from the solver so a
    # non-12-DOF leg set (a non-G1 robot_spec) still packs correctly.
    state_dim = 3 + len(ik_solver.leg_qpos_adr)
    init_states = np.zeros((N, state_dim), dtype=np.float64)
    init_frame_errors = np.zeros(N, dtype=np.float64)
    prev_leg_qpos = None
    prev_root_pos = None

    for t, target in enumerate(frame_targets):
        ik_result, init_states[t], init_frame_errors[t] = _solve_single_frame_ik(
            data,
            ik_solver,
            target,
            prev_leg_qpos=prev_leg_qpos,
            prev_root_pos=prev_root_pos,
            locked_double_stance=bool(locked_double_stance[t]),
            default_initial_leg_qpos=target.ref_leg_qpos,
        )
        prev_leg_qpos = ik_result.leg_qpos.copy()
        prev_root_pos = ik_result.qpos[:3].copy()
        if progress_prefix and (t % progress_every == 0 or t == N - 1):
            en.logger.debug(f"{progress_prefix} frame {t}/{N}")

    return init_states, init_frame_errors


def _maybe_stabilize_live_frame(
    data,
    ik_solver,
    target,
    base_state,
    motion,
    t,
    base_pos_err,
    stab_state,
    active_until,
    params,
):
    trigger_err = float(params.ik_stabilizer_trigger_mm) / 1000.0
    cooldown = max(int(params.ik_stabilizer_cooldown_frames), 0)
    if base_pos_err > trigger_err:
        active_until = max(active_until, t + cooldown)

    left_contact = motion.left_contact[t]
    right_contact = motion.right_contact[t]
    need_lb_stab = bool(left_contact or right_contact) and (base_pos_err > trigger_err or t <= active_until)
    if need_lb_stab:
        stabilized_state = _run_lower_body_stabilizer_frame(
            data,
            ik_solver,
            target,
            base_state,
            left_contact,
            right_contact,
            stab_state,
            params,
        )
        _apply_state_to_model(data, ik_solver, target, stabilized_state)
        final_pos_err = max(
            np.linalg.norm(data.xpos[ik_solver.left_foot_body_id] - target.left_pos),
            np.linalg.norm(data.xpos[ik_solver.right_foot_body_id] - target.right_pos),
        )
        return stabilized_state, final_pos_err, active_until

    _update_lower_body_stabilizer_memory(
        data,
        ik_solver,
        target,
        base_state,
        left_contact,
        right_contact,
        stab_state,
    )
    return base_state, base_pos_err, active_until


def _apply_state_to_model(data, ik_solver, target, state):
    """Write one packed (root_xyz + leg_q) IK state back into MuJoCo qpos and run forward kinematics."""
    root_pos, leg_q = ik_solver.unpack_state(state)
    data.qpos[:3] = root_pos
    data.qpos[3:7] = target.root_quat
    data.qpos[7:] = target.fixed_upper_body_qpos
    data.qpos[ik_solver.leg_qpos_adr] = leg_q
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    mujoco.mj_forward(ik_solver.model, data)
    return root_pos, leg_q


def _solve_ik_state_sequence(
    params,
    data,
    ik_solver,
    frame_targets,
    locked_double_stance,
    left_contact,
    right_contact,
    init_progress_prefix=None,
    init_progress_every=200,
    lb_progress_prefix=None,
    lb_progress_every=200,
):
    base_states, init_frame_errors = _compute_single_frame_init_states(
        data,
        ik_solver,
        frame_targets,
        locked_double_stance,
        progress_prefix=init_progress_prefix,
        progress_every=init_progress_every,
    )

    solved_states = base_states.copy()
    # Frame mode is the only IK mode ported (upstream's window/trajectory IK branch dropped).

    if _lower_body_stabilizer_enabled(params):
        en.logger.info("  Lower-body stabilizer enabled")
        solved_states = _run_lower_body_stabilizer_sequence(
            data,
            ik_solver,
            frame_targets,
            left_contact,
            right_contact,
            solved_states,
            params,
            base_frame_errors=init_frame_errors,
            progress_prefix=lb_progress_prefix,
            progress_every=lb_progress_every,
        )

    return solved_states, init_frame_errors


def _record_solved_sequence(
    data,
    ghost_model,
    ghost_data,
    ik_solver,
    frame_targets,
    solved_states,
    links_mapping,
    dofs_mapping,
    pelvis_positions,
    opt_dofs_pos,
    opt_links_pos,
    opt_links_quat,
    ghost_dofs_pos,
    ghost_links_pos,
    ghost_links_quat,
    motion,
):
    ik_pos_errors = []
    for t, target in enumerate(frame_targets):
        _apply_state_to_model(data, ik_solver, target, solved_states[t])

        lf_bid = links_mapping[motion.robot_spec.left_foot_index]
        rf_bid = links_mapping[motion.robot_spec.right_foot_index]
        perr = np.linalg.norm(data.qpos[:3] - target.root_pos) * 1000.0
        lferr = np.linalg.norm(data.xpos[lf_bid] - target.left_pos) * 1000.0
        rferr = np.linalg.norm(data.xpos[rf_bid] - target.right_pos) * 1000.0
        max_err = max(perr, lferr, rferr)
        ik_pos_errors.append(max_err)
        if max_err > 200.0:
            lc = "S" if motion.left_contact[t] else "W"
            rc = "S" if motion.right_contact[t] else "W"
            en.logger.debug(
                f"  [IK-ERR] f={t:4d} L:{lc} R:{rc}  perr={perr:.0f}mm "
                f"lferr={lferr:.0f}mm rferr={rferr:.0f}mm  "
                f"pelvis_z={target.root_pos[2]:.3f} "
                f"lf_z={target.left_pos[2]:.3f} rf_z={target.right_pos[2]:.3f}"
            )

        opt_dofs_pos[t] = reorder_from_mujoco(data.qpos[7:], dofs_mapping)
        for npz_i, mj_bid in enumerate(links_mapping):
            opt_links_pos[t, npz_i] = data.xpos[mj_bid]
            opt_links_quat[t, npz_i] = data.xquat[mj_bid]

        update_robot_pose(
            ghost_data,
            root_pos=pelvis_positions[t],
            root_quat=target.root_quat,
            dofs_qpos=target.fixed_upper_body_qpos,
        )
        mujoco.mj_forward(ghost_model, ghost_data)
        ghost_dofs_pos[t] = reorder_from_mujoco(ghost_data.qpos[7:], dofs_mapping)
        for npz_i, mj_bid in enumerate(links_mapping):
            ghost_links_pos[t, npz_i] = ghost_data.xpos[mj_bid]
            ghost_links_quat[t, npz_i] = ghost_data.xquat[mj_bid]

    return np.asarray(ik_pos_errors, dtype=np.float64)


def _capsule_clearance(seg0_a, seg0_b, radius0, seg1_a, seg1_b, radius1):
    """Signed capsule clearance: positive = separated, negative = overlapping."""
    seg0_a, seg0_b, seg1_a, seg1_b = (np.asarray(p, dtype=np.float64) for p in (seg0_a, seg0_b, seg1_a, seg1_b))
    cp0, cp1 = segment_closest_points(seg0_a, seg0_b, seg1_a, seg1_b)
    dist_vec = cp0 - cp1
    dist = float(np.linalg.norm(dist_vec))
    return dist - (float(radius0) + float(radius1)), cp0, cp1


def _solve_knee_position(hip_pos, ankle_pos, ref_knee_pos, thigh_len, shin_len):
    """Closed-form 2-link knee reconstruction using a reference bend direction."""
    hip_pos = np.asarray(hip_pos, dtype=np.float64)
    ankle_pos = np.asarray(ankle_pos, dtype=np.float64)
    ref_knee_pos = np.asarray(ref_knee_pos, dtype=np.float64)

    ha = ankle_pos - hip_pos
    d = float(np.linalg.norm(ha))
    if d < 1e-8:
        return ref_knee_pos.copy()

    d_dir = ha / d
    pole = ref_knee_pos - hip_pos
    pole -= d_dir * np.dot(pole, d_dir)
    if np.linalg.norm(pole) < 1e-8:
        pole = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        pole -= d_dir * np.dot(pole, d_dir)
    if np.linalg.norm(pole) < 1e-8:
        pole = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        pole -= d_dir * np.dot(pole, d_dir)
    bend_dir = _safe_normalize(pole, [0.0, 1.0, 0.0])

    d_clamped = float(np.clip(d, abs(thigh_len - shin_len) + 1e-6, thigh_len + shin_len - 1e-6))
    x = (thigh_len**2 - shin_len**2 + d_clamped**2) / (2.0 * d_clamped)
    y_sq = max(thigh_len**2 - x**2, 0.0)
    y = np.sqrt(y_sq)

    knee_base = hip_pos + d_dir * x
    side = 1.0 if np.dot(ref_knee_pos - knee_base, bend_dir) >= 0.0 else -1.0
    return knee_base + side * y * bend_dir


def _foot_capsule_endpoints(ankle_pos, foot_quat, robot_spec):
    R = quat_to_R(foot_quat)
    heel = ankle_pos + R @ np.asarray(robot_spec.heel_offset)
    toe = ankle_pos + R @ np.asarray(robot_spec.toe_offset)
    return heel, toe


def _estimate_leg_capsules(
    motion,
    pelvis_positions,
    left_foot_pos,
    right_foot_pos,
    left_foot_quats,
    right_foot_quats,
):
    """Estimate hip/knee/ankle and foot capsules from planned pelvis + feet."""
    robot_spec = motion.robot_spec
    pelvis_delta = pelvis_positions - motion.links_pos[:, robot_spec.pelvis_index]

    left_hip = motion.links_pos[:, robot_spec.left_hip_index] + pelvis_delta
    right_hip = motion.links_pos[:, robot_spec.right_hip_index] + pelvis_delta
    left_ref_knee = motion.links_pos[:, robot_spec.left_knee_index] + pelvis_delta
    right_ref_knee = motion.links_pos[:, robot_spec.right_knee_index] + pelvis_delta

    left_thigh_len = float(
        np.median(
            np.linalg.norm(
                motion.links_pos[:, robot_spec.left_knee_index] - motion.links_pos[:, robot_spec.left_hip_index], axis=1
            )
        )
    )
    left_shin_len = float(
        np.median(
            np.linalg.norm(
                motion.links_pos[:, robot_spec.left_foot_index] - motion.links_pos[:, robot_spec.left_knee_index],
                axis=1,
            )
        )
    )
    right_thigh_len = float(
        np.median(
            np.linalg.norm(
                motion.links_pos[:, robot_spec.right_knee_index] - motion.links_pos[:, robot_spec.right_hip_index],
                axis=1,
            )
        )
    )
    right_shin_len = float(
        np.median(
            np.linalg.norm(
                motion.links_pos[:, robot_spec.right_foot_index] - motion.links_pos[:, robot_spec.right_knee_index],
                axis=1,
            )
        )
    )

    n = left_foot_pos.shape[0]
    left_knee = np.zeros((n, 3), dtype=np.float64)
    right_knee = np.zeros((n, 3), dtype=np.float64)
    left_heel = np.zeros((n, 3), dtype=np.float64)
    left_toe = np.zeros((n, 3), dtype=np.float64)
    right_heel = np.zeros((n, 3), dtype=np.float64)
    right_toe = np.zeros((n, 3), dtype=np.float64)

    for i in range(n):
        left_knee[i] = _solve_knee_position(
            left_hip[i],
            left_foot_pos[i],
            left_ref_knee[i],
            left_thigh_len,
            left_shin_len,
        )
        right_knee[i] = _solve_knee_position(
            right_hip[i],
            right_foot_pos[i],
            right_ref_knee[i],
            right_thigh_len,
            right_shin_len,
        )
        left_heel[i], left_toe[i] = _foot_capsule_endpoints(left_foot_pos[i], left_foot_quats[i], robot_spec)
        right_heel[i], right_toe[i] = _foot_capsule_endpoints(right_foot_pos[i], right_foot_quats[i], robot_spec)

    return {
        "left_hip": left_hip,
        "right_hip": right_hip,
        "left_knee": left_knee,
        "right_knee": right_knee,
        "left_heel": left_heel,
        "left_toe": left_toe,
        "right_heel": right_heel,
        "right_toe": right_toe,
    }


def _latch_stance_foot_targets(
    foot_pos,
    contact_mask,
    terrain=None,
    foot_nominal_z=0.0,
    ground_tol=0.05,
):
    """Fix each stance run to the first *grounded* stance-frame xyz.

    Phase-derived contact can start a few frames before the foot is actually
    close to the terrain.  Latching to that transition frame can create an
    unreachable IK target.  We therefore anchor only from the first stance frame
    whose target z is close to the terrain contact height; if a run has no such
    frame, it is left unchanged.  Swing frames are untouched.
    """
    out = np.asarray(foot_pos, dtype=np.float64).copy()
    contact = np.asarray(contact_mask, dtype=bool)
    n = contact.shape[0]
    i = 0
    while i < n:
        if not contact[i]:
            i += 1
            continue
        st = i
        while i < n and contact[i]:
            i += 1
        en_ = i

        anchor_idx = st
        if terrain is not None:
            run = out[st:en_]
            floor_z = terrain.height_batch(run[:, 0], run[:, 1]) + float(foot_nominal_z)
            grounded = np.abs(run[:, 2] - floor_z) <= float(ground_tol)
            if not np.any(grounded):
                continue
            anchor_idx = st + int(np.flatnonzero(grounded)[0])

        anchor = out[anchor_idx].copy()
        if terrain is not None:
            anchor[2] = float(terrain.height_at(float(anchor[0]), float(anchor[1]))) + float(foot_nominal_z)
        out[anchor_idx:en_] = anchor[None, :]
    return out


def _collision_frame_count(
    motion,
    pelvis_positions,
    left_foot_pos,
    right_foot_pos,
    left_foot_quats,
    right_foot_quats,
    shin_radius,
    foot_radius,
    collision_margin,
):
    """Count frames with any lower-leg/foot capsule interference."""
    capsules = _estimate_leg_capsules(
        motion,
        pelvis_positions,
        left_foot_pos,
        right_foot_pos,
        left_foot_quats,
        right_foot_quats,
    )
    n = left_foot_pos.shape[0]
    count = 0
    for i in range(n):
        pairs = (
            (
                capsules["left_knee"][i],
                left_foot_pos[i],
                shin_radius,
                capsules["right_knee"][i],
                right_foot_pos[i],
                shin_radius,
            ),
            (
                capsules["left_knee"][i],
                left_foot_pos[i],
                shin_radius,
                capsules["right_heel"][i],
                capsules["right_toe"][i],
                foot_radius,
            ),
            (
                capsules["left_heel"][i],
                capsules["left_toe"][i],
                foot_radius,
                capsules["right_knee"][i],
                right_foot_pos[i],
                shin_radius,
            ),
            (
                capsules["left_heel"][i],
                capsules["left_toe"][i],
                foot_radius,
                capsules["right_heel"][i],
                capsules["right_toe"][i],
                foot_radius,
            ),
        )
        frame_collide = False
        for seg0_a, seg0_b, r0, seg1_a, seg1_b, r1 in pairs:
            clearance, _, _ = _capsule_clearance(seg0_a, seg0_b, r0, seg1_a, seg1_b, r1)
            if clearance < collision_margin:
                frame_collide = True
                break
        count += int(frame_collide)
    return count


def _resolve_lower_leg_foot_collisions(
    motion,
    pelvis_positions,
    left_foot_pos,
    right_foot_pos,
    left_foot_quats,
    right_foot_quats,
    left_contact,
    right_contact,
    terrain,
    foot_nominal_z,
    n_iters=2,
    shin_radius=0.045,
    foot_radius=0.03,
    collision_margin=0.015,
    push_gain=0.9,
    max_push_per_frame=0.04,
    pelvis_side_margin=0.025,
    pelvis_side_gain=0.5,
):
    """Collision-aware post-pass using shin/foot capsules and swing-foot pushes."""
    if n_iters <= 0:
        return left_foot_pos, right_foot_pos

    left_foot_pos = np.asarray(left_foot_pos, dtype=np.float64).copy()
    right_foot_pos = np.asarray(right_foot_pos, dtype=np.float64).copy()
    left_contact = np.asarray(left_contact, dtype=bool)
    right_contact = np.asarray(right_contact, dtype=bool)
    kernel = np.array([0.25, 0.5, 0.25], dtype=np.float64)

    def smooth_delta(delta):
        out = delta.copy()
        for axis in range(2):
            for _ in range(2):
                out[:, axis] = np.convolve(out[:, axis], kernel, mode="same")
        return out

    for _ in range(n_iters):
        capsules = _estimate_leg_capsules(
            motion,
            pelvis_positions,
            left_foot_pos,
            right_foot_pos,
            left_foot_quats,
            right_foot_quats,
        )
        left_delta = np.zeros((left_foot_pos.shape[0], 2), dtype=np.float64)
        right_delta = np.zeros((right_foot_pos.shape[0], 2), dtype=np.float64)

        for i in range(left_foot_pos.shape[0]):
            left_movable = not left_contact[i]
            right_movable = not right_contact[i]
            if not left_movable and not right_movable:
                continue

            lateral = capsules["left_hip"][i] - capsules["right_hip"][i]
            lateral[2] = 0.0
            left_out = _safe_normalize(lateral, [0.0, 1.0, 0.0])[:2]
            right_out = -left_out

            pairs = (
                (
                    capsules["left_knee"][i],
                    left_foot_pos[i],
                    shin_radius,
                    capsules["right_knee"][i],
                    right_foot_pos[i],
                    shin_radius,
                ),
                (
                    capsules["left_knee"][i],
                    left_foot_pos[i],
                    shin_radius,
                    capsules["right_heel"][i],
                    capsules["right_toe"][i],
                    foot_radius,
                ),
                (
                    capsules["left_heel"][i],
                    capsules["left_toe"][i],
                    foot_radius,
                    capsules["right_knee"][i],
                    right_foot_pos[i],
                    shin_radius,
                ),
                (
                    capsules["left_heel"][i],
                    capsules["left_toe"][i],
                    foot_radius,
                    capsules["right_heel"][i],
                    capsules["right_toe"][i],
                    foot_radius,
                ),
            )

            total_push = 0.0
            for seg0_a, seg0_b, r0, seg1_a, seg1_b, r1 in pairs:
                clearance, _, _ = _capsule_clearance(seg0_a, seg0_b, r0, seg1_a, seg1_b, r1)
                if clearance < collision_margin:
                    total_push += collision_margin - clearance

            if total_push > 0.0:
                push = min(push_gain * total_push, max_push_per_frame)
                if left_movable and right_movable:
                    left_delta[i] += 0.5 * push * left_out
                    right_delta[i] += 0.5 * push * right_out
                elif left_movable:
                    left_delta[i] += push * left_out
                elif right_movable:
                    right_delta[i] += push * right_out

            yaw = float(yaw_from_quat(motion.links_quat[i, motion.robot_spec.pelvis_index]))
            left_axis = np.array([-np.sin(yaw), np.cos(yaw)], dtype=np.float64)
            pelvis_xy = pelvis_positions[i, :2]
            side_push_gain = pelvis_side_gain * push_gain

            if left_movable:
                left_lat = float(np.dot(left_foot_pos[i, :2] - pelvis_xy, left_axis))
                if left_lat < pelvis_side_margin:
                    left_delta[i] += (
                        min(
                            side_push_gain * (pelvis_side_margin - left_lat),
                            max_push_per_frame,
                        )
                        * left_axis
                    )

            if right_movable:
                right_lat = float(np.dot(right_foot_pos[i, :2] - pelvis_xy, left_axis))
                if right_lat > -pelvis_side_margin:
                    right_delta[i] -= (
                        min(
                            side_push_gain * (pelvis_side_margin + right_lat),
                            max_push_per_frame,
                        )
                        * left_axis
                    )

        left_delta = smooth_delta(left_delta)
        right_delta = smooth_delta(right_delta)

        left_mag = np.linalg.norm(left_delta, axis=1)
        right_mag = np.linalg.norm(right_delta, axis=1)
        left_scale = np.minimum(1.0, max_push_per_frame / np.maximum(left_mag, 1e-8))
        right_scale = np.minimum(1.0, max_push_per_frame / np.maximum(right_mag, 1e-8))
        left_delta *= left_scale[:, None]
        right_delta *= right_scale[:, None]

        # Apply deltas to swing frames only (stance frames remain locked)
        left_swing = ~left_contact
        right_swing = ~right_contact
        left_foot_pos[left_swing, :2] += left_delta[left_swing]
        right_foot_pos[right_swing, :2] += right_delta[right_swing]

        # Floor clamp (only for swing frames)
        left_floor = terrain.height_batch(left_foot_pos[:, 0], left_foot_pos[:, 1]) + foot_nominal_z
        right_floor = terrain.height_batch(right_foot_pos[:, 0], right_foot_pos[:, 1]) + foot_nominal_z
        left_foot_pos[left_swing, 2] = np.maximum(left_foot_pos[left_swing, 2], left_floor[left_swing])
        right_foot_pos[right_swing, 2] = np.maximum(right_foot_pos[right_swing, 2], right_floor[right_swing])

    return left_foot_pos, right_foot_pos


def precompute_mppi_foot(
    precomputed_ref,
    precomputed_quats,
    contact_mask,
    foot_nominal_z,
    terrain,
    mppi_params,
    fps,
    robot_spec,
):
    """Run MPPI trajectory optimization over all swing phases of one foot.

    Core idea: plan on the mid-foot virtual center (midpoint of toe and heel)
    rather than the ankle origin. The planning point sits closer to the ground
    contact surface, reducing toe/heel penetration at step edges. After
    optimization the mid-foot result is transformed back to ankle coordinates
    for IK.

    A swing between stances at the same height (< 1.5 cm apart) whose reference
    clears the terrain is kept as it is.

    Parameters
    ----------
    precomputed_ref : np.ndarray
        (N, 3) ankle reference positions from TerrainReference.
    precomputed_quats : np.ndarray
        (N, 4) ankle reference quaternions (wxyz) from TerrainReference.
    contact_mask : np.ndarray
        (N,) boolean contact mask (True = stance, False = swing).
    foot_nominal_z : float
        Nominal ankle z height on flat ground (m).
    terrain : TerrainSource
        Terrain height source.
    mppi_params : MppiFootParams
        Optimizer parameters.
    fps : float
        Motion frame rate (Hz).
    robot_spec : TCRSRobotOptions
        Robot geometry; provides the mid-foot offset for the ankle<->midfoot conversion.

    Returns
    -------
    optimized_pos : np.ndarray
        (N, 3) ankle positions (stance frames unchanged, optimized swing frames overwritten).
    optimized_quats : np.ndarray
        (N, 4) quaternions: yaw-only during stance (sole stays level), the reference ones
        during swing, slerp-blended over up to 10 frames at each stance boundary.
    """
    opt = MppiFootOptimizer(mppi_params)
    n = len(contact_mask)
    out_pos = precomputed_ref.copy()
    out_quats = precomputed_quats.copy()

    # ===== Step 1: lock stance-phase quaternions to pure yaw =====
    # On discrete stair-like terrain, the stance foot stays level by default (yaw only).
    # This avoids bogus pitch/roll from finite-difference normal estimates at step edges.
    stance_phases = _find_stance_phases(contact_mask)
    for st, en_ in stance_phases:
        yaw = float(yaw_from_quat(precomputed_quats[st]))
        locked_quat = xyz_to_quat(np.array([0.0, 0.0, yaw]))  # yaw-only: sole stays level
        for i in range(st, en_):
            out_quats[i] = locked_quat

    # ===== Step 2: run MPPI optimization for each swing phase =====
    phases = _find_swing_phases(contact_mask)
    _n_skipped = 0

    if len(phases) > 0:
        en.logger.info(f"    MPPI optimizing {len(phases)} swing phases...")

    for phase_idx, (swing_start, swing_end) in enumerate(phases):
        if phase_idx == 0 or (phase_idx + 1) % 5 == 0 or phase_idx == len(phases) - 1:
            en.logger.debug(f"      phase {phase_idx + 1}/{len(phases)} frames=[{swing_start},{swing_end})")
        liftoff_idx = swing_start - 1  # liftoff frame = the frame before the swing starts
        landing_idx = swing_end  # landing frame = the first frame after the swing ends

        # Bounds check: skip optimization if the liftoff/landing frame is out of range.
        # Swing frames keep the original mocap quaternions (precomputed_quats), not overwritten.
        if liftoff_idx < 0 or landing_idx >= n:
            continue

        # ===== Conditional skip: preserve original motion when possible =====
        # Check height difference between consecutive stances
        liftoff_z = float(out_pos[liftoff_idx, 2])
        landing_z = float(out_pos[landing_idx, 2])
        height_diff = abs(landing_z - liftoff_z)

        if height_diff < 0.015:  # Same level (< 1.5cm height change)
            # Check: does the original swing trajectory collide with terrain?
            _has_collision = False
            _clearance_margin = mppi_params.h_clearance if hasattr(mppi_params, "h_clearance") else 0.02
            for _k in range(swing_start, swing_end):
                _fx, _fy, _fz = precomputed_ref[_k]
                _tz = float(terrain.height_at(_fx, _fy))
                if _fz < _tz + foot_nominal_z + _clearance_margin:
                    _has_collision = True
                    break

            if not _has_collision:
                # SKIP: no height change, no collision → keep original trajectory
                _n_skipped += 1
                continue

        n_swing = swing_end - swing_start  # number of frames in the swing phase

        # --- Swing quaternions: keep the original mocap quaternions, tracking the
        # reference motion's pitch/roll ---
        # The stance phase was locked to pure yaw (level sole) in step 1;
        # the swing phase keeps the original mocap values from precomputed_quats.
        # Slerp over blend_frames frames at the start and end of the swing to avoid
        # quaternion jumps at the stance (pure yaw) -> swing (mocap) boundary.
        blend_frames = min(10, n_swing // 2)  # number of transition frames, at most half a swing
        if blend_frames > 0:
            q_liftoff = out_quats[liftoff_idx]  # pure yaw (locked during stance)
            q_landing = out_quats[landing_idx]  # pure yaw (locked during stance)
            # Swing start: transition from liftoff pure yaw to mocap
            for b in range(blend_frames):
                alpha = (b + 1) / (blend_frames + 1)  # 0 -> 1 excluding the endpoints
                idx_b = swing_start + b
                q_mocap = precomputed_quats[idx_b]
                q_blend = slerp_quaternion(q_liftoff, q_mocap, np.array([alpha]))[0]
                out_quats[idx_b] = q_blend
            # Swing end: transition from mocap to landing pure yaw
            for b in range(blend_frames):
                alpha = (b + 1) / (blend_frames + 1)
                idx_b = swing_end - blend_frames + b
                q_mocap = precomputed_quats[idx_b]
                q_blend = slerp_quaternion(q_mocap, q_landing, np.array([alpha]))[0]
                out_quats[idx_b] = q_blend

        # --- Convert ankle coordinates to mid-foot coordinates for MPPI planning ---
        liftoff_midfoot = _ankle_to_midfoot(out_pos[liftoff_idx], out_quats[liftoff_idx], robot_spec)
        landing_midfoot = _ankle_to_midfoot(out_pos[landing_idx], out_quats[landing_idx], robot_spec)

        # Special case: with fewer than 2 swing frames MPPI cannot optimize
        # (at least 2 frames are needed to form a valid trajectory) — use linear
        # interpolation instead.
        if n_swing < 2:
            t_frac = np.linspace(0, 1, n_swing + 2)[1:-1]  # drop the two endpoint samples
            for k, frac in enumerate(t_frac):
                midfoot_interp = (1 - frac) * liftoff_midfoot + frac * landing_midfoot
                out_pos[swing_start + k] = _midfoot_to_ankle(midfoot_interp, out_quats[swing_start + k], robot_spec)
            continue

        swing_duration = n_swing / fps  # swing duration (seconds)
        ref_indices = np.arange(swing_start, swing_end)  # indices of the swing frames

        # --- Build the MPPI tracking target: convert reference ankle positions to
        # mid-foot coordinates ---
        ref_traj_midfoot = np.zeros((n_swing, 3), dtype=np.float64)
        for k, idx in enumerate(ref_indices):
            ref_traj_midfoot[k] = _ankle_to_midfoot(precomputed_ref[idx], out_quats[idx], robot_spec)
        ref_times = np.linspace(0, swing_duration, n_swing)  # time sample points

        # optimize_swing samples xyz jointly: tracking + smoothness costs, plus
        # terrain-penetration and height-ceiling penalties on z.
        try:
            traj_x, traj_y, traj_z = opt.optimize_swing(
                liftoff_midfoot,
                landing_midfoot,
                swing_duration,
                ref_traj_midfoot,
                ref_times,
                terrain,
            )
        except Exception as e:
            en.logger.warning(f"  [MPPI] Warning: swing [{swing_start}:{swing_end}] failed: {e}")
            continue  # keep the original TerrainReference trajectory when optimization fails

        # --- Evaluate the optimized mid-foot trajectory and convert back to ankle
        # coordinates ---
        t_eval = np.linspace(0, swing_duration, n_swing)
        midfoot_x = traj_x.eval(t_eval)
        midfoot_y = traj_y.eval(t_eval)
        midfoot_z = traj_z.eval(t_eval)

        for k, idx in enumerate(ref_indices):
            midfoot_pos = np.array([midfoot_x[k], midfoot_y[k], midfoot_z[k]])
            out_pos[idx] = _midfoot_to_ankle(midfoot_pos, out_quats[idx], robot_spec)  # mid-foot -> ankle

        # --- Blend back toward original ref if landing at same height ---
        # When MPPI was needed (collision on flat ground), blend the end of
        # the swing back toward the precomputed_ref to restore original motion shape.
        if height_diff < 0.015 and n_swing > 6:
            _blend_start = swing_start + int(0.7 * n_swing)
            for _k in range(_blend_start, swing_end):
                _alpha = (_k - _blend_start) / max(swing_end - _blend_start, 1)
                out_pos[_k] = (1.0 - _alpha) * out_pos[_k] + _alpha * precomputed_ref[_k]

    if _n_skipped > 0:
        en.logger.info(f"    Skipped {_n_skipped}/{len(phases)} swing phases (no height change, no collision)")

    return out_pos, out_quats


# ---------------------------------------------------------------------------
# Export filtering
# ---------------------------------------------------------------------------


def apply_zero_phase_lowpass(traj, fps, cutoff_hz=10.0, order=4):
    """Zero-phase Butterworth lowpass on a (N,D) or (N,) array.

    Uses scipy.signal.filtfilt for zero phase shift.
    Gracefully handles short sequences: if the signal is too short for
    the requested filter order, the order is reduced until filtfilt can
    run, or the original is returned unfiltered.
    """
    from scipy.signal import butter, filtfilt

    if cutoff_hz <= 0:
        return traj.copy()
    nyquist = fps / 2.0
    if cutoff_hz >= nyquist:
        return traj.copy()

    traj = np.asarray(traj, dtype=np.float64)
    N = traj.shape[0]

    # filtfilt requires padlen <= N - 1; default padlen = 3 * max(len(a), len(b))
    # For butter order k: len(b) = len(a) = k + 1, so padlen = 3 * (k + 1).
    # If N is too short, reduce order until it fits, or skip.
    eff_order = int(order)
    while eff_order >= 1:
        padlen = 3 * (eff_order + 1)
        if padlen < N:
            break
        eff_order -= 1
    if eff_order < 1:
        return traj.copy()  # too short to filter at all

    b, a = butter(eff_order, cutoff_hz / nyquist, btype="low")

    def _filt1d(x):
        return filtfilt(b, a, x)

    if traj.ndim == 1:
        return _filt1d(traj)
    out = np.empty_like(traj)
    if traj.ndim == 2:
        for j in range(traj.shape[1]):
            out[:, j] = _filt1d(traj[:, j])
    elif traj.ndim == 3:
        for b_idx in range(traj.shape[1]):
            for d in range(traj.shape[2]):
                out[:, b_idx, d] = _filt1d(traj[:, b_idx, d])
    else:
        out = traj.copy()
    return out


def _smooth_quaternions_logmap(quats, fps, cutoff_hz, order):
    """Smooth quaternion sequences via log-map + zero-phase lowpass + exp-map.

    For each body:
      1. Ensure quaternion sign continuity (flip to shortest arc).
      2. Compute incremental rotations relative to the first frame.
      3. Map to axis-angle (log map) — 3D signal per frame.
      4. Apply zero-phase lowpass on the 3D log signal.
      5. Map back to quaternions (exp map).

    This avoids component-wise filtering artefacts (non-unit quaternions,
    wrong interpolation paths).
    """
    from scipy.spatial.transform import Rotation

    quats = np.asarray(quats, dtype=np.float64).copy()  # (N, n_bodies, 4) wxyz
    n_bodies = quats.shape[1]

    for bi in range(n_bodies):
        q = quats[:, bi]  # (N, 4) wxyz

        # Sign continuity: flip if dot with previous < 0
        for i in range(1, len(q)):
            if np.dot(q[i], q[i - 1]) < 0:
                q[i] *= -1

        # Convert wxyz → xyzw for scipy
        q_xyzw = q[:, [1, 2, 3, 0]]
        R0_inv = Rotation.from_quat(q_xyzw[0]).inv()
        # Incremental rotations relative to frame 0
        R_all = Rotation.from_quat(q_xyzw)
        R_inc = R0_inv * R_all  # relative rotation
        rotvec = R_inc.as_rotvec()  # (N, 3) axis-angle

        # Lowpass the 3D rotation vector
        rotvec_filt = apply_zero_phase_lowpass(rotvec, fps, cutoff_hz, order)

        # Map back: R0 * exp(filtered rotvec)
        R_filt = Rotation.from_rotvec(rotvec_filt)
        R_out = Rotation.from_quat(q_xyzw[0]) * R_filt
        q_out_xyzw = R_out.as_quat()  # (N, 4) xyzw
        quats[:, bi] = q_out_xyzw[:, [3, 0, 1, 2]]  # back to wxyz

    return quats


def filter_motion_for_export(
    dofs_pos,
    links_pos,
    links_quat,
    fps,
    cutoff_hz=10.0,
    order=4,
):
    """Apply zero-phase lowpass to all exported channels consistently.

    - dofs_pos, links_pos: Butterworth filtfilt.
    - links_quat: log-map → filtfilt → exp-map (proper SO(3) smoothing).

    (Upstream also recomputed velocities from the filtered channels; the port
    returns positions/quaternions only.)
    """
    if cutoff_hz <= 0:
        return dofs_pos.copy(), links_pos.copy(), links_quat.copy()

    filt_dofs_pos = apply_zero_phase_lowpass(dofs_pos, fps, cutoff_hz, order)
    filt_links_pos = apply_zero_phase_lowpass(links_pos, fps, cutoff_hz, order)
    filt_links_quat = _smooth_quaternions_logmap(links_quat, fps, cutoff_hz, order)
    return filt_dofs_pos, filt_links_pos, filt_links_quat


def filter_lower_body_motion_for_export(
    model,
    robot_spec,
    dofs_mapping,
    links_mapping,
    dofs_pos,
    links_pos,
    links_quat,
    fps,
    cutoff_hz=5.0,
    order=4,
):
    """Smooth root position + waist + leg joints, then rebuild full-body poses by FK.

    This preserves shoulder/elbow/wrist motion while making the support chain
    smoother and keeping exported body poses consistent with the exported
    joint commands.

    Root orientation is intentionally left unchanged. Smoothing pelvis
    quaternions over long clips can create catastrophic whole-body heading
    artefacts even when IK position error stays small, because the batch
    error metric does not constrain root orientation.

    (Upstream loaded its own MjModel from the XML and recomputed velocities;
    the port reuses the caller's model/mappings and returns positions only.)
    """
    dofs_pos = np.asarray(dofs_pos, dtype=np.float64)
    links_pos = np.asarray(links_pos, dtype=np.float64)
    links_quat = np.asarray(links_quat, dtype=np.float64)
    if cutoff_hz <= 0:
        return dofs_pos.copy(), links_pos.copy(), links_quat.copy()

    smooth_names = (
        list(robot_spec.left_leg_dofs_name) + list(robot_spec.right_leg_dofs_name) + list(robot_spec.waist_dofs_name)
    )
    smooth_policy_idx = np.array(
        [robot_spec.dofs_name.index(name) for name in smooth_names],
        dtype=np.int64,
    )

    filt_dofs_pos = dofs_pos.copy()
    filt_dofs_pos[:, smooth_policy_idx] = apply_zero_phase_lowpass(
        dofs_pos[:, smooth_policy_idx],
        fps,
        cutoff_hz,
        order,
    )

    filt_root_pos = apply_zero_phase_lowpass(
        links_pos[:, robot_spec.pelvis_index, :],
        fps,
        cutoff_hz,
        order,
    )
    filt_root_quat = links_quat[:, robot_spec.pelvis_index].copy()

    data = mujoco.MjData(model)

    filt_links_pos = np.zeros_like(links_pos, dtype=np.float64)
    filt_links_quat = np.zeros_like(links_quat, dtype=np.float64)
    for t in range(dofs_pos.shape[0]):
        dofs_qpos = reorder_to_mujoco(filt_dofs_pos[t], dofs_mapping)
        update_robot_pose(
            data,
            root_pos=filt_root_pos[t],
            root_quat=filt_root_quat[t],
            dofs_qpos=dofs_qpos,
        )
        mujoco.mj_forward(model, data)
        for npz_i, mj_bid in enumerate(links_mapping):
            filt_links_pos[t, npz_i] = data.xpos[mj_bid]
            filt_links_quat[t, npz_i] = data.xquat[mj_bid]

    return filt_dofs_pos, filt_links_pos, filt_links_quat


# ---------------------------------------------------------------------------
# SE(2) clip placement
# ---------------------------------------------------------------------------


def _apply_se2_transform(motion, dx, dy, dyaw, pelvis_index):
    """Apply an SE(2) transform to a MotionClip in place.

    1) Yaw rotation about the frame-0 pelvis xy, 2) xy translation.
    """
    # 1) Yaw rotation around initial pelvis xy
    if abs(dyaw) > 1e-8:
        cos_y, sin_y = np.cos(dyaw), np.sin(dyaw)
        R2 = np.array([[cos_y, -sin_y], [sin_y, cos_y]], dtype=np.float64)
        q_yaw = np.array([np.cos(dyaw / 2), 0.0, 0.0, np.sin(dyaw / 2)], dtype=np.float64)
        pivot_xy = motion.links_pos[0, pelvis_index, :2].copy()
        xy_all = motion.links_pos[:, :, :2] - pivot_xy[None, None, :]
        motion.links_pos[:, :, :2] = np.einsum("ij,...j->...i", R2, xy_all) + pivot_xy[None, None, :]
        # q_yaw ⊗ q for every body quat; transform_quat_by_quat(v, u) returns u ⊗ v
        # and does not broadcast a (4,) u against (T, K, 4), hence the broadcast_to.
        q = motion.links_quat
        motion.links_quat = transform_quat_by_quat(q, np.broadcast_to(q_yaw, q.shape)).astype(q.dtype)
        if hasattr(motion, "links_lin_vel"):
            n_bodies = motion.links_pos.shape[1]
            for bi in range(n_bodies):
                xy_v = motion.links_lin_vel[:, bi, :2].copy()
                motion.links_lin_vel[:, bi, :2] = (R2 @ xy_v.T).T

    # 2) XY translation
    if abs(dx) > 1e-8 or abs(dy) > 1e-8:
        motion.links_pos[:, :, 0] += dx
        motion.links_pos[:, :, 1] += dy


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@dataclass
class ClipArrays:
    """Input motion clip as plain arrays (orders defined by the adapter's TCRSRobotOptions)."""

    fps: float
    dofs_pos: np.ndarray
    """(T, J) joint positions in ``robot_spec.dofs_name`` order."""
    links_pos_w: np.ndarray
    """(T, K, 3) world link positions in ``robot_spec.links_name`` order."""
    links_quat_w: np.ndarray
    """(T, K, 4) world link quaternions, wxyz."""
    links_lin_vel_w: np.ndarray | None = None
    """(T, K, 3) world link linear velocities; finite-differenced when omitted."""


@dataclass
class MotionSet:
    """One output motion variant (positions only; velocities are derived downstream)."""

    dofs_pos: np.ndarray
    """(T, J) joint positions in ``robot_spec.dofs_name`` order."""
    links_pos_w: np.ndarray
    """(T, K, 3) world link positions in ``robot_spec.links_name`` order."""
    links_quat_w: np.ndarray
    """(T, K, 4) world link quaternions, wxyz."""


@dataclass
class AdaptResult:
    """Output of :meth:`TCRSMotionAdapter.adapt` for one clip placement."""

    raw: MotionSet
    """SE(2)-placed original motion (no terrain adaptation)."""
    ghost: MotionSet
    """Raw joints + terrain-adjusted pelvis z, rebuilt by FK only (no IK)."""
    optimized: MotionSet
    """IK-solved motion on terrain; root position, waist and legs zero-phase lowpassed (``lowpass_cutoff_hz``)."""
    left_contact: np.ndarray
    """(T,) bool detected left-foot contacts."""
    right_contact: np.ndarray
    """(T,) bool detected right-foot contacts."""
    ik_pos_errors: np.ndarray
    """(T,) per-frame max IK position error in millimeters."""
    ik_bad_frame_count: int
    """Number of frames whose IK error exceeds ``params.reject_ik_mm``."""


class TCRSMotionAdapter:
    """Terrain-adaptive motion optimizer: adapts flat-ground clips to a terrain.

    Loads the MuJoCo model once and reuses the IK solver across :meth:`adapt`
    calls; each call allocates fresh ``MjData`` so no state leaks between clips.

    Parameters
    ----------
    model_path : str
        Path to the robot's MuJoCo MJCF (floating base + 1-DOF joints).
    robot_spec : TCRSRobotOptions | None
        Robot geometry/naming; defaults to
        :class:`~eden.options.extensions.motion_optimization.TCRSRobotOptions` (the G1 spec).
    params : TCRSPlannerOptions | None
        Pipeline knobs; defaults to
        :class:`~eden.options.extensions.motion_optimization.TCRSPlannerOptions`.
    """

    def __init__(
        self,
        model_path: str,
        robot_spec: TCRSRobotOptions | None = None,
        params: TCRSPlannerOptions | None = None,
    ):
        self.robot_spec = robot_spec if robot_spec is not None else TCRSRobotOptions()
        self.params = params if params is not None else TCRSPlannerOptions()
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        if self.model.njnt == 0 or self.model.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError(
                f"The MuJoCo model at '{model_path}' must have a free joint first (floating base); "
                "the pipeline addresses qpos as [root(7), joints...]."
            )
        n_model_joints = self.model.nq - 7
        if len(self.robot_spec.dofs_name) != n_model_joints:
            raise ValueError(
                f"TCRSRobotOptions.dofs_name has {len(self.robot_spec.dofs_name)} joints but the MuJoCo model at "
                f"'{model_path}' has nq-7={n_model_joints} joint DOFs; the clip joint order must cover exactly "
                "the model's non-root joints."
            )
        self.dofs_mapping = build_dofs_mapping(self.model, self.robot_spec.dofs_name)
        self.links_mapping = build_links_mapping(self.model, self.robot_spec.links_name)
        self.ik_solver, self.root_xy_weight, self.root_z_weight = _build_ik_solver(
            self.model, self.robot_spec, log_config=False
        )

    def adapt(
        self,
        clip: ClipArrays,
        terrain: TerrainSource,
        dx: float = 0.0,
        dy: float = 0.0,
        dyaw: float = 0.0,
    ) -> AdaptResult:
        """Run the full pipeline for one clip with a given SE(2) placement transform.

        Parameters
        ----------
        clip : ClipArrays
            Input motion arrays.
        terrain : TerrainSource
            Terrain height source.
        dx : float
            X translation applied after the yaw rotation (m).
        dy : float
            Y translation applied after the yaw rotation (m).
        dyaw : float
            Yaw rotation about the frame-0 pelvis xy (rad).

        Returns
        -------
        AdaptResult
            Raw / ghost / optimized motion sets and IK quality metrics.
        """
        robot_spec = self.robot_spec
        params = self.params

        # ===== Load motion =====
        motion = MotionClip(
            robot_spec=robot_spec,
            fps=clip.fps,
            dofs_pos=clip.dofs_pos,
            links_pos_w=clip.links_pos_w,
            links_quat_w=clip.links_quat_w,
            links_lin_vel_w=clip.links_lin_vel_w,
            contact_z_threshold=params.contact_z_threshold,
            contact_speed_threshold=params.contact_speed_threshold,
        )
        phase_clock_trusted = _should_trust_phase_clock(motion.phase_params)
        if motion.phase_params is not None:
            pp = motion.phase_params
            trust_str = "trusted" if phase_clock_trusted else "disabled"
            en.logger.info(f"  Phase clock: type={pp.gait_type} fit={pp.fit_quality:.2f} [{trust_str}]")

        # ===== Apply SE(2) transform (dx, dy, dyaw) =====
        _apply_se2_transform(motion, dx, dy, dyaw, robot_spec.pelvis_index)

        N = motion.n_frames
        fps = float(motion.fps)

        # ===== Terrain reference =====
        ref_builder = TerrainReference(
            terrain=terrain,
            motion=motion,
            lookahead=params.lookahead,
            smoothing_alpha=params.smoothing_alpha,
            toe_offset_x=robot_spec.toe_probe_offset_x,
            heel_offset_x=params.heel_offset_x,
            mid_offset_x=params.mid_offset_x,
            swing_floor_margin=params.swing_floor_margin,
        )

        # ===== Swing trajectory optimization =====
        midfoot_contact_z = motion.foot_nominal_z + float(np.asarray(robot_spec.mid_foot_offset)[2])
        opt_params = MppiFootParams(
            n_knots=params.mppi_n_knots,
            n_samples=params.mppi_n_samples,
            n_iterations=params.mppi_n_iterations,
            temperature=params.mppi_temperature,
            w_track=params.w_track,
            w_terrain=params.w_terrain,
            h_clearance=params.h_clearance,
            h_contact_z=midfoot_contact_z,
            seed=params.seed,
        )

        pelvis_quats = motion.links_quat[:, robot_spec.pelvis_index]

        # ===== Enforce lateral separation: prevent L/R foot xy crossover =====
        _min_lateral_sep = 0.02
        _pelvis_xy = motion.links_pos[:, robot_spec.pelvis_index, :2]
        for i in range(len(pelvis_quats)):
            yaw = float(yaw_from_quat(pelvis_quats[i]))
            lat = np.array([-np.sin(yaw), np.cos(yaw)])
            lf_lat = np.dot(ref_builder.left_precomputed[i, :2] - _pelvis_xy[i], lat)
            rf_lat = np.dot(ref_builder.right_precomputed[i, :2] - _pelvis_xy[i], lat)
            if lf_lat - rf_lat < _min_lateral_sep:
                deficit = _min_lateral_sep - (lf_lat - rf_lat)
                ref_builder.left_precomputed[i, :2] += 0.5 * deficit * lat
                ref_builder.right_precomputed[i, :2] -= 0.5 * deficit * lat

        mppi_lfoot, mppi_lfoot_quats = precompute_mppi_foot(
            ref_builder.left_precomputed,
            ref_builder.left_precomputed_quats,
            motion.left_contact,
            motion.foot_nominal_z,
            terrain,
            opt_params,
            motion.fps,
            robot_spec=robot_spec,
        )
        mppi_rfoot, mppi_rfoot_quats = precompute_mppi_foot(
            ref_builder.right_precomputed,
            ref_builder.right_precomputed_quats,
            motion.right_contact,
            motion.foot_nominal_z,
            terrain,
            opt_params,
            motion.fps,
            robot_spec=robot_spec,
        )

        # Post-MPPI lateral clamp: MPPI may re-introduce crossover via terrain avoidance,
        # so the same 2 cm separation as before MPPI is applied to its output.
        _min_lateral_sep = 0.02
        _pelvis_xy = motion.links_pos[:, robot_spec.pelvis_index, :2]
        _n_post_fixed = 0
        for i in range(N):
            yaw = float(yaw_from_quat(pelvis_quats[i]))
            lat = np.array([-np.sin(yaw), np.cos(yaw)])
            lf_lat = np.dot(mppi_lfoot[i, :2] - _pelvis_xy[i], lat)
            rf_lat = np.dot(mppi_rfoot[i, :2] - _pelvis_xy[i], lat)
            if lf_lat - rf_lat < _min_lateral_sep:
                deficit = _min_lateral_sep - (lf_lat - rf_lat)
                mppi_lfoot[i, :2] += 0.5 * deficit * lat
                mppi_rfoot[i, :2] -= 0.5 * deficit * lat
                _n_post_fixed += 1
        if _n_post_fixed > 0:
            en.logger.info(f"  Post-MPPI lateral clamp: {_n_post_fixed}/{N} frames")

        # ===== Post-MPPI side-wall collision fix =====
        # Probe horizontally (terrain.foot_side_penetration) and push feet away from terrain side walls.
        if hasattr(terrain, "foot_side_penetration"):
            _n_side_fixed = 0
            for i in range(N):
                yaw = float(yaw_from_quat(pelvis_quats[i]))
                fx, fy = np.cos(yaw), np.sin(yaw)
                # Left foot
                px_l, py_l = _push_side(terrain, mppi_lfoot[i], fx, fy)
                if abs(px_l) > 1e-4 or abs(py_l) > 1e-4:
                    mppi_lfoot[i, 0] += px_l
                    mppi_lfoot[i, 1] += py_l
                    _n_side_fixed += 1
                # Right foot
                px_r, py_r = _push_side(terrain, mppi_rfoot[i], fx, fy)
                if abs(px_r) > 1e-4 or abs(py_r) > 1e-4:
                    mppi_rfoot[i, 0] += px_r
                    mppi_rfoot[i, 1] += py_r
                    _n_side_fixed += 1
            if _n_side_fixed > 0:
                en.logger.info(f"  Post-MPPI side-wall fix: {_n_side_fixed} foot-frames pushed")

        # ===== Pelvis z: construct target externally, filter only smooths =====
        #   support_z   — terrain-driven weighted foot height
        #   z_anchor    — support_z + nominal clearance
        #   z_style     — zero-mean relative pelvis variation from raw motion
        #   z_target    — z_anchor + z_style, clamped by reachability
        root_z_filter = SupportAwareRootZFilter(
            alpha_up=0.5,
            alpha_down=0.35,
            max_delta=0.035,
        )
        raw_pelvis = motion.links_pos[:, robot_spec.pelvis_index].copy()
        nominal_root_clearance = float(motion.pelvis_height_above_foot)
        raw_pelvis_z0 = float(raw_pelvis[0, 2])
        raw_style_z = raw_pelvis[:, 2] - raw_pelvis_z0
        max_reach = robot_spec.max_reach

        pelvis_z_out = np.zeros(N, dtype=np.float64)
        root_z_filter.reset()
        for i in range(N):
            lw, rw, is_flight = _support_state(motion, i, phase_clock_trusted)

            w_sum = lw + rw
            if w_sum > 1e-6:
                wn = np.array([lw, rw]) / w_sum
                support_z = float(wn[0] * mppi_lfoot[i, 2] + wn[1] * mppi_rfoot[i, 2])
            else:
                support_z = float(terrain.height_at(float(raw_pelvis[i, 0]), float(raw_pelvis[i, 1])))

            z_anchor = support_z + nominal_root_clearance
            z_target = z_anchor + float(raw_style_z[i])

            z_upper_candidates = []
            if lw > 1e-3:
                z_upper_candidates.append(_pelvis_reach_z_upper_bound(raw_pelvis[i, :2], mppi_lfoot[i], lw, max_reach))
            if rw > 1e-3:
                z_upper_candidates.append(_pelvis_reach_z_upper_bound(raw_pelvis[i, :2], mppi_rfoot[i], rw, max_reach))
            if not z_upper_candidates:
                z_upper_candidates.extend(
                    [
                        _pelvis_reach_z_upper_bound(raw_pelvis[i, :2], mppi_lfoot[i], 0.0, max_reach),
                        _pelvis_reach_z_upper_bound(raw_pelvis[i, :2], mppi_rfoot[i], 0.0, max_reach),
                    ]
                )
            z_target = min(z_target, min(z_upper_candidates))

            pelvis_z_out[i] = root_z_filter.step(z_target, is_flight=is_flight)

        pelvis_positions = raw_pelvis.copy()
        pelvis_positions[:, 2] = pelvis_z_out

        # ===== Collision resolution =====
        if params.leg_collision_iters > 0:
            mppi_lfoot, mppi_rfoot = _resolve_lower_leg_foot_collisions(
                motion,
                pelvis_positions,
                mppi_lfoot,
                mppi_rfoot,
                mppi_lfoot_quats,
                mppi_rfoot_quats,
                motion.left_contact,
                motion.right_contact,
                terrain,
                motion.foot_nominal_z,
                n_iters=params.leg_collision_iters,
                shin_radius=params.shin_collision_radius,
                foot_radius=params.foot_collision_radius,
                collision_margin=params.leg_collision_margin,
                push_gain=params.collision_push_gain,
                max_push_per_frame=params.collision_max_push,
                pelvis_side_margin=params.pelvis_side_margin,
                pelvis_side_gain=params.pelvis_side_gain,
            )

        # Remove stance-foot target drift: each stance run keeps the first grounded stance xyz.
        if params.stance_latch:
            mppi_lfoot = _latch_stance_foot_targets(
                mppi_lfoot,
                motion.left_contact,
                terrain,
                motion.foot_nominal_z,
                ground_tol=params.stance_latch_ground_tol,
            )
            mppi_rfoot = _latch_stance_foot_targets(
                mppi_rfoot,
                motion.right_contact,
                terrain,
                motion.foot_nominal_z,
                ground_tol=params.stance_latch_ground_tol,
            )

        left_point_scales = _precompute_contact_point_scales(
            mppi_lfoot,
            mppi_lfoot_quats,
            motion.left_contact,
            terrain,
            robot_spec,
        )
        right_point_scales = _precompute_contact_point_scales(
            mppi_rfoot,
            mppi_rfoot_quats,
            motion.right_contact,
            terrain,
            robot_spec,
        )

        # ===== MuJoCo model + IK setup (model/solver reused from __init__) =====
        model = self.model
        data = mujoco.MjData(model)
        dofs_mapping = self.dofs_mapping
        links_mapping = self.links_mapping
        ik_solver = self.ik_solver
        root_xy_weight = self.root_xy_weight
        root_z_weight = self.root_z_weight

        # Ghost FK model (for offset/ghost recording)
        ghost_model, ghost_data = _build_ghost_fk_model(model)

        n_links_export = len(robot_spec.links_name)
        n_dofs = model.nq - 7  # 29 for G1

        # Buffers: raw is read from motion directly
        # Ghost (offset): raw joints + terrain-adjusted pelvis z
        ghost_dofs_pos = np.zeros((N, n_dofs), dtype=np.float64)
        ghost_links_pos = np.zeros((N, n_links_export, 3), dtype=np.float64)
        ghost_links_quat = np.zeros((N, n_links_export, 4), dtype=np.float64)
        # Optimized (MPPI + IK)
        opt_dofs_pos = np.zeros((N, n_dofs), dtype=np.float64)
        opt_links_pos = np.zeros((N, n_links_export, 3), dtype=np.float64)
        opt_links_quat = np.zeros((N, n_links_export, 4), dtype=np.float64)

        frame_targets, locked_double_stance = _build_frame_ik_targets(
            motion,
            pelvis_positions,
            mppi_lfoot,
            mppi_rfoot,
            mppi_lfoot_quats,
            mppi_rfoot_quats,
            dofs_mapping,
            ik_solver,
            left_point_scales,
            right_point_scales,
            root_xy_weight,
            root_z_weight,
            phase_clock_trusted,
        )

        solved_states, _ = _solve_ik_state_sequence(
            params,
            data,
            ik_solver,
            frame_targets,
            locked_double_stance,
            motion.left_contact[:N],
            motion.right_contact[:N],
            init_progress_prefix=None,
            init_progress_every=200,
            lb_progress_prefix="    [lb-stab]",
            lb_progress_every=200,
        )
        ik_pos_errors = _record_solved_sequence(
            data,
            ghost_model,
            ghost_data,
            ik_solver,
            frame_targets,
            solved_states,
            links_mapping,
            dofs_mapping,
            pelvis_positions,
            opt_dofs_pos,
            opt_links_pos,
            opt_links_quat,
            ghost_dofs_pos,
            ghost_links_pos,
            ghost_links_quat,
            motion,
        )

        # IK error summary
        ik_pos_errors = np.array(ik_pos_errors)
        en.logger.info(
            f"  IK summary: p50={np.percentile(ik_pos_errors, 50):.0f}mm "
            f"p95={np.percentile(ik_pos_errors, 95):.0f}mm "
            f"max={ik_pos_errors.max():.0f}mm  "
            f"bad(>{params.reject_ik_mm:.0f}mm)={np.sum(ik_pos_errors > float(params.reject_ik_mm))}/{N}"
        )

        raw_dofs_pos = motion.dofs_pos[:N].copy()
        raw_links_pos = motion.links_pos[:N].copy()
        raw_links_quat = motion.links_quat[:N].copy()

        # Export lowpass on the optimized output only (raw and ghost stay unfiltered,
        # matching the upstream batch export path).
        effective_cutoff = params.lowpass_cutoff_hz
        if effective_cutoff > 0:
            opt_dofs_pos, opt_links_pos, opt_links_quat = filter_lower_body_motion_for_export(
                model,
                robot_spec,
                dofs_mapping,
                links_mapping,
                opt_dofs_pos,
                opt_links_pos,
                opt_links_quat,
                fps,
                cutoff_hz=effective_cutoff,
                order=params.lowpass_order,
            )

        return AdaptResult(
            raw=MotionSet(dofs_pos=raw_dofs_pos, links_pos_w=raw_links_pos, links_quat_w=raw_links_quat),
            ghost=MotionSet(dofs_pos=ghost_dofs_pos, links_pos_w=ghost_links_pos, links_quat_w=ghost_links_quat),
            optimized=MotionSet(dofs_pos=opt_dofs_pos, links_pos_w=opt_links_pos, links_quat_w=opt_links_quat),
            left_contact=motion.left_contact.copy(),
            right_contact=motion.right_contact.copy(),
            ik_pos_errors=ik_pos_errors.copy(),
            ik_bad_frame_count=int(np.sum(ik_pos_errors > float(params.reject_ik_mm))),
        )
