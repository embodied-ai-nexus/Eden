"""Joint-order permutations and per-joint constants for the GEAR-SONIC policy.

GEAR-SONIC was trained in IsaacLab joint order (interleaved L/R groups), while
Eden's :class:`~eden.options.robots.unitree_g1.G1Dofs29` uses MJCF joint order
(all left joints, then all right joints, then waist, then arms). These tables
reproduce ``gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/policy_parameters.hpp``
so the policy I/O can be mapped between the two conventions.

Naming convention follows the upstream C++:

    ``IL_TO_MJ[mujoco_idx] = isaaclab_idx``
    ``MJ_TO_IL[isaaclab_idx] = mujoco_idx``

i.e. each table is used as a gather-index that converts FROM the order named
on the *right* of ``_TO_`` TO the order named on the *left*. For example,
``t_isaaclab = t_mujoco[..., MJ_TO_IL]`` picks each isaaclab slot's value out
of the mujoco-ordered source, and ``t_mujoco = t_isaaclab[..., IL_TO_MJ]``
goes back. (The naming is unfortunate but kept consistent with the upstream
C++ deployment code so cross-references stay greppable.)

Constants come from ``policy_parameters.hpp``:

* ``DEFAULT_ANGLES_MJ`` — standing pose, MJCF / Eden order (29).
* ``ACTION_SCALE_MJ`` — ``0.25 * EFFORT_LIMIT / kp`` per joint, MJCF / Eden order (29).
* ``SONIC_DOFS_POS`` / ``SONIC_DOFS_KP`` / ``SONIC_DOFS_KD`` — the standing pose and PD
  gains the released checkpoint is deployed under, keyed for ``RobotOptions``.

The gains and the action scale follow the same Raibert heuristic Eden's stock
:class:`~eden.options.robots.unitree_g1.G1Dofs29` does, over the same per-joint motor
table — with one substitution: SONIC's hip pitch is a **7520-22**, where Eden's stock
G1 spec says 7520-14 (the upstream header notes it as "old is 7520_14 new is
7520_22"). :data:`SONIC_HIP_PITCH_SPEC` is that substitution, and every table below
derives from it, so the gains and the action scale cannot drift apart.
"""

from __future__ import annotations

from fnmatch import fnmatchcase

from eden.options.actuators import Unitree7520_22
from eden.options.robots.unitree_g1 import G1_ACTION_SCALE, G1Dofs29

# ---------------------------------------------------------------------------
# Joint-order permutations (29 entries each, MJCF order ↔ IsaacLab order).
# Values lifted verbatim from policy_parameters.hpp.
# ---------------------------------------------------------------------------

IL_TO_MJ: tuple[int, ...] = (
    0,
    3,
    6,
    9,
    13,
    17,
    1,
    4,
    7,
    10,
    14,
    18,
    2,
    5,
    8,
    11,
    15,
    19,
    21,
    23,
    25,
    27,
    12,
    16,
    20,
    22,
    24,
    26,
    28,
)

MJ_TO_IL: tuple[int, ...] = (
    0,
    6,
    12,
    1,
    7,
    13,
    2,
    8,
    14,
    3,
    9,
    15,
    22,
    4,
    10,
    16,
    23,
    5,
    11,
    17,
    24,
    18,
    25,
    19,
    26,
    20,
    27,
    21,
    28,
)

# Lower body (12 DOFs): hips + knees + ankles, in MJCF order. The values are
# the MATCHING isaaclab indices for each mujoco position 0..11.
LOWER_BODY_IL_INDICES: tuple[int, ...] = tuple(IL_TO_MJ[i] for i in range(12))


# ---------------------------------------------------------------------------
# Standing pose (MJCF / Eden order, 29 entries). Mirrors policy_parameters.hpp
# `default_angles[]` and matches ProtoMotionV3's `_PD_ACTION_OFFSET` exactly.
# ---------------------------------------------------------------------------

DEFAULT_ANGLES_MJ: tuple[float, ...] = (
    -0.312,
    0.0,
    0.0,
    0.669,
    -0.363,
    0.0,  # left leg
    -0.312,
    0.0,
    0.0,
    0.669,
    -0.363,
    0.0,  # right leg
    0.0,
    0.0,
    0.0,  # waist (yaw, roll, pitch)
    0.2,
    0.2,
    0.0,
    0.6,
    0.0,
    0.0,
    0.0,  # left arm + wrist
    0.2,
    -0.2,
    0.0,
    0.6,
    0.0,
    0.0,
    0.0,  # right arm + wrist
)


# ---------------------------------------------------------------------------
# The one motor SONIC's G1 does not share with Eden's stock G1Dofs29 spec.
#
# SONIC derives every per-joint constant from a motor table by the Raibert
# heuristic — ``kp = armature * ω²``, ``kd = 2 * ζ * armature * ω`` (ω = 10 Hz,
# ζ = 2), ``action_scale = 0.25 * effort_limit / kp``. Its table is Eden's, except
# that hip pitch is a 7520-22; ``policy_parameters.hpp`` flags this on each hip-pitch
# row as "old is 7520_14 new is 7520_22".
#
# Everything SONIC-specific below reads this one name, so the gains, the armature-
# derived stiffness and the action scale cannot disagree about which motor the hips
# have (overriding the action scale but not the gains runs the hips at 0.41x the
# released checkpoint's stiffness).
#
# Deliberately *not* pushed into the robot's own ``dofs_spec``: which motor a physical
# G1 carries is a hardware-revision question this table cannot answer, and
# ``RobotOptions`` also feeds armature and effort limits to the solver. The scope here
# is the control gains the released checkpoint expects — see ``SonicPolicy.robot_options``.
# ---------------------------------------------------------------------------

SONIC_HIP_PITCH_SPEC = Unitree7520_22

_HIP_PITCH_PATTERN = "*_hip_pitch_joint"


def _with_sonic_hip_pitch(stock: dict[str, float], value: float) -> dict[str, float]:
    """Return ``stock`` with the hip-pitch entry replaced by SONIC's 7520-22 ``value``.

    Replaces the existing key rather than adding a narrower one: Eden resolves these
    dicts through :func:`~eden.utils.string.resolve_matching_names_values`, which raises
    when two patterns match the same DOF, so an added ``"*_hip_pitch_joint"`` alongside
    the stock entry would be a hard error rather than an override.

    Parameters
    ----------
    stock : dict[str, float]
        A stock ``G1Dofs29`` per-DOF map (``default_dofs_kp`` / ``default_dofs_kd``).
    value : float
        The SONIC hip-pitch value to substitute.

    Returns
    -------
    dict[str, float]
        A new map; ``stock`` is left untouched.
    """
    if _HIP_PITCH_PATTERN not in stock:
        raise KeyError(
            f"Stock G1Dofs29 no longer keys its per-DOF map by {_HIP_PITCH_PATTERN!r} "
            f"(keys: {sorted(stock)}). SONIC's hip-pitch override has to be re-expressed "
            "against the new spelling, or it silently stops applying."
        )
    return {**stock, _HIP_PITCH_PATTERN: value}


# ---------------------------------------------------------------------------
# The robot tables the released checkpoint is deployed under, keyed for
# ``RobotOptions.default_dofs_pos`` / ``default_dofs_kp`` / ``default_dofs_kd``.
#
# The gains mirror ``policy_parameters.hpp``'s ``kps[]`` / ``kds[]``. The doubled entries
# there (``2.0 * STIFFNESS_5020`` on the ankles and waist roll/pitch) are Eden's
# ``Unitree5020_16_DOUBLE`` parallel-linkage spec, so they already agree.
#
# ``SONIC_DOFS_POS`` is ``DEFAULT_ANGLES_MJ`` as an exact-name map. The robot has to
# *stand* in the pose the action transform is centred on: the MuJoCo sim-to-sim backend
# resets to ``RobotOptions.default_dofs_pos`` and the G1 backend's INIT ramps to it, so the
# stock G1 pose would leave the robot 0.37 rad off at the knees (0.9 rad at the elbows) when
# the policy takes over. Identical to Holosoma's ``G1_LOCO_DEFAULT_DOF_POS``.
# ---------------------------------------------------------------------------

_STOCK_G1 = G1Dofs29()

SONIC_DOFS_POS: dict[str, float] = dict(zip(_STOCK_G1.dofs_name, DEFAULT_ANGLES_MJ, strict=True))

SONIC_DOFS_KP: dict[str, float] = _with_sonic_hip_pitch(
    _STOCK_G1.default_dofs_kp, SONIC_HIP_PITCH_SPEC.RAIBERT_HEURISTIC_KP()
)
SONIC_DOFS_KD: dict[str, float] = _with_sonic_hip_pitch(
    _STOCK_G1.default_dofs_kd, SONIC_HIP_PITCH_SPEC.RAIBERT_HEURISTIC_KD()
)


# ---------------------------------------------------------------------------
# Per-joint action scale (MJCF / Eden order, 29 entries) = ``0.25 * effort_limit / kp``.
# ---------------------------------------------------------------------------

ACTION_SCALE_MJ: tuple[float, ...] = tuple(
    (
        SONIC_HIP_PITCH_SPEC.RAIBERT_HEURISTIC_ACTION_SCALE()
        if fnmatchcase(name, _HIP_PITCH_PATTERN)
        else next(v for p, v in G1_ACTION_SCALE.items() if fnmatchcase(name, p))
    )
    for name in _STOCK_G1.dofs_name
)


# ---------------------------------------------------------------------------
# Reference-sim torque clamps for the teleop sim body (MJCF-name patterns).
#
# Where they are tighter than the actuator-spec effort limits: the reference
# MuJoCo sim clips per-joint torque to motor_effort_limit_list
# (g1_29dof_sonic_model12.yaml) intersected with the MJCF actuatorfrcrange —
# hip roll at 88 (not the 7520-22's 139) and every non-thumb-base finger at
# 0.7 (not the MJCF's 1.4).
# ---------------------------------------------------------------------------
# Reference-sim armature: every joint class of the GEAR-SONIC MJCF
# (``g1_29dof_with_hand_sonic.xml``) declares armature 0.01, body and fingers alike.
# Stated explicitly because an empty ``default_dofs_armature`` does *not* defer to
# the asset: the entity falls back to the robot's actuator-spec reflected inertias
# (0.0251 at the knee), and only DOFs without a spec entry keep the MJCF value.
SONIC_MJCF_ARMATURE: float = 0.01

SONIC_FORCE_LIMIT_OVERRIDES_MJ: dict[str, float] = {"*_hip_roll_joint": 88.0}
SONIC_HAND_FORCE_LIMIT_OVERRIDES: dict[str, float] = {
    "*_hand_thumb_[12]_joint": 0.7,
    "*_hand_middle_*_joint": 0.7,
    "*_hand_index_*_joint": 0.7,
}


# ---------------------------------------------------------------------------
# Encoder mode IDs (matches gear_sonic config and the encoder_mode_4 one-hot).
# Mode 3 is the "all-active" fallback used at deploy time; the released SONIC
# checkpoint supports g1 (0), teleop (1), smpl (2). The one-hot vector has
# width 4 to leave room for a future mode.
# ---------------------------------------------------------------------------

MODE_G1: int = 0
MODE_TELEOP: int = 1
MODE_SMPL: int = 2
NUM_MODES: int = 4
