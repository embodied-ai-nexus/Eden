"""Shape, joint-order, and per-joint constants for the TWIST2 G1 policy.

TWIST2 (https://github.com/lzyang2000/twist2_mjlab) is an MJLab motion-tracking
task for the Unitree G1. Its deploy-time inference contract lives in
``deploy/policy/twist2_policy.py``; the constants here reproduce that contract
exactly so the same exported ONNX actor can be driven inside an Eden Genesis
environment.

Two pieces of good luck make the Eden port direct:

* TWIST2's ``POLICY_JOINT_NAMES`` (standard MuJoCo G1 order: left leg, right
  leg, waist, left arm, right arm) is **identical** to Eden's
  :data:`~eden.options.robots.unitree_g1.G1_DOFS29_NAME`, so there is no
  IsaacLab↔MJCF permutation to maintain (unlike the SONIC port).
* TWIST2 resolves its action scale from mjlab's ``G1_ACTION_SCALE``, which is
  built from the same actuator spec (effort / stiffness, Raibert heuristic) as
  Eden's :data:`~eden.options.robots.unitree_g1.G1_ACTION_SCALE`. The two were
  verified to be byte-for-byte equal over ``G1_DOFS29_NAME``, so we derive the
  scale from the Eden constant rather than baking magic numbers.

The standing pose is mjlab's ``KNEES_BENT_KEYFRAME`` (hip pitch -0.312, knee
0.669, ankle pitch -0.363, elbow 0.6, shoulder pitch 0.2, shoulder roll
±0.2). Eden's stock G1 default pose is a *different* (home-ish) pose, so the
policy's :meth:`robot_options` overrides ``default_dofs_pos`` with the
knees-bent values below — without that the robot spawns in the wrong pose and
the ``joint_pos - default`` proprioception is offset.
"""

from __future__ import annotations

from fnmatch import fnmatchcase

from eden.options.robots.unitree_g1 import G1_ACTION_SCALE, G1_DOFS29_NAME

# ---------------------------------------------------------------------------
# Joint / body order (must exactly match training)
# ---------------------------------------------------------------------------

# Policy joint order == Eden's MJCF DOF order == mjlab G1 actuator order.
POLICY_JOINT_NAMES: tuple[str, ...] = tuple(G1_DOFS29_NAME)

# ---------------------------------------------------------------------------
# Observation / action shapes (frozen by the exported ONNX graph)
# ---------------------------------------------------------------------------

NUM_JOINTS: int = 29
# Ankle DOF velocities are masked out of the proprioception (matches training).
ANKLE_DOF_INDICES: frozenset[int] = frozenset({4, 5, 10, 11})

BASE_ANG_VEL_SCALE: float = 0.25
JOINT_POS_SCALE: float = 1.0
JOINT_VEL_SCALE: float = 0.05

ACTOR_MIMIC_DIM: int = 35  # root vel_xy(2)+z(1)+rp(2)+yaw_w(1)+joint_pos(29)
ACTOR_PROPRIO_DIM: int = 92  # ang_vel(3)+imu_rp(2)+q_rel(29)+qd(29)+last_act(29)
ACTOR_CURRENT_DIM: int = ACTOR_MIMIC_DIM + ACTOR_PROPRIO_DIM  # 127
ACTOR_HISTORY_LENGTH: int = 11
# Flat actor obs = current ++ history(HISTORY_LENGTH frames, newest last).
ACTOR_OBS_DIM: int = ACTOR_CURRENT_DIM * (1 + ACTOR_HISTORY_LENGTH)  # 1524

BLEND_DURATION: float = 3.0  # seconds for blend-in / blend-out
DEFAULT_HEIGHT: float = 0.76  # KNEES_BENT_KEYFRAME pelvis height

# ---------------------------------------------------------------------------
# Standing pose (mjlab KNEES_BENT_KEYFRAME). The dict feeds the robot's
# ``default_dofs_pos`` override; the resolved 29-vector feeds the obs/action.
# ---------------------------------------------------------------------------

# fnmatch patterns → joint angle [rad]. First match wins (matches the mjlab
# regex keyframe; specific names listed before any wildcard that could shadow
# them). Joints with no entry default to 0.0.
KNEES_BENT_DOFS_POS: dict[str, float] = {
    "*_hip_pitch_joint": -0.312,
    "*_hip_roll_joint": 0.0,
    "*_hip_yaw_joint": 0.0,
    "*_knee_joint": 0.669,
    "*_ankle_pitch_joint": -0.363,
    "*_ankle_roll_joint": 0.0,
    "waist_yaw_joint": 0.0,
    "waist_roll_joint": 0.0,
    "waist_pitch_joint": 0.0,
    "left_shoulder_pitch_joint": 0.2,
    "right_shoulder_pitch_joint": 0.2,
    "left_shoulder_roll_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
    "*_shoulder_yaw_joint": 0.0,
    "*_elbow_joint": 0.6,
    "*_wrist_roll_joint": 0.0,
    "*_wrist_pitch_joint": 0.0,
    "*_wrist_yaw_joint": 0.0,
}


def _resolve(patterns: dict[str, float], names: tuple[str, ...], default: float = 0.0) -> tuple[float, ...]:
    """First-match (fnmatch) resolution of a {pattern: value} dict over ``names``."""
    out: list[float] = []
    for name in names:
        value = default
        for pattern, v in patterns.items():
            if fnmatchcase(name, pattern):
                value = v
                break
        out.append(value)
    return tuple(out)


# Resolved 29-vectors in POLICY_JOINT_NAMES order.
DEFAULT_POS: tuple[float, ...] = _resolve(KNEES_BENT_DOFS_POS, POLICY_JOINT_NAMES)
JOINT_SCALES: tuple[float, ...] = _resolve(G1_ACTION_SCALE, POLICY_JOINT_NAMES)
JOINT_VEL_SCALES: tuple[float, ...] = tuple(
    0.0 if i in ANKLE_DOF_INDICES else JOINT_VEL_SCALE for i in range(NUM_JOINTS)
)
