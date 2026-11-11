"""Constants for the Humanoid-GPT G1 motion-tracking policy.

All values are lifted from the Humanoid-GPT reference deploy code
(``tracking/{constants,infer_utils}.py`` in the Humanoid-GPT repo) and are kept
here so the extension is self-contained. The 29-DOF joint ordering used by
Humanoid-GPT (``ACTION_JOINT_NAMES``) is identical to eden's
:data:`eden.options.robots.unitree_g1.G1_DOFS29_NAME`, so every per-joint array
below is in that shared order.
"""

from __future__ import annotations

from fnmatch import fnmatchcase

from eden.options.robots.unitree_g1 import G1_ACTION_SCALE, G1_DOFS29_NAME

# Observation / action dimensions of the released MLP checkpoint
# (``pns_wo_priv216.onnx``): obs 136 -> action 29.
NUM_DOFS: int = 29
ACTION_DIM: int = 29
OBS_DIM: int = 136

# Humanoid-GPT nominal joint pose = ``DEFAULT_QPOS[7:]`` (drops the 7-DOF free
# base). Used as the reference point for BOTH the observation joint deltas and
# the action offset. NOTE: this differs from eden's ``G1Dofs29`` default pose
# (shoulder pitch 0->0.2, shoulder roll +-0.15->+-0.3, elbow 1.5->1.28), so it
# must be baked here rather than read from the robot config.
# Order matches ``G1_DOFS29_NAME``:
#   L-leg(6), R-leg(6), waist yaw/roll/pitch(3), L-arm(7), R-arm(7)
HGPT_DEFAULT_DOFS_POS: tuple[float, ...] = (
    # left leg
    -0.1,
    0.0,
    0.0,
    0.3,
    -0.2,
    0.0,
    # right leg
    -0.1,
    0.0,
    0.0,
    0.3,
    -0.2,
    0.0,
    # waist (yaw, roll, pitch)
    0.0,
    0.0,
    0.0,
    # left arm (shoulder pitch/roll/yaw, elbow, wrist roll/pitch/yaw)
    0.2,
    0.3,
    0.0,
    1.28,
    0.0,
    0.0,
    0.0,
    # right arm
    0.2,
    -0.3,
    0.0,
    1.28,
    0.0,
    0.0,
    0.0,
)
assert len(HGPT_DEFAULT_DOFS_POS) == NUM_DOFS


def resolve_action_scale() -> tuple[float, ...]:
    """Per-joint action scale in ``G1_DOFS29_NAME`` order.

    Humanoid-GPT applies ``motor_target = nominal + action * 0.25 * (torque_limit / kp)``.
    eden's :data:`G1_ACTION_SCALE` is the Raibert-heuristic ``0.25 * effort_limit / kp``
    built from the *same* Unitree actuator specs, so the two are numerically
    identical (verified to < 2e-8). We resolve eden's fnmatch-keyed dict into a
    dense per-joint tuple here instead of re-hardcoding torque/kp tables.
    """
    out: list[float] = []
    for name in G1_DOFS29_NAME:
        value = next(v for pattern, v in G1_ACTION_SCALE.items() if fnmatchcase(name, pattern))
        out.append(float(value))
    return tuple(out)


# ONNX I/O tensor names of the released checkpoint.
ONNX_INPUT_NAME: str = "obs"
ONNX_OUTPUT_NAME: str = "continuous_actions"

# Pelvis CoM offset in the pelvis body frame, from the G1 MJCF pelvis body
# (``<inertial pos="0 0 -0.07605">`` in ``g1_29dof_rev_1_0.xml``; the inertial
# quat is ~identity so the position alone suffices). Used to convert the
# reference pelvis linear velocity from link-origin (eden ``links_lin_vel_w``)
# to the CoM point that MuJoCo ``cvel`` reports: ``v_com = v_origin + w x r``.
PELVIS_COM_OFFSET: tuple[float, float, float] = (0.0, 0.0, -0.07605)
