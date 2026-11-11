"""The HUSKY skateboarding port's frozen I/O contract, against the released checkpoint's ONNX metadata.

Numbers from the ONNX metadata and the compiled mjlab training scene (TeleHuman/humanoid_skateboarding);
a drift here runs the policy in a shifted frame with no error.
"""

from __future__ import annotations

import math

import pytest

from eden.extensions.policies.husky import constants as C

# joint_stiffness / joint_damping / action_scale from the ONNX metadata, by actuator group.
_ONNX_KP = {"7520_14": 40.179, "7520_22": 99.098, "5020x2": 28.501, "5020": 14.251}
_ONNX_KD = {"7520_14": 2.558, "7520_22": 6.309, "5020x2": 1.814, "5020": 0.907}
_ONNX_ACTION_SCALE = {"7520_14": 0.548, "7520_22": 0.351, "5020x2": 0.439, "5020": 0.439}
_GROUP_OF = {
    **dict.fromkeys(("hip_pitch", "hip_yaw", "waist_yaw"), "7520_14"),
    **dict.fromkeys(("hip_roll", "knee"), "7520_22"),
    **dict.fromkeys(("ankle_pitch", "ankle_roll", "waist_roll", "waist_pitch"), "5020x2"),
    **dict.fromkeys(("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow"), "5020"),
}
# default_joint_pos, in POLICY_JOINT_NAMES order.
_ONNX_DEFAULT_POS = [
    0.0, 0.0, 0.0, 0.23, -0.20, 0.0,
    -0.7, 0.0, 0.0, 1.17, -0.45, 0.0,
    0.0, 0.0, 0.0,
    -0.03, 0.45, -0.21, 1.32,
    -0.7, -0.845, 0.83, 1.19,
]  # fmt: skip


def _group(joint: str) -> str:
    return _GROUP_OF[joint.removeprefix("left_").removeprefix("right_").removesuffix("_joint")]


@pytest.mark.parametrize(
    "value, expected",
    [
        (C.NUM_JOINTS, 23),
        (C.FRAME_DIM, 79),
        (sum(C.OBS_BLOCK_DIMS), 79),
        (C.HISTORY_LENGTH, 5),
        (C.ACTOR_OBS_DIM, 395),
        (C.COMMAND_SCALE, (2.0, 1.0)),
        (C.HEADING_SCALE, pytest.approx(1.0 / math.pi)),
        (C.ANG_VEL_SCALE, 0.25),
        (C.DOFS_VEL_SCALE, 0.05),
        (C.CYCLE_TIME_S, 6.0),
        (C.STILL_VX_THRESHOLD, 0.1),
    ],
)
def test_frozen_scalars(value, expected):
    assert value == expected


def test_default_pos_matches_onnx_metadata():
    assert list(C.DEFAULT_DOFS_POS) == pytest.approx(_ONNX_DEFAULT_POS)


@pytest.mark.parametrize(
    "resolved, expected",
    [(C.DOFS_KP, _ONNX_KP), (C.DOFS_KD, _ONNX_KD), (C.ACTION_SCALE, _ONNX_ACTION_SCALE)],
    ids=["kp", "kd", "action_scale"],
)
def test_per_joint_values_match_onnx_metadata(resolved, expected):
    assert len(resolved) == len(C.POLICY_JOINT_NAMES)
    for joint, value in zip(C.POLICY_JOINT_NAMES, resolved):
        assert value == pytest.approx(expected[_group(joint)], abs=5e-3), joint
