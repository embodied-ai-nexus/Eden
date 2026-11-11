"""``G1Dofs29Dex3``: the SONIC teleop body's DOF order, hand gains, armature and force limits."""

from __future__ import annotations

import pytest

from eden.options.robots.unitree_g1 import (
    G1_DEX3_LEFT_DOFS_NAME,
    G1_DEX3_RIGHT_DOFS_NAME,
    G1_DOFS29_NAME,
    G1Dofs29,
    G1Dofs29Dex3,
)
from tests.support.sonic import resolve_pattern


def test_dex3_dof_names_and_document_order():
    # 29 body + 7 + 7 hand DOFs, body first, left hand before right.
    robot = G1Dofs29Dex3()
    assert len(robot.dofs_name) == 43
    assert robot.dofs_name[:29] == G1_DOFS29_NAME
    assert robot.dofs_name[29:36] == G1_DEX3_LEFT_DOFS_NAME
    assert robot.dofs_name[36:] == G1_DEX3_RIGHT_DOFS_NAME
    # Dex3 motor order per side: thumb_0/1/2, middle_0/1, index_0/1 (MJCF
    # document order — the reference sim maps Dex3 motor i to the i-th hand
    # joint in this order).
    assert G1_DEX3_LEFT_DOFS_NAME == [
        "left_hand_thumb_0_joint",
        "left_hand_thumb_1_joint",
        "left_hand_thumb_2_joint",
        "left_hand_middle_0_joint",
        "left_hand_middle_1_joint",
        "left_hand_index_0_joint",
        "left_hand_index_1_joint",
    ]
    assert G1_DEX3_RIGHT_DOFS_NAME == [n.replace("left_", "right_") for n in G1_DEX3_LEFT_DOFS_NAME]


def test_hand_gains_and_armature():
    robot = G1Dofs29Dex3()
    for name in G1_DEX3_LEFT_DOFS_NAME + G1_DEX3_RIGHT_DOFS_NAME:
        assert resolve_pattern(robot.default_dofs_kp, name) == 1.5  # dex3_hands.hpp init gains
        assert resolve_pattern(robot.default_dofs_kd, name) == 0.1
        assert resolve_pattern(robot.default_dofs_armature, name) == 0.01  # source MJCF, not Eden's 0.1 default
        assert resolve_pattern(robot.default_dofs_pos, name) == 0.0


def test_hand_force_limits_match_mjcf_actuatorfrcrange():
    robot = G1Dofs29Dex3()
    for side in ("left", "right"):
        assert resolve_pattern(robot.default_dofs_force_limits, f"{side}_hand_thumb_0_joint") == 2.45
        for name in (f"{side}_hand_thumb_1_joint", f"{side}_hand_middle_0_joint", f"{side}_hand_index_1_joint"):
            assert resolve_pattern(robot.default_dofs_force_limits, name) == 1.4


def test_body_joints_keep_g1dofs29_settings():
    base = G1Dofs29()
    dex3 = G1Dofs29Dex3()
    for name in G1_DOFS29_NAME:
        assert resolve_pattern(dex3.default_dofs_kp, name) == resolve_pattern(base.default_dofs_kp, name)
        assert resolve_pattern(dex3.default_dofs_force_limits, name) == resolve_pattern(
            base.default_dofs_force_limits, name
        )
        assert resolve_pattern(dex3.default_dofs_pos, name) == pytest.approx(
            resolve_pattern(base.default_dofs_pos, name)
        )
    assert dex3.default_root_pos == base.default_root_pos


def test_every_dof_matches_exactly_one_pattern():
    # Overlapping patterns raise at env build; catch it here instead.
    robot = G1Dofs29Dex3()
    for name in robot.dofs_name:
        for patterns in (
            robot.default_dofs_pos,
            robot.default_dofs_kp,
            robot.default_dofs_kd,
            robot.default_dofs_armature,
            robot.default_dofs_force_limits,
        ):
            resolve_pattern(patterns, name)
