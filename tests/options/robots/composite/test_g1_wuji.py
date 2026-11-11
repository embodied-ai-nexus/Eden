"""``G1Wuji``: the welds it emits, and the built rig -- mount rotation, finger direction, chirality and mass.

As on :class:`OliWuji`, each failure mode of the weld is silent: a weld that keeps the Wuji hand's 180 deg load
alignment, or a mount quaternion about the wrong axis, points the hand somewhere else, and swapped chirality
faces the palms outboard. The scene builds and steps either way.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from genesis.utils import geom as gu

from eden.options.robots import (
    COMPOSITE_ROBOT_REGISTRY,
    G1Dofs29NoHands,
    G1Wuji,
    WujiHand2Beta2WithMount_L,
    WujiHand2Beta2WithMount_R,
)
from eden.options.robots.composite.base import weld_quat
from tests.options.robots.conftest import build_scene, link_pose, links_pos, set_dofs

_TIPS = ["thumb_tip", "index_finger_tip", "middle_finger_tip", "ring_finger_tip", "pinky_tip"]
# (side, link prefix, world-y sign of the side): the left arm hangs at +y, so its palm faces -y (inboard).
_SIDES = [("left", "l", +1.0), ("right", "r", -1.0)]
#: -90 deg about the flange y axis, the rig's mount: the Wuji fingers (-z) onto the G1 forearm (+x).
_MOUNT = gu.quat_to_R(np.array([math.sqrt(0.5), 0.0, -math.sqrt(0.5), 0.0]))


# ── options ────────────────────────────────────────────────────────────────────────────────────────


def test_the_rig_expands_into_the_hand_stripped_g1_and_two_mounted_hands():
    rig = G1Wuji(base_pos=(0.0, 0.0, 0.793))
    entities, attachments = rig.expand()
    assert {n: type(e).__name__ for n, e in entities.items()} == {
        "robot": "G1Dofs29NoHands",
        "left_hand": "WujiHand2Beta2WithMount_L",
        "right_hand": "WujiHand2Beta2WithMount_R",
    }
    assert tuple(entities["robot"].default_root_pos) == (0.0, 0.0, 0.793)
    assert (len(rig.robot_dofs), len(rig.left_hand_dofs), len(rig.right_hand_dofs)) == (29, 20, 20)
    assert rig.right_grasp_links == [f"r_{t}" for t in _TIPS]
    assert COMPOSITE_ROBOT_REGISTRY.get("g1_wuji") is G1Wuji
    assert set(attachments) == {"left_hand", "right_hand"}


def test_each_hand_is_welded_to_its_ee_link_with_the_mount_through_weld_quat():
    """One quaternion for both sides: the chirality lives in the ``_L`` / ``_R`` hand assets."""
    rig = G1Wuji()
    for side, hand in (("left", rig.left_hand), ("right", rig.right_hand)):
        parent, link, offset, quat = rig.attachments()[f"{side}_hand"]
        assert (parent, link, offset) == ("robot", f"{side}_ee_link", (0.0, 0.0, 0.0))
        assert quat == pytest.approx(weld_quat(hand, rig.mount_quat))


# ── built ──────────────────────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def env():
    """The rig in its zero pose (arms hanging), plus a rig welded without ``weld_quat`` beside it.

    The G1 MJCF carries a free joint, so the base cannot be pinned: the checks are kinematic, read before any step.
    """
    naive = G1Wuji(
        robot=G1Dofs29NoHands(default_root_pos=(3.0, 0.0, 0.793)),
        robot_name="naive",
        left_hand_name="naive_left",
        right_hand_name="naive_right",
        left_hand=WujiHand2Beta2WithMount_L(is_fixed_base=False, collision=False),
        right_hand=WujiHand2Beta2WithMount_R(is_fixed_base=False, collision=False),
    )
    naive_entities, naive_welds = naive.expand()
    naive_welds["naive_right"] = ("naive", naive.right_ee_link, tuple(naive.mount_offset), tuple(naive.mount_quat))
    return build_scene(robot=G1Wuji(), **naive_entities, attachments_dict=naive_welds)


def _tips_in(env, hand: str, prefix: str, frame_link: str, robot: str = "robot") -> np.ndarray:
    pos, quat = link_pose(env.entities[robot], frame_link)
    return (links_pos(env.entities[hand], [f"{prefix}_{t}" for t in _TIPS]) - pos) @ gu.quat_to_R(quat)


def test_the_rig_expands_into_a_29_dof_humanoid_and_two_20_dof_hands(env):
    assert [env.entities[n].num_dofs for n in ("robot", "left_hand", "right_hand")] == [29, 20, 20]


@pytest.mark.parametrize("side, prefix, y_sign", _SIDES)
def test_the_hand_base_sits_on_the_ee_link_turned_by_the_mount(env, side, prefix, y_sign):
    """The flange is the ``*_ee_link`` 4 cm down the forearm; the hand base is it, turned -90 deg about y."""
    flange_pos, flange_quat = link_pose(env.entities["robot"], f"{side}_ee_link")
    base_pos, base_quat = link_pose(env.entities[f"{side}_hand"], f"{prefix}_mount")
    np.testing.assert_allclose(gu.quat_to_R(flange_quat).T @ gu.quat_to_R(base_quat), _MOUNT, atol=1e-5)
    np.testing.assert_allclose(base_pos, flange_pos, atol=1e-6)
    wrist_pos, wrist_quat = link_pose(env.entities["robot"], f"{side}_wrist_yaw_link")
    np.testing.assert_allclose((flange_pos - wrist_pos) @ gu.quat_to_R(wrist_quat), [0.04, 0.0, 0.0], atol=1e-5)


@pytest.mark.parametrize("side, prefix, y_sign", _SIDES)
def test_the_fingers_continue_the_forearm_thumb_on_top(env, side, prefix, y_sign):
    tips = _tips_in(env, f"{side}_hand", prefix, f"{side}_wrist_yaw_link")
    assert tips.mean(axis=0)[0] > 0.15  # past the wrist along the forearm (+x)
    assert tips[0][2] > 0.05  # the thumb on the wrist frame's +z: forward with the arm hanging
    hanging = _tips_in(env, f"{side}_hand", prefix, "pelvis")
    assert hanging.mean(axis=0)[2] < hanging[0][2] and hanging[0][0] > hanging.mean(axis=0)[0]


def test_an_uncancelled_weld_points_the_hand_elsewhere(env):
    """The mutation the tests above exist to catch, shown to be real rather than a comment."""
    tips = _tips_in(env, "naive_right", "r", "right_wrist_yaw_link", robot="naive")
    assert tips.mean(axis=0)[0] < 0.15


@pytest.mark.parametrize("side, prefix, y_sign", _SIDES)
def test_the_palm_faces_the_body_midline(env, side, prefix, y_sign):
    """Flexion curls the tips toward the palm, so their motion *is* the palm normal; swapped hands flip it."""
    hand = env.entities[f"{side}_hand"]
    tips = [f"{prefix}_{t}" for t in _TIPS]
    set_dofs(hand, {})
    before = links_pos(hand, tips).mean(axis=0)
    set_dofs(
        hand, {f"{prefix}_{f}_{j}": 0.6 for f in ("index_finger", "middle_finger") for j in ("mcp_flex", "pip", "dip")}
    )
    after = links_pos(hand, tips).mean(axis=0)
    set_dofs(hand, {})
    curl = (after - before) / np.linalg.norm(after - before)
    assert -y_sign * curl[1] > 0.5  # toward y = 0 from the arm's side


def test_each_welded_hand_weighs_what_a_real_one_does(env):
    hands = sum(float(env.entities[f"{s}_hand"].get_links_inertial_mass().sum()) for s in ("left", "right"))
    assert hands == pytest.approx(1.384, abs=0.02)


def test_the_hands_collide_with_nothing_of_the_arm(env):
    env.scene.step()
    for side in ("left", "right"):
        contacts = env.entities[f"{side}_hand"].get_contacts(with_entity=env.entities["robot"])
        assert int(contacts["valid_mask"].sum()) == 0
