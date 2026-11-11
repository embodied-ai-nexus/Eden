"""``OliWuji``: the two welds it emits, and the built rig -- frame, chirality, seating, mass and actuation.

Both failure modes of this weld are silent: a weld that keeps the Wuji hand's 180 deg load alignment
points the hand back up the forearm, and swapped chirality faces the palms outboard. The scene builds
and steps either way.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from genesis.utils import geom as gu

from eden.options.robots import (
    COMPOSITE_ROBOT_REGISTRY,
    OliNoHands,
    OliWuji,
    WujiHand2Beta2WithMount_L,
    WujiHand2Beta2WithMount_R,
)
from eden.options.robots.composite.base import weld_quat
from eden.utils.geom import align_up_and_front
from tests.options.robots.conftest import build_scene, link_pose, links_pos, set_dofs

_TIPS = ["thumb_tip", "index_finger_tip", "middle_finger_tip", "ring_finger_tip", "pinky_tip"]
# (side, link prefix, flange frame, wrist link, palm-normal sign in the wrist frame). Oli's arms hang
# along -Z palm inboard: +Y on the right, -Y on the left, the same convention the Wuji ``front`` declares.
_SIDES = [
    ("right", "r", "right_hand_mount", "right_wrist_roll_link", +1.0),
    ("left", "l", "left_hand_mount", "left_wrist_roll_link", -1.0),
]


# ── options ────────────────────────────────────────────────────────────────────────────────────────


def test_the_rig_expands_into_the_hand_stripped_oli_and_two_mounted_hands():
    """Bare wrists, or the moulded hand sits inside the Wuji one; ``WithMount``, or the flange plate is missing."""
    rig = OliWuji(base_pos=(0.0, 0.0, 0.903))
    entities, attachments = rig.expand()
    assert {n: type(e).__name__ for n, e in entities.items()} == {
        "robot": "OliNoHands",
        "left_hand": "WujiHand2Beta2WithMount_L",
        "right_hand": "WujiHand2Beta2WithMount_R",
    }
    assert tuple(entities["robot"].default_root_pos) == (0.0, 0.0, 0.903)
    assert (len(rig.robot_dofs), len(rig.left_hand_dofs), len(rig.right_hand_dofs)) == (31, 20, 20)
    assert rig.right_grasp_links == [f"r_{t}" for t in _TIPS]
    assert COMPOSITE_ROBOT_REGISTRY.get("oli_wuji") is OliWuji
    assert set(attachments) == {"left_hand", "right_hand"}


def test_the_welds_cancel_the_hand_alignment_at_zero_offset_on_the_flange():
    """180 deg about (1, +/-1, 0)/sqrt(2): the inverse of ``align_up_and_front(up, front)``, not ``mount_quat``."""
    rig = OliWuji()
    for side, hand, sign in (("left", rig.left_hand, -1.0), ("right", rig.right_hand, 1.0)):
        parent, link, offset, quat = rig.attachments()[f"{side}_hand"]
        assert (parent, link, offset) == ("robot", f"{side}_hand_mount", (0.0, 0.0, 0.0))
        assert quat == pytest.approx((0.0, math.sqrt(0.5), sign * math.sqrt(0.5), 0.0))
        np.testing.assert_allclose(gu.quat_to_R(np.asarray(quat)), align_up_and_front(hand.up, hand.front).T, atol=1e-9)


def test_mount_quat_composes_onto_the_cancelled_alignment():
    yaw90 = (math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5))
    rig = OliWuji(mount_quat=yaw90)
    expected = gu.quat_to_R(np.asarray(yaw90)) @ align_up_and_front(rig.right_hand.up, rig.right_hand.front).T
    np.testing.assert_allclose(gu.quat_to_R(np.asarray(rig.attachments()["right_hand"][3])), expected, atol=1e-9)


def test_the_rig_stands_on_its_humanoids_feet_unless_it_overrides_them():
    """``feet_links_name`` is the override (``None`` = inherit); ``feet_links`` is the resolved answer."""
    assert OliWuji().feet_links_name is None
    assert OliWuji().feet_links == tuple(OliNoHands.feet_links_name)
    rig = OliWuji(feet_links_name=("left_sole", "right_sole"))
    assert rig.feet_links == ("left_sole", "right_sole")
    assert tuple(rig.robot.feet_links_name) == tuple(OliNoHands.feet_links_name)  # no write-through
    # An explicit empty tuple is an override too: the velocity estimator's opt-out, as on Go2W.
    assert OliWuji(feet_links_name=()).feet_links == ()


# ── built ──────────────────────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def env():
    """The rig with a pinned pelvis (kinematic checks, not balance), and a naively welded rig beside it.

    The naive rig's right weld keeps the alignment in (``attach_quat = mount_quat``, the obvious-looking
    thing to write); its hands do not collide, so their buried geometry cannot disturb the stepping tests.
    """
    naive = OliWuji(
        robot=OliNoHands(is_fixed_base=True, default_root_pos=(3.0, 0.0, 0.903)),
        robot_name="naive",
        left_hand_name="naive_left",
        right_hand_name="naive_right",
        left_hand=WujiHand2Beta2WithMount_L(is_fixed_base=False, collision=False),
        right_hand=WujiHand2Beta2WithMount_R(is_fixed_base=False, collision=False),
    )
    naive_entities, naive_welds = naive.expand()
    naive_welds["naive_right"] = ("naive", naive.right_ee_link, tuple(naive.mount_offset), tuple(naive.mount_quat))
    return build_scene(
        robot=OliWuji(robot=OliNoHands(is_fixed_base=True)),
        **naive_entities,
        attachments_dict=naive_welds,
    )


def _in_wrist_frame(env, points: np.ndarray, wrist_link: str, robot: str = "robot") -> np.ndarray:
    wrist_pos, wrist_quat = link_pose(env.entities[robot], wrist_link)
    return (points - wrist_pos) @ gu.quat_to_R(wrist_quat)


def _tips_in_wrist_frame(env, hand: str, prefix: str, wrist_link: str, robot: str = "robot") -> np.ndarray:
    return _in_wrist_frame(env, links_pos(env.entities[hand], [f"{prefix}_{t}" for t in _TIPS]), wrist_link, robot)


def _collision_verts(entity, link_name) -> np.ndarray:
    """World-frame collision-hull vertices of one link (``get_verts`` is world-frame after build)."""
    link = entity.get_link(link_name)
    return np.concatenate([np.asarray(geom.get_verts().cpu()).reshape(-1, 3) for geom in link.geoms])


def test_the_rig_expands_into_a_31_dof_humanoid_and_two_20_dof_hands(env):
    assert [env.entities[n].num_dofs for n in ("robot", "left_hand", "right_hand")] == [31, 20, 20]


@pytest.mark.parametrize("side, prefix, mount, wrist, palm", _SIDES)
def test_the_hand_base_frame_coincides_with_the_flange_frame(env, side, prefix, mount, wrist, palm):
    """Mounting on ``*_hand_mount`` makes the whole weld an identity; left uncancelled it is the 180 deg one."""
    mount_pos, mount_quat = link_pose(env.entities["robot"], mount)
    base_pos, base_quat = link_pose(env.entities[f"{side}_hand"], f"{prefix}_mount")
    np.testing.assert_allclose(gu.quat_to_R(mount_quat).T @ gu.quat_to_R(base_quat), np.eye(3), atol=1e-5)
    np.testing.assert_allclose(base_pos, mount_pos, atol=1e-6)


@pytest.mark.parametrize("side, prefix, mount, wrist, palm", _SIDES)
def test_the_fingers_extend_down_the_arm_thumb_forward(env, side, prefix, mount, wrist, palm):
    tips = _tips_in_wrist_frame(env, f"{side}_hand", prefix, wrist)
    assert tips.mean(axis=0)[2] < -0.15  # past the wrist along -Z, where the moulded hand was
    assert tips[0][0] > 0.05  # the thumb points +X


def test_an_uncancelled_weld_really_points_the_hand_up_the_forearm(env):
    """The mutation the two tests above exist to catch, shown to be real rather than a comment."""
    assert _tips_in_wrist_frame(env, "naive_right", "r", "right_wrist_roll_link", robot="naive").mean(axis=0)[2] > 0.05
    emitted = OliWuji().attachments()["right_hand"][3]
    assert emitted == weld_quat(OliWuji().right_hand, (1.0, 0.0, 0.0, 0.0))
    assert not np.allclose(emitted, (1.0, 0.0, 0.0, 0.0), atol=1e-6)


@pytest.mark.parametrize("side, prefix, mount, wrist, palm_sign", _SIDES)
def test_the_palm_faces_inboard_on_each_side(env, side, prefix, mount, wrist, palm_sign):
    """Flexion curls the tips toward the palm, so their motion *is* the palm normal; swapped hands flip it."""
    hand = env.entities[f"{side}_hand"]
    tips = [f"{prefix}_{t}" for t in _TIPS]
    set_dofs(hand, {})
    before = _in_wrist_frame(env, links_pos(hand, tips), wrist).mean(axis=0)
    flex = {f"{prefix}_{f}_{j}": 0.6 for f in ("index_finger", "middle_finger") for j in ("mcp_flex", "pip", "dip")}
    set_dofs(hand, flex)
    after = _in_wrist_frame(env, links_pos(hand, tips), wrist).mean(axis=0)
    set_dofs(hand, {})
    curl = (after - before) / np.linalg.norm(after - before)
    assert curl[1] * palm_sign > 0.5


@pytest.mark.parametrize("side, prefix, mount, wrist, palm", _SIDES)
def test_the_mount_plate_seats_flush_on_the_wrist_flange(env, side, prefix, mount, wrist, palm):
    """Against the built collision geometry, since Genesis filters the weld pair out of contact reporting.

    This caught the flange frame 2.1 cm out when the asset was revised to strip the wrist end-cap.
    """
    stub = _in_wrist_frame(env, _collision_verts(env.entities["robot"], wrist), wrist)
    plate = _in_wrist_frame(env, _collision_verts(env.entities[f"{side}_hand"], f"{prefix}_mount"), wrist)
    seam = float(stub[:, 2].min() - plate[:, 2].max())
    assert abs(seam) < 1e-4, f"flange seam is {seam * 1e3:+.3f} mm"
    buried = ((plate[:, 2] > stub[:, 2].min() + 1e-4) & (plate[:, 2] < stub[:, 2].max())).sum()
    assert buried == 0, f"{buried}/{len(plate)} plate vertices inside the wrist hull"


def test_each_welded_hand_weighs_what_a_real_one_does(env):
    """~0.69 kg per hand, mount plate included; OliNoHands' bare wrists are pinned in ``test_limx_oli``."""
    hands = sum(float(env.entities[f"{s}_hand"].get_links_inertial_mass().sum()) for s in ("left", "right"))
    assert hands == pytest.approx(1.384, abs=0.02)


def test_the_hands_collide_with_nothing_of_the_arm_and_track_their_own_pd_targets(env):
    """The weld fixes the hand's base link only; all 20 finger DOFs stay actuated."""
    env.scene.step()
    for side in ("left", "right"):
        contacts = env.entities[f"{side}_hand"].get_contacts(with_entity=env.entities["robot"])
        assert int(contacts["valid_mask"].sum()) == 0

    hand = env.entities["right_hand"]
    probe = ("r_index_finger_mcp_flex", "r_middle_finger_mcp_flex", "r_thumb_cmc_flex")
    target = hand.get_dofs_pos().clone()
    for name in probe:
        target[:, hand.dofs_name.index(name)] = 0.8
    hand.control_dofs_pos(target)
    try:
        for _ in range(300):
            env.scene.step()
        reached = hand.get_dofs_pos()
        for name in probe:
            assert float(reached[0, hand.dofs_name.index(name)]) == pytest.approx(0.8, abs=0.05)
    finally:
        hand.control_dofs_pos(torch.zeros_like(target))
