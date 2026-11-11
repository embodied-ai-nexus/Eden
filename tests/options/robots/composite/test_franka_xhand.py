"""``FrankaXHand``: a Franka arm with a 12-DOF XHand on the LeFranX coupling, joint-space driven rather than opened and
closed.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from genesis.utils import geom as gu

from eden.options.robots import COMPOSITE_ROBOT_REGISTRY, FrankaXHand
from tests.options.robots.conftest import build_scene, link_pose

_TIPS = ["left_hand_thumb_rota_tip", "left_hand_index_rota_tip", "left_hand_mid_tip", "left_hand_ring_tip"]
_LEFRANX_OFFSET = np.array([0.0, 0.037, 0.046])
_LEFRANX_QUAT = np.array([0.5, 0.5, -0.5, 0.5])
_FRANKA_READY = (0.0, -math.pi / 4, 0.0, -3 * math.pi / 4, 0.0, math.pi / 2, math.pi / 4)


def test_the_rig_welds_an_xhand_on_the_attachment_body_through_the_coupling_transform():
    rig = FrankaXHand(base_pos=(0.0, 0.0, 0.4))
    entities, attachments = rig.expand()
    assert {n: type(e).__name__ for n, e in entities.items()} == {"robot": "FrankaEmikaPandaNoHand", "hand": "XHand1_L"}
    child, link, pos, quat = attachments["hand"]
    assert (child, link) == ("robot", "attachment")
    np.testing.assert_allclose(pos, _LEFRANX_OFFSET)
    np.testing.assert_allclose(quat, _LEFRANX_QUAT)
    assert len(rig.hand_dofs) == 12
    assert rig.grasp_links == rig.fingertip_links == [*_TIPS, "left_hand_pinky_tip"]
    assert COMPOSITE_ROBOT_REGISTRY.get("franka_x_hand") is FrankaXHand


def test_a_dex_hand_has_no_scalar_open_close_target():
    """That API belongs to the two-finger manipulator only."""
    assert not hasattr(FrankaXHand(), "gripper_open")


def test_the_coupling_turns_the_fingers_square_to_the_flange_axis_and_the_palm_along_it():
    """The hand frame: +x palm normal, +z fingers. The attachment frame's +z is the flange axis (out of the arm)."""
    rot = gu.quat_to_R(_LEFRANX_QUAT)
    np.testing.assert_allclose(rot[:, 0], [0.0, 0.0, 1.0], atol=1e-12)  # palm faces along the flange axis
    np.testing.assert_allclose(rot[:, 2], [0.0, -1.0, 0.0], atol=1e-12)  # fingers perpendicular to it
    # The wrist face is 46 mm out from the flange face and 37 mm off its axis, opposite the way the fingers point.
    assert _LEFRANX_OFFSET[2] == pytest.approx(0.046)
    assert float(np.dot(_LEFRANX_OFFSET[:2], rot[:2, 2])) == pytest.approx(-0.037)


@pytest.fixture(scope="module")
def _rig_env():
    return build_scene(robot=FrankaXHand())


@pytest.fixture
def rig_env(_rig_env):
    yield _rig_env
    _rig_env.reset()


def test_the_hand_root_sits_at_the_coupling_transform_off_the_attachment_in_any_arm_pose(rig_env):
    arm, hand = rig_env.entities["robot"], rig_env.entities["hand"]
    lo, hi = (bound[0].cpu().numpy() for bound in arm.get_dofs_limit())
    twisted = lo + np.random.default_rng(0).uniform(0.2, 0.8, lo.shape) * (hi - lo)
    for q in (np.array(_FRANKA_READY), twisted):
        arm.set_qpos(torch.as_tensor(q, dtype=torch.float32, device=rig_env.device)[None])
        ee_pos, ee_quat = link_pose(arm, "attachment")
        hand_pos, hand_quat = link_pose(hand, "left_hand_link")
        np.testing.assert_allclose(hand_pos, ee_pos + gu.transform_by_quat(_LEFRANX_OFFSET, ee_quat), atol=1e-5)
        np.testing.assert_allclose(
            gu.quat_to_R(hand_quat), gu.quat_to_R(gu.transform_quat_by_quat(_LEFRANX_QUAT, ee_quat)), atol=1e-4
        )


def test_at_the_franka_ready_pose_the_hand_points_left_palm_down_thumb_to_the_base(rig_env):
    """The clocking measured on the lab robot."""
    arm, hand = rig_env.entities["robot"], rig_env.entities["hand"]
    arm.set_qpos(torch.tensor([_FRANKA_READY], dtype=torch.float32, device=rig_env.device))
    hand_pos, hand_quat = link_pose(hand, "left_hand_link")
    base_pos, _ = link_pose(arm, "link0")
    rot = gu.quat_to_R(hand_quat)
    np.testing.assert_allclose(rot[:, 2], [0.0, -1.0, 0.0], atol=1e-4)  # fingertips toward -y
    np.testing.assert_allclose(rot[:, 0], [0.0, 0.0, -1.0], atol=1e-4)  # palm down
    np.testing.assert_allclose(rot[:, 1], [1.0, 0.0, 0.0], atol=1e-4)  # index -> pinky away from the base
    np.testing.assert_allclose(hand_pos - base_pos, [0.307, 0.037, 0.544], atol=2e-3)
