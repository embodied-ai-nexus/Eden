"""Tests for ``eden.utils.motion_reference.lookup``: reading a motion reference against a built entity."""

from __future__ import annotations

import math

import pytest
import torch

from eden.utils.geom import quat_mul
from eden.utils.motion_reference.lookup import (
    anchor_relative_links_once,
    anchor_state_once,
    assert_motion_field_present,
    compute_anchor_relative_links,
    compute_anchor_relative_links_multi_future,
    current_motion_frame,
    select_motion_frames,
)
from tests.support.motion import build_tracking_env, set_reference, write_tracking_library


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    return build_tracking_env(write_tracking_library(tmp_path_factory.mktemp("lookup")), num_frames=3)


@pytest.mark.parametrize(
    "entity, field, match",
    [
        ("robot", "base_pos_w", None),
        ("arm", "links_pos_w", None),  # link fields are fine on a fixed base
        ("arm", "base_pos_w", "is_free_floating=False"),
        ("robot", "contact_points_valid", "does not populate 'contact_points_valid'"),
        ("ghost", "base_pos_w", "'ghost' is not tracked by any motion term"),
    ],
)
def test_field_presence_guard(env, entity, field, match):
    if match is None:
        assert_motion_field_present(env, entity, field)
    else:
        with pytest.raises(ValueError, match=match):
            assert_motion_field_present(env, entity, field)


@pytest.mark.parametrize(
    "shape, sel", [((3, 5, 7, 3), [5, 1, 3]), ((3, 5, 7, 4), None), ((3, 5, 11), [0, 4, 9]), ((3, 5, 3), None)]
)
def test_frames_are_windowed_on_axis_1_and_selections_on_axis_2(shape, sel):
    field = torch.randn(*shape)
    sel_idx = None if sel is None else torch.tensor(sel)
    offsets = torch.tensor([1, 3])

    future = select_motion_frames(field, offsets, sel_idx)
    current = current_motion_frame(field, sel_idx)

    expected_future = field[:, [1, 3]] if sel is None else field[:, [1, 3]][:, :, sel]
    torch.testing.assert_close(future, expected_future, rtol=0, atol=0)
    torch.testing.assert_close(current, field[:, 0] if sel is None else field[:, 0][:, sel], rtol=0, atol=0)
    if sel is None:  # the hot current-frame read is a view, not a copy
        assert current.data_ptr() == field.data_ptr()


def _yaw(theta: float) -> torch.Tensor:
    return torch.tensor([math.cos(theta / 2), 0.0, 0.0, math.sin(theta / 2)])


def _robot_anchor(env):
    robot = env.entities["robot"]
    idx = robot.find_named_links_idx_local(["pelvis"], preserve_order=True)[1]
    return robot.get_links_pos(ls_idx_local=idx)[:, 0].cpu(), robot.get_links_quat(ls_idx_local=idx)[:, 0].cpu()


def _write_anchor_and_torso(env, anchor_pos, anchor_quat, torso_pos, torso_quat):
    """Reference link 0 is the pelvis (the anchor) and link 1 the torso; all ``(N, F, 3|4)``."""
    ref = env.motion_manager.data["robot"]
    pos, quat = ref.links_pos_w.clone().cpu(), ref.links_quat_w.clone().cpu()
    pos[:, :, 0], quat[:, :, 0], pos[:, :, 1], quat[:, :, 1] = anchor_pos, anchor_quat, torso_pos, torso_quat
    set_reference(env, "robot", links_pos_w=pos, links_quat_w=quat, validity=torch.ones(4, 3, dtype=torch.bool))


def test_the_single_frame_window_equals_the_single_frame_helper(env):
    env.entities["robot"].set_quat(_yaw(-math.pi / 12).expand(4, 4).to(env.device))
    anchor_quat = _yaw(math.pi / 6).expand(4, 3, 4)
    _write_anchor_and_torso(
        env,
        torch.tensor([0.5, 0.5, 1.0]).expand(4, 3, 3),
        anchor_quat,
        torch.tensor([1.5, 0.5, 1.0]).expand(4, 3, 3),
        _yaw(math.pi / 3).expand(4, 3, 4),
    )

    single = compute_anchor_relative_links(env, "robot", anchor_link_name="pelvis", tracked_link_names=["torso_link"])
    window = compute_anchor_relative_links_multi_future(
        env, "robot", anchor_link_name="pelvis", tracked_link_names=["torso_link"], frame_offsets=(0,)
    )

    for a, b in zip(single, window[:4]):
        torch.testing.assert_close(b.squeeze(1) if b.dim() == 4 else b, a)
    robot_torso = single[2][:, 0]
    assert (single[0][:, 0] - robot_torso).abs().max() > 1e-3  # the warp moved the reference


def test_frame_0_of_the_shared_warp_is_bit_identical_to_the_single_frame_helper(env):
    """The tracking metrics read frame 0 of the warp the rewards ask for: their logged values must not move."""
    env.entities["robot"].set_quat(_yaw(-math.pi / 12).expand(4, 4).to(env.device))
    _write_anchor_and_torso(
        env,
        torch.tensor([0.5, 0.5, 1.0]).expand(4, 3, 3),
        _yaw(math.pi / 6).expand(4, 3, 4),
        torch.tensor([1.5, 0.5, 1.0]).expand(4, 3, 3),
        _yaw(math.pi / 3).expand(4, 3, 4),
    )

    single = compute_anchor_relative_links(env, "robot", anchor_link_name="pelvis", tracked_link_names=["torso_link"])
    pos_rel, quat_rel, robot_pos, robot_quat, _ = anchor_relative_links_once(env, "robot", "pelvis", ["torso_link"])

    for expected, shared in zip(single, (pos_rel[:, 0], quat_rel[:, 0], robot_pos, robot_quat), strict=True):
        assert torch.equal(shared, expected)


def test_the_shared_warp_is_produced_once_per_pass_and_afresh_outside_one(env):
    args = ("robot", "pelvis", ["torso_link"])
    with env.compute_pass():
        first = anchor_relative_links_once(env, *args)
        assert anchor_relative_links_once(env, *args) is first
    assert anchor_relative_links_once(env, *args) is not first


def test_the_anchor_state_is_read_once_per_pass_and_afresh_outside_one(env):
    """The anchor terminations and the BeyondMimic anchor metrics share it within a step's graded pass."""
    with env.compute_pass():
        first = anchor_state_once(env, "robot", "pelvis")
        assert anchor_state_once(env, "robot", "pelvis") is first
    assert anchor_state_once(env, "robot", "pelvis") is not first


def test_each_future_frame_is_warped_by_its_own_anchor(env):
    """Frame k's anchor is the robot's yawed by theta_k, so the warp is yaw(-theta_k)."""
    env.entities["robot"].set_quat(torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 4, device=env.device))
    robot_pos, robot_quat = _robot_anchor(env)
    thetas = [0.0, math.pi / 4, math.pi / 2]
    anchor_quat = torch.stack([quat_mul(_yaw(t).expand(4, 4), robot_quat) for t in thetas], dim=1)
    _write_anchor_and_torso(
        env,
        robot_pos[:, None].expand(4, 3, 3),
        anchor_quat,
        (robot_pos + torch.tensor([1.0, 0.0, 0.0]))[:, None].expand(4, 3, 3),
        torch.tensor([1.0, 0.0, 0.0, 0.0]).expand(4, 3, 4),
    )

    pos, quat, _, _, validity = compute_anchor_relative_links_multi_future(
        env, "robot", anchor_link_name="pelvis", tracked_link_names=["torso_link"], frame_offsets=(0, 1, 2)
    )

    for k, theta in enumerate(thetas):
        expected = robot_pos + torch.tensor([math.cos(theta), -math.sin(theta), 0.0])
        torch.testing.assert_close(pos[:, k, 0].cpu(), expected, atol=1e-5, rtol=0)
        torch.testing.assert_close(quat[:, k, 0].cpu(), _yaw(-theta).expand(4, 4), atol=1e-5, rtol=0)
    assert validity.all()
