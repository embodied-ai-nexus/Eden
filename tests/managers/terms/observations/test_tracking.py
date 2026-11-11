"""Tests for ``eden.managers.terms.observations.tracking``: the future-frame motion observations.

The terms read a reference written into a built env's motion buffer (nine frames) and, for the
anchor-frame terms, the real G1's anchor link.
"""

from __future__ import annotations

import math

import genesis.utils.geom as gu
import pytest
import torch

from eden.managers.terms.observations.tracking import (
    entity_base_ori_b,
    entity_base_pos_b,
    motion_future_base_ori_b,
    motion_future_base_pos_b,
    motion_future_dofs_pos,
    motion_future_dofs_vel,
    motion_future_links_ori_b,
    motion_future_links_pos_b,
)
from eden.utils.geom import quat_from_euler_xyz
from tests.support.motion import build_tracking_env, set_reference, write_tracking_library

OFFSETS = (1, 2, 4, 8)


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    return build_tracking_env(write_tracking_library(tmp_path_factory.mktemp("tracking")))


@pytest.fixture
def ramp(env):
    """A distinct value in every (env, frame, dof) slot of the robot's reference."""
    shape = env.motion_manager.data["robot"].dofs_pos.shape
    values = torch.arange(math.prod(shape), dtype=torch.float32).reshape(shape)
    set_reference(env, "robot", dofs_pos=values, dofs_vel=0.1 * values)
    return values


def _anchor(env):
    robot = env.entities["robot"]
    idx = robot.find_named_links_idx_local(["pelvis"], preserve_order=True)[1]
    return robot.get_links_pos(ls_idx_local=idx)[:, 0], robot.get_links_quat(ls_idx_local=idx)[:, 0]


def test_future_dofs_pick_the_offset_frames(env, ramp):
    offsets = list(OFFSETS)

    flat = motion_future_dofs_pos(env, entity_name="robot", frame_offsets=OFFSETS)
    structured = motion_future_dofs_pos(env, entity_name="robot", frame_offsets=OFFSETS, flatten=False)
    vel = motion_future_dofs_vel(env, entity_name="robot", frame_offsets=(0,))

    torch.testing.assert_close(structured.cpu(), ramp[:, offsets])
    torch.testing.assert_close(flat.cpu(), ramp[:, offsets].reshape(4, -1))
    torch.testing.assert_close(vel.cpu(), 0.1 * ramp[:, 0])


def test_future_dofs_subset_is_picked_by_name(env, ramp):
    names = env.motion_manager.get_target_dofs_name("robot")

    out = motion_future_dofs_pos(
        env, entity_name="robot", frame_offsets=(1, 2), dofs_name=[names[2], names[0]], flatten=False
    )

    torch.testing.assert_close(out.cpu(), ramp[:, [1, 2]][:, :, [2, 0]])


@pytest.mark.parametrize(
    "term, kwargs",
    [
        (motion_future_dofs_pos, {}),
        (motion_future_links_pos_b, {"links_name": ["pelvis"], "anchor_link_name": "pelvis"}),
    ],
)
@pytest.mark.parametrize("offsets, match", [((1, 2, 9), "num_frames"), ((), "frame_offsets")])
def test_future_offsets_must_fit_the_buffer(env, term, kwargs, offsets, match):
    with pytest.raises(ValueError, match=match):
        term(env, entity_name="robot", frame_offsets=offsets, **kwargs)


@pytest.mark.parametrize(
    "representation, expected",
    [("rot6d", [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]), ("tan_norm", [1.0, 0.0, 0.0, 0.0, 0.0, 1.0])],
)
def test_future_link_orientation_matching_the_anchor_is_the_identity(env, representation, expected):
    _, anchor_quat = _anchor(env)
    set_reference(env, "robot", links_quat_w=anchor_quat[:, None, None, :])

    out = motion_future_links_ori_b(
        env,
        entity_name="robot",
        links_name=["pelvis", "torso_link"],
        anchor_link_name="pelvis",
        frame_offsets=OFFSETS,
        representation=representation,
        flatten=False,
    )

    torch.testing.assert_close(out.cpu(), torch.tensor(expected).expand(4, 4, 2, 6))
    flat = motion_future_links_ori_b(
        env,
        entity_name="robot",
        links_name=["pelvis", "torso_link"],
        anchor_link_name="pelvis",
        frame_offsets=OFFSETS,
        representation=representation,
    )
    assert flat.shape == (4, 4 * 2 * 6)


def test_future_link_orientation_rejects_an_unknown_representation(env):
    with pytest.raises(ValueError, match="representation"):
        motion_future_links_ori_b(
            env,
            entity_name="robot",
            links_name=["pelvis"],
            anchor_link_name="pelvis",
            frame_offsets=(1,),
            representation="bogus",
        )


def test_future_link_positions_are_expressed_in_the_robots_anchor_frame(env):
    robot = env.entities["robot"]
    half = math.sqrt(0.5)
    robot.set_quat(torch.tensor([[half, 0.0, 0.0, half]] * 4, device=env.device))  # yawed +90 deg
    anchor_pos, anchor_quat = _anchor(env)
    torch.testing.assert_close(anchor_quat.abs().cpu(), torch.tensor([[half, 0.0, 0.0, half]] * 4), atol=1e-5, rtol=0)
    links = env.motion_manager.data["robot"].links_pos_w.clone()
    ahead = torch.arange(9, dtype=torch.float32, device=links.device)  # frame k: k m along world +x
    links[:, :, 1] = anchor_pos[:, None] + ahead[None, :, None] * torch.tensor([1.0, 0.0, 0.0], device=links.device)
    set_reference(env, "robot", links_pos_w=links)

    out = motion_future_links_pos_b(
        env,
        entity_name="robot",
        links_name=["torso_link"],
        anchor_link_name="pelvis",
        frame_offsets=OFFSETS,
        flatten=False,
    )

    # World +x is the yawed anchor's -y.
    expected = torch.stack([torch.zeros(4), -torch.tensor(OFFSETS, dtype=torch.float32), torch.zeros(4)], dim=-1)
    torch.testing.assert_close(out.cpu(), expected[None, :, None, :].expand(4, 4, 1, 3), atol=1e-5, rtol=0)
    assert motion_future_links_pos_b(
        env, entity_name="robot", links_name=["pelvis", "torso_link"], anchor_link_name="pelvis", frame_offsets=OFFSETS
    ).shape == (4, 4 * 2 * 3)


# ---------------------------------------------------------------------------
# Cross-entity terms: the object (a free base) in the robot's live pelvis frame
# ---------------------------------------------------------------------------


def _euler(roll: float, pitch: float, yaw: float) -> torch.Tensor:
    return quat_from_euler_xyz(*(torch.tensor([a]) for a in (roll, pitch, yaw)))


def _pose_robot_and_object(env):
    """Tilt + yaw the robot and place the box off-axis, so no frame is an identity by accident."""
    robot, obj = env.entities["robot"], env.entities["object"]
    tilt = _euler(0.3, -0.2, 1.1).repeat(4, 1).to(env.device)
    robot.set_quat(tilt)
    robot.set_pos(torch.tensor([[0.2, -0.1, 0.9]] * 4, device=env.device))
    obj.set_pos(torch.tensor([[1.0, 0.5, 0.3]] * 4, device=env.device))
    obj.set_quat(_euler(0.0, 0.7, 0.4).repeat(4, 1).to(env.device))
    return _anchor(env)


def _in_anchor(anchor_pos, anchor_quat, pos, quat):
    """Independent reference: R_a^T (p - a) and the first two columns of R_a^T R_t, row-major."""
    r_a = gu.quat_to_R(anchor_quat)
    pos_b = torch.einsum("...ji,...j->...i", r_a, pos - anchor_pos)
    rel = r_a.transpose(-1, -2) @ gu.quat_to_R(quat)
    return pos_b, rel[..., :2].flatten(-2)


def test_the_live_object_pose_is_read_in_the_robots_pelvis_frame(env):
    anchor_pos, anchor_quat = _pose_robot_and_object(env)
    obj = env.entities["object"]
    kwargs = {"entity_name": "object", "anchor_entity_name": "robot", "anchor_link_name": "pelvis"}

    pos_b = entity_base_pos_b(env, **kwargs)
    ori_b = entity_base_ori_b(env, **kwargs)

    exp_pos, exp_ori = _in_anchor(anchor_pos, anchor_quat, obj.get_pos(), obj.get_quat())
    assert pos_b.shape == (4, 3) and ori_b.shape == (4, 6)
    torch.testing.assert_close(pos_b, exp_pos, atol=1e-5, rtol=0)
    torch.testing.assert_close(ori_b, exp_ori, atol=1e-5, rtol=0)


def test_the_future_reference_object_pose_is_read_in_the_robots_live_pelvis_frame(env):
    anchor_pos, anchor_quat = _pose_robot_and_object(env)
    frames = torch.arange(9, dtype=torch.float32, device=env.device)
    ref_pos = torch.stack([frames * 0.1, frames * -0.05, 0.3 + 0.01 * frames], dim=-1)  # (9, 3)
    ref_quat = quat_from_euler_xyz(0.1 * frames, 0.2 + 0 * frames, -0.05 * frames)
    set_reference(env, "object", base_pos_w=ref_pos[None], base_quat_w=ref_quat[None])
    kwargs = {
        "entity_name": "object",
        "anchor_entity_name": "robot",
        "anchor_link_name": "pelvis",
        "frame_offsets": OFFSETS,
    }

    pos_b = motion_future_base_pos_b(env, flatten=False, **kwargs)
    ori_b = motion_future_base_ori_b(env, flatten=False, **kwargs)

    offsets = list(OFFSETS)
    exp_pos, exp_ori = _in_anchor(
        anchor_pos[:, None],
        anchor_quat[:, None],
        ref_pos[offsets][None].expand(4, -1, -1),
        ref_quat[offsets][None].expand(4, -1, -1),
    )
    assert pos_b.shape == (4, 4, 3) and ori_b.shape == (4, 4, 6)
    torch.testing.assert_close(pos_b, exp_pos, atol=1e-5, rtol=0)
    torch.testing.assert_close(ori_b, exp_ori, atol=1e-5, rtol=0)
    assert motion_future_base_pos_b(env, **kwargs).shape == (4, 12)
    assert motion_future_base_ori_b(env, representation="tan_norm", **kwargs).shape == (4, 24)


def test_the_future_reference_object_pose_needs_a_base_pose_in_the_reference(env):
    with pytest.raises(ValueError, match="is_free_floating=False"):
        motion_future_base_pos_b(
            env, entity_name="arm", anchor_entity_name="robot", anchor_link_name="pelvis", frame_offsets=(0,)
        )
