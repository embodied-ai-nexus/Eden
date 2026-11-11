"""Tests for ``eden.utils.motion_reference.utils``: retiming a motion and finite-differencing its velocities."""

from __future__ import annotations

import math

import pytest
import torch

from eden.utils.motion_reference.motion_data import BatchMotionSequence
from eden.utils.motion_reference.utils import (
    MOTION_FIELD_ROWS,
    estimate_motion_velocities,
    interpolate_body_model,
    interpolate_motion,
    lerp_rows,
    nearest_rows,
    slerp_rows,
)


def _identity(*shape):
    q = torch.zeros(*shape, 4)
    q[..., 0] = 1.0
    return q


def test_interpolation_resamples_linearly_onto_the_target_rate():
    frames = 10  # 0.3 s at 30 Hz -> 18 frames at 60 Hz
    root = torch.zeros(frames, 3)
    root[:, 0] = torch.linspace(0, 1, frames)
    dofs = torch.linspace(0, 5, frames)[:, None].repeat(1, 5)
    base_lin_vel = torch.zeros(frames, 3)
    base_lin_vel[:, 0] = torch.linspace(0, 2, frames)

    out = interpolate_motion(
        root,
        _identity(frames),
        dofs,
        torch.zeros(frames, 2, 3),
        _identity(frames, 2),
        30.0,
        60.0,
        base_lin_vel_w=base_lin_vel,
    )
    root_out, dofs_out, lin_vel_out, ang_vel_out = out[0], out[2], out[5], out[6]

    t = torch.arange(18) / 60.0  # a clip of 0.3 s: x(t) = t / 0.3, and so on
    torch.testing.assert_close(root_out[:, 0], t / 0.3)
    torch.testing.assert_close(dofs_out[:, 3], 5 * t / 0.3)
    torch.testing.assert_close(lin_vel_out[:, 0], 2 * t / 0.3)
    assert ang_vel_out is None  # not given, not invented
    assert out[10] is None and out[11] is None


def test_contact_is_resampled_nearest_neighbour_never_blended():
    frames, links = 10, 2
    valid = torch.zeros(frames, links, 1, dtype=torch.bool)
    valid[::2] = True
    # Each frame's points carry its frame index: a blended point would fall between two.
    points = torch.arange(frames, dtype=torch.float32).view(-1, 1, 1, 1).expand(-1, links, 1, 3).clone()

    out = interpolate_motion(
        torch.zeros(frames, 3),
        _identity(frames),
        None,
        None,
        None,
        20.0,
        50.0,
        contact_points_w=points,
        contact_points_valid=valid,
    )
    new_points, new_valid = out[10], out[11]

    assert new_points.shape[0] == out[0].shape[0] and new_valid.dtype == torch.bool
    assert torch.isin(new_points.unique(), points.unique()).all()
    source_frame = new_points[:, 0, 0, 0].long()
    assert torch.equal(new_valid[:, 0, 0], valid[source_frame, 0, 0])


def test_velocities_of_constant_rate_motion_are_that_rate():
    frames, dt, yaw_rate = 10, 0.1, 0.5
    t = torch.arange(frames) * dt
    root = torch.stack([2.0 * t, torch.zeros(frames), torch.zeros(frames)], dim=-1)
    yaw = torch.stack(
        [torch.cos(yaw_rate * t / 2), torch.zeros(frames), torch.zeros(frames), torch.sin(yaw_rate * t / 2)], -1
    )
    dofs = t[:, None].repeat(1, 5)

    lin_vel, ang_vel, dofs_vel, links_lin_vel, links_ang_vel = estimate_motion_velocities(
        root, yaw, dofs, torch.zeros(frames, 2, 3), _identity(frames, 2), dt
    )

    torch.testing.assert_close(lin_vel, torch.tensor([2.0, 0.0, 0.0]).expand(frames, 3))
    torch.testing.assert_close(dofs_vel, torch.ones(frames, 5))
    # Interior frames: a central difference of a constant-rate yaw is exact.
    torch.testing.assert_close(ang_vel[1:-1], torch.tensor([0.0, 0.0, yaw_rate]).expand(frames - 2, 3))
    torch.testing.assert_close(links_ang_vel, torch.zeros(frames, 2, 3))
    assert math.isclose(float(links_lin_vel.abs().max()), 0.0)


def test_a_two_frame_motion_gets_its_one_interval_on_both_frames_angular_and_linear_alike():
    """A central difference needs three frames; two have one interval, whose one-sided rate both frames take."""
    dt, yaw_rate, roll_rate = 0.1, 0.5, -2.0
    t = torch.tensor([0.0, dt])
    root = torch.stack([2.0 * t, torch.zeros(2), torch.zeros(2)], dim=-1)
    yaw = torch.stack([torch.cos(yaw_rate * t / 2), torch.zeros(2), torch.zeros(2), torch.sin(yaw_rate * t / 2)], -1)
    roll = torch.stack([torch.cos(roll_rate * t / 2), torch.sin(roll_rate * t / 2), torch.zeros(2), torch.zeros(2)], -1)

    lin_vel, ang_vel, _, links_lin_vel, links_ang_vel = estimate_motion_velocities(
        root, yaw, None, root[:, None].repeat(1, 3, 1), roll[:, None].repeat(1, 3, 1), dt
    )

    torch.testing.assert_close(lin_vel, torch.tensor([2.0, 0.0, 0.0]).expand(2, 3))
    torch.testing.assert_close(ang_vel, torch.tensor([0.0, 0.0, yaw_rate]).expand(2, 3))
    torch.testing.assert_close(links_lin_vel, torch.tensor([2.0, 0.0, 0.0]).expand(2, 3, 3))
    torch.testing.assert_close(links_ang_vel, torch.tensor([roll_rate, 0.0, 0.0]).expand(2, 3, 3))


def test_a_one_frame_motion_has_no_velocity_and_says_so():
    with pytest.raises(ValueError, match=r"needs at least 2 frames .* got a sequence of 1 frame\(s\)"):
        estimate_motion_velocities(torch.zeros(1, 3), _identity(1), torch.zeros(1, 5), None, None, 0.02)


def test_contact_labels_are_resampled_with_the_other_contact_fields():
    """The 13th return: the same nearest-neighbour source frame as ``contact_points_valid``."""
    frames = 10
    valid = torch.zeros(frames, 2, 1, dtype=torch.bool)
    valid[::2] = True
    labels = torch.arange(frames, dtype=torch.int8)[:, None].expand(-1, 2).clone()

    out = interpolate_motion(
        torch.zeros(frames, 3),
        _identity(frames),
        None,
        None,
        None,
        20.0,
        50.0,
        contact_points_valid=valid,
        contact_label=labels,
    )
    new_valid, new_labels = out[11], out[12]
    assert new_labels.dtype == torch.int8 and new_labels.shape[0] == out[0].shape[0]
    assert torch.equal(new_valid[:, 0, 0], valid[new_labels[:, 0].long(), 0, 0])
    assert interpolate_motion(torch.zeros(frames, 3), _identity(frames), None, None, None, 20.0, 50.0)[12] is None


def test_body_model_pose_is_slerped_per_joint_and_its_shape_lerped_on_the_motions_grid():
    """A joint turning 90 degrees about z over one source frame is at 45 halfway; a shape coefficient going 0 -> 1 is
    at 0.5; and the output frames are interpolate_motion's.
    """
    pose = _identity(2, 3)
    pose[1, 1] = torch.tensor([math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)])  # joint 1: 90 deg about z
    shape = torch.tensor([[0.0, 2.0], [1.0, 2.0]])

    new_pose, new_shape = interpolate_body_model(pose, shape, 10.0, 20.0)
    assert new_pose.shape == (2, 3, 4) and new_shape.shape == (2, 2)  # 0.1 s at 20 Hz: t = 0, 0.05
    assert len(new_pose) == len(interpolate_motion(torch.zeros(2, 3), _identity(2), None, None, None, 10.0, 20.0)[0])
    half = torch.tensor([math.cos(math.pi / 8), 0.0, 0.0, math.sin(math.pi / 8)])
    torch.testing.assert_close(new_pose[1, 1], half)
    torch.testing.assert_close(new_pose[1, 0], _identity(1)[0])  # an unmoving joint stays put
    torch.testing.assert_close(new_shape[1], torch.tensor([0.5, 2.0]))


def test_base_frame_link_distances_refuse_a_dataset_without_body_frame_fields(tmp_path):
    """A library packed with ``derive_body_frame=False`` has no ``links_*_b``: the base-frame distances must refuse
    it rather than score against the reference buffer's default.
    """
    from eden.utils.motion_reference.utils import get_link_position_distance, get_link_rotation_distance
    from tests.support.motion import build_tracking_env, write_tracking_library

    env = build_tracking_env(write_tracking_library(tmp_path))
    term = next(iter(env.motion_manager._terms.values()))
    source = term.data_source.motion_sequences[term.entities_map["robot"]]
    source.links_pos_b = source.links_quat_b = None  # as the packer leaves an opted-out library

    with pytest.raises(ValueError, match="does not populate 'links_pos_b'"):
        get_link_position_distance(env, "robot", in_base_frame=True)
    with pytest.raises(ValueError, match="does not populate 'links_quat_b'"):
        get_link_rotation_distance(env, "robot", in_base_frame=True)


def test_row_blends_weigh_each_row_by_its_own_blend():
    """The shared gather-and-blend step of ``interpolate_motion`` and the temporal dataset modifiers."""
    seq = torch.arange(12.0).view(4, 3)
    index_0, index_1, blend = torch.tensor([0, 2, 3]), torch.tensor([1, 3, 3]), torch.tensor([0.25, 0.5, 0.0])

    torch.testing.assert_close(
        lerp_rows(seq, index_0, index_1, blend), torch.tensor([[0.75, 1.75, 2.75], [7.5, 8.5, 9.5], [9.0, 10.0, 11.0]])
    )
    assert nearest_rows(seq, index_0, index_1, blend).tolist() == [seq[0].tolist(), seq[3].tolist(), seq[3].tolist()]


def test_slerp_rows_blends_every_link_of_a_row_alike():
    half_turn_z = torch.tensor([0.0, 0.0, 0.0, 1.0])
    seq = torch.stack([_identity(2), half_turn_z.expand(2, 4)])  # (2 frames, 2 links, 4)

    out = slerp_rows(seq, torch.tensor([0, 0]), torch.tensor([1, 1]), torch.tensor([0.0, 0.5]))

    quarter_turn_z = torch.tensor([math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)])
    torch.testing.assert_close(out[0], _identity(2))
    torch.testing.assert_close(out[1], quarter_turn_z.expand(2, 4))
    assert slerp_rows(torch.zeros(3, 0, 4), torch.tensor([0]), torch.tensor([1]), torch.tensor([0.5])).shape == (
        1,
        0,
        4,
    )


def test_every_per_frame_field_has_a_resampling_rule():
    """``SpeedScale`` resamples by this table: a field missing from it would keep its old length, out of step."""
    assert set(MOTION_FIELD_ROWS) == set(BatchMotionSequence.attrs_with_frame_dim)
    assert {name for name, rows in MOTION_FIELD_ROWS.items() if rows is slerp_rows} == {
        "base_quat_w",
        "links_quat_w",
        "links_quat_b",
        "body_pose_quat",
    }
    assert {name for name, rows in MOTION_FIELD_ROWS.items() if rows is nearest_rows} == {
        "contact_points_w",
        "contact_points_valid",
        "contact_label",
    }
