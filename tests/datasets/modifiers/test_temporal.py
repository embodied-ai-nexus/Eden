"""Tests for ``eden.datasets.modifiers.temporal``: speed scaling and trimming resample every clip."""

import pytest
import torch

from eden.datasets.modifiers import SpeedScale, TrimFrames


def test_speed_scale_shortens_clips_and_scales_their_velocities(make_sequence):
    seq = make_sequence(buffer_lengths=(10, 6))
    # The first frame survives resampling exactly (phase 0), so it isolates the multiplier.
    first = [seq.base_lin_vel_w[i][0].clone() for i in range(2)]
    mean_abs = seq.base_lin_vel_w.concatenated_tensor.abs().mean().item()

    out = SpeedScale(options=SpeedScale.configure(speed=2.0)).apply_to_sequence(seq)

    assert out.buffer_length.tolist() == [5, 3]
    for i in range(2):
        torch.testing.assert_close(out.base_lin_vel_w[i][0], 2.0 * first[i])
    # Linear interpolation preserves the mean magnitude, so it roughly doubles.
    assert out.base_lin_vel_w.concatenated_tensor.abs().mean().item() == pytest.approx(2.0 * mean_abs, rel=0.15)


def test_speed_scale_of_one_keeps_every_length(make_sequence):
    assert SpeedScale(options=SpeedScale.configure(speed=1.0)).apply_to_sequence(
        make_sequence()
    ).buffer_length.tolist() == [5, 3]


def test_trim_frames_shortens_each_clip(make_sequence):
    out = TrimFrames(options=TrimFrames.configure(trim_first=2, trim_last=1)).apply_to_sequence(
        make_sequence(buffer_lengths=(10, 6))
    )
    assert out.buffer_length.tolist() == [7, 3]


@pytest.mark.parametrize(
    ("cls", "kwargs", "lengths", "message"),
    [
        (SpeedScale, {"speed": -1.0}, (5, 3), "must be > 0"),
        (TrimFrames, {"trim_first": 3, "trim_last": 3}, (5,), "leaves"),
        (TrimFrames, {"trim_first": 2, "trim_last": 2}, (5,), r"leaves 1 frame\(s\) .* needs at least 2"),
        (TrimFrames, {"trim_first": -1}, (5, 3), "non-negative"),
    ],
    ids=["non-positive-speed", "over-trim", "trim-to-one-frame", "negative-trim"],
)
def test_an_invalid_setting_is_refused(make_sequence, cls, kwargs, lengths, message):
    with pytest.raises(ValueError, match=message):
        cls(options=cls.configure(**kwargs)).apply_to_sequence(make_sequence(buffer_lengths=lengths))


def _label_frames(seq):
    """Code every frame's contact label (and flag) with its frame index within the clip, so a trim or resample is readable."""
    for i, n in enumerate(seq.buffer_length.tolist()):
        frames = torch.arange(n, dtype=torch.int8)
        seq.contact_label[i] = frames[:, None].expand(-1, seq.contact_label._data_shape[0]).clone()
        seq.contact_points_valid[i] = (frames % 2 == 0)[:, None, None].expand(-1, *seq.contact_points_valid._data_shape)
    return seq


def test_trim_frames_trims_the_contact_annotations_with_the_rest_of_the_clip(make_sequence):
    out = TrimFrames(options=TrimFrames.configure(trim_first=2, trim_last=1)).apply_to_sequence(
        _label_frames(make_sequence(buffer_lengths=(10, 6)))
    )
    for i, n in enumerate(out.buffer_length.tolist()):
        assert out.contact_label[i].shape[0] == out.contact_points_valid[i].shape[0] == out.dofs_pos[i].shape[0] == n
        assert out.contact_label[i][:, 0].tolist() == list(range(2, 2 + n))


def test_speed_scale_resamples_the_contact_annotations_nearest_neighbour(make_sequence):
    out = SpeedScale(options=SpeedScale.configure(speed=2.0)).apply_to_sequence(
        _label_frames(make_sequence(buffer_lengths=(10, 6)))
    )
    for i, n in enumerate(out.buffer_length.tolist()):
        source_frames = out.contact_label[i][:, 0]
        assert source_frames.shape[0] == out.contact_points_valid[i].shape[0] == n
        assert out.contact_label.dtype == torch.int8 and out.contact_points_valid.dtype == torch.bool
        # A label is a source frame's label, and the flag moved with it: never a blend of two frames.
        assert torch.equal(out.contact_points_valid[i][:, 0, 0], source_frames % 2 == 0)


def test_trim_frames_trims_the_body_model_fields_with_the_rest_of_the_clip(make_sequence):
    from tests.datasets.modifiers.conftest import with_body_model

    out = TrimFrames(options=TrimFrames.configure(trim_first=2, trim_last=1)).apply_to_sequence(
        with_body_model(make_sequence(buffer_lengths=(10, 6)))
    )
    for i, n in enumerate(out.buffer_length.tolist()):
        assert out.body_pose_quat[i].shape == (n, 15, 4) and out.body_shape[i].shape[0] == n
        assert out.body_shape[i][:, 0].tolist() == list(range(2, 2 + n))
    assert out.body_model == ("mano_right", "mano_right")


def test_speed_scale_resamples_the_body_model_fields_on_the_clip_s_own_time_grid(make_sequence):
    from tests.datasets.modifiers.conftest import with_body_model

    seq = with_body_model(make_sequence(buffer_lengths=(10, 6)))
    # Every joint turns 0.1 rad about z per frame, so the pose reads back the source time just as the shape does.
    for i, n in enumerate(seq.buffer_length.tolist()):
        half_angle = 0.05 * torch.arange(n, dtype=torch.float32)[:, None].expand(n, 15)
        seq.body_pose_quat[i] = torch.stack(
            [half_angle.cos(), torch.zeros_like(half_angle), torch.zeros_like(half_angle), half_angle.sin()], -1
        )

    out = SpeedScale(options=SpeedScale.configure(speed=2.0)).apply_to_sequence(seq)

    for i, (old, new) in enumerate(zip((10, 6), out.buffer_length.tolist(), strict=True)):
        source_time = torch.linspace(0.0, old - 1, new)
        torch.testing.assert_close(out.body_shape[i][:, 0], source_time)
        pose = out.body_pose_quat[i]
        torch.testing.assert_close(pose.norm(dim=-1), torch.ones(new, 15))  # slerped, not blended off the sphere
        torch.testing.assert_close(
            2 * torch.atan2(pose[..., 3], pose[..., 0]), 0.1 * source_time[:, None].expand(new, 15)
        )
    assert out.body_model == ("mano_right", "mano_right")
