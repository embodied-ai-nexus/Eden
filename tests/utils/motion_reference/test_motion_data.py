"""Tests for ``eden.utils.motion_reference.motion_data``: clip batches and reference buffers."""

from __future__ import annotations

import pytest
import torch

from eden.utils.motion_reference.motion_data import (
    _BODY_FRAME_LINK_FIELDS,
    BatchMotionSequence,
    MotionReferenceData,
    MotionSequence,
)


def _batch(buffer_lengths=(8, 12), num_dofs=3, num_links=2, **metadata):
    seq = BatchMotionSequence.make_empty(buffer_lengths=list(buffer_lengths), num_dofs=num_dofs, num_links=num_links)
    seq.framerate = torch.full((len(buffer_lengths),), 50.0)
    for key, value in metadata.items():
        setattr(seq, key, value)
    return seq


def _clip(num_frames, num_links=3, with_contact=True):
    seq = MotionSequence.make_empty(num_frames, num_dofs=2, num_links=num_links)
    seq.framerate = torch.tensor(20.0)
    seq.buffer_length = torch.tensor(num_frames)
    if not with_contact:
        seq.contact_points_w = seq.contact_points_valid = None
    return seq


def _random_clip(num_frames: int, num_links: int = 2, framerate: float = 50.0) -> MotionSequence:
    """A clip whose every frame field holds distinct random values."""
    fields = {"dofs_pos": (3,), "dofs_vel": (3,), "base_pos_w": (3,), "base_lin_vel_w": (3,)}
    fields |= {"base_quat_w": (4,), "base_ang_vel_w": (3,)}
    if num_links:
        for name in ("links_pos_b", "links_pos_w", "links_lin_vel_b", "links_ang_vel_b", "links_lin_vel_w"):
            fields[name] = (num_links, 3)
        fields |= {"links_quat_b": (num_links, 4), "links_quat_w": (num_links, 4), "links_ang_vel_w": (num_links, 3)}
    return MotionSequence(
        framerate=torch.tensor(framerate),
        buffer_length=torch.tensor(num_frames),
        **{name: torch.randn(num_frames, *shape) for name, shape in fields.items()},
    )


def _assert_same_frames(batch: BatchMotionSequence, clips: list[MotionSequence]) -> None:
    for attr in MotionSequence.attrs_with_frame_dim:
        if getattr(clips[0], attr) is None:
            continue
        for i, clip in enumerate(clips):
            assert torch.equal(getattr(batch, attr)[i], getattr(clip, attr)), (attr, i)


def test_ragged_clips_pack_frame_for_frame_with_their_own_scalars():
    clips = [_random_clip(5, framerate=30.0), _random_clip(10), _random_clip(3, framerate=60.0)]

    batch = BatchMotionSequence.from_motion_sequences(clips)

    assert batch.buffer_length.tolist() == [5, 10, 3]
    assert batch.framerate.tolist() == [30.0, 50.0, 60.0]
    _assert_same_frames(batch, clips)


def test_clips_without_link_data_get_zero_width_link_buffers_unless_told_the_link_count():
    clips = [_random_clip(5, num_links=0), _random_clip(5, num_links=0)]

    assert BatchMotionSequence.from_motion_sequences(clips).links_pos_w.shape == (10, 0, 3)
    assert BatchMotionSequence.from_motion_sequences(clips, num_links=4).links_pos_w.shape == (10, 4, 3)


def test_the_state_dict_holds_only_plain_values_and_round_trips_through_mmap(tmp_path):
    """The dataset cache is ``torch.save`` of this dict, loaded back with ``mmap=True``."""
    clips = [_random_clip(5), _random_clip(7), _random_clip(2)]
    batch = BatchMotionSequence.from_motion_sequences(clips)
    state = batch.to_state_dict()
    assert all(isinstance(v, (torch.Tensor, list, str, int, float, bool)) or v is None for v in state.values())

    torch.save(state, tmp_path / "state.pt")
    restored = BatchMotionSequence.from_state_dict(torch.load(tmp_path / "state.pt", mmap=True, weights_only=False))

    assert torch.equal(restored.framerate, batch.framerate)
    _assert_same_frames(restored, clips)


def test_metadata_defaults_to_a_free_floating_entity_of_no_kind():
    seq = _batch()

    assert seq.is_free_floating is True and seq.entity_kind is None


@pytest.mark.parametrize(
    "derive",
    [
        lambda s: s.select(torch.tensor([0])),
        lambda s: s.select_dofs(slice(0, 2)),
        lambda s: s.select_links(slice(0, 1)),
        lambda s: s.to("cpu"),
        lambda s: BatchMotionSequence.from_state_dict(s.to_state_dict()),
    ],
    ids=["select", "select_dofs", "select_links", "to", "state_dict"],
)
def test_metadata_survives_every_derived_sequence(derive):
    derived = derive(_batch(is_free_floating=False, entity_kind="rigid_object"))

    assert derived.is_free_floating is False and derived.entity_kind == "rigid_object"


def test_a_cache_written_before_the_metadata_loads_with_the_defaults():
    state = _batch(is_free_floating=False, entity_kind="humanoid").to_state_dict()
    for field in BatchMotionSequence.metadata_fields:
        state.pop(f"_metadata/{field}")

    restored = BatchMotionSequence.from_state_dict(state)

    assert restored.is_free_floating is True and restored.entity_kind is None


def _without_body_frame(clip: MotionSequence) -> MotionSequence:
    """``clip`` as ``_pack_motion_sequence(derive_body_frame=False)`` leaves it: world-frame links only."""
    for name in _BODY_FRAME_LINK_FIELDS:
        setattr(clip, name, None)
    return clip


def test_body_frame_fields_left_out_beside_world_frame_links_stay_absent_not_zero():
    """A zero ``links_quat_b`` would pass the field-presence check and read as data."""
    batch = BatchMotionSequence.from_motion_sequences([_without_body_frame(_random_clip(n)) for n in (5, 7)])

    assert (batch.links_pos_b, batch.links_quat_b, batch.links_lin_vel_b, batch.links_ang_vel_b) == (None,) * 4
    assert batch.links_pos_w.shape == (12, 2, 3)
    with pytest.raises(ValueError, match="body-frame link fields must be all-or-none"):
        BatchMotionSequence.from_motion_sequences([_random_clip(5), _without_body_frame(_random_clip(5))])


def test_contact_fields_are_all_or_none_across_a_batch():
    absent = BatchMotionSequence.from_motion_sequences([_clip(5, with_contact=False) for _ in range(3)])
    assert absent.contact_points_w is None and absent.contact_points_valid is None
    assert absent.links_pos_w is not None  # other absent fields zero-fill

    with pytest.raises(ValueError, match="all-or-none"):
        BatchMotionSequence.from_motion_sequences([_clip(5), _clip(5, with_contact=False)])


def test_contact_flags_pack_as_bool_and_follow_link_selection_and_the_state_dict():
    clips = [_clip(5, num_links=4), _clip(7, num_links=4)]
    clips[0].contact_points_valid[:, 3, 0] = True
    batch = BatchMotionSequence.from_motion_sequences(clips)
    assert batch.contact_points_w._data_shape == (4, 1, 3)
    assert batch.contact_points_valid.dtype == torch.bool

    sub = batch.select_links([1, 3])
    flags = sub.contact_points_valid.contiguous()
    assert sub.contact_points_w._data_shape == (2, 1, 3)
    assert flags[:5, 1, 0].all() and not flags[:, 0, 0].any()

    restored = BatchMotionSequence.from_state_dict(batch.to_state_dict())
    assert restored.contact_points_valid.dtype == torch.bool
    assert torch.equal(restored.contact_points_w.contiguous(), batch.contact_points_w.contiguous())


def test_the_contact_partner_axis_follows_the_data():
    two = _clip(4)
    two.contact_points_w = torch.zeros(4, 3, 2, 3)
    two.contact_points_valid = torch.zeros(4, 3, 2, dtype=torch.bool)

    assert _clip(4).num_contact_partners == 1
    assert _clip(4, with_contact=False).num_contact_partners == 1
    assert BatchMotionSequence.from_motion_sequences([two]).num_contact_partners == 2


@pytest.mark.parametrize("partners", [1, 2])
def test_reference_buffers_allocate_empty_contact_fields(partners):
    data = MotionReferenceData.make_empty(2, 4, 3, 5, num_contact_partners=partners)

    assert data.contact_points_w.shape == (2, 4, 5, partners, 3)
    assert data.contact_points_valid.shape == (2, 4, 5, partners)
    assert data.contact_points_valid.dtype == torch.bool and not data.contact_points_valid.any()


def _labelled_clip(num_frames, num_links=4, label=None):
    seq = _clip(num_frames, num_links=num_links, with_contact=False)
    seq.contact_label = label
    return seq


def test_contact_labels_pack_as_int8_and_follow_link_selection_and_the_state_dict():
    labels = [torch.randint(-1, 2, (n, 4), dtype=torch.int8) for n in (5, 7)]
    batch = BatchMotionSequence.from_motion_sequences(
        [_labelled_clip(5, label=labels[0]), _labelled_clip(7, label=labels[1])]
    )
    assert batch.contact_label.dtype == torch.int8 and batch.contact_label._data_shape == (4,)
    assert torch.equal(batch.contact_label[1], labels[1])

    sub = batch.select_links([3, 0])
    assert torch.equal(sub.contact_label[0], labels[0][:, [3, 0]])

    restored = BatchMotionSequence.from_state_dict(batch.to_state_dict())
    assert restored.contact_label.dtype == torch.int8
    assert torch.equal(restored.contact_label.contiguous(), batch.contact_label.contiguous())


def test_contact_labels_are_all_or_none_across_a_batch():
    assert BatchMotionSequence.from_motion_sequences([_labelled_clip(5), _labelled_clip(6)]).contact_label is None
    with pytest.raises(ValueError, match="contact_label.*all-or-none"):
        BatchMotionSequence.from_motion_sequences(
            [_labelled_clip(5, label=torch.zeros(5, 4, dtype=torch.int8)), _labelled_clip(6)]
        )


def test_reference_buffers_allocate_an_unlabelled_contact_label():
    data = MotionReferenceData.make_empty(2, 4, 3, 5)
    assert data.contact_label.shape == (2, 4, 5) and data.contact_label.dtype == torch.int8
    assert not data.contact_label.any()  # 0 = unlabelled, ignored by every consumer


def _body_clip(num_frames: int, model: str | None = "mano_right", joints: int = 15) -> MotionSequence:
    """A clip with the opt-in body-model fields (``model=None``: without them); each frame's values are distinct."""
    clip = _random_clip(num_frames, num_links=0)
    if model is not None:
        quat = torch.randn(num_frames, joints, 4)
        clip.body_pose_quat = quat / quat.norm(dim=-1, keepdim=True)
        clip.body_shape = torch.randn(num_frames, 10)
        clip.body_model = model
    return clip


def test_body_model_fields_pack_per_clip_and_follow_selection_and_the_state_dict():
    clips = [_body_clip(5, "mano_right"), _body_clip(7, "mano_left"), _body_clip(4, "mano_right")]
    batch = BatchMotionSequence.from_motion_sequences(clips)
    assert batch.body_model == ("mano_right", "mano_left", "mano_right")
    assert torch.equal(batch.body_pose_quat[1], clips[1].body_pose_quat)
    assert torch.equal(batch.body_shape[2], clips[2].body_shape)

    # The model names are per clip, so a subset carries its own clips' names, whatever the index form.
    sub = batch.select([2, 1])
    assert sub.body_model == ("mano_right", "mano_left")
    assert torch.equal(sub.body_pose_quat[1], clips[1].body_pose_quat)
    assert batch.select(torch.tensor([False, True, True])).body_model == ("mano_left", "mano_right")
    assert batch.select(slice(1, 2)).body_model == ("mano_left",)
    assert batch.select_links([]).body_model == batch.to("cpu").body_model == batch.body_model

    restored = BatchMotionSequence.from_state_dict(batch.to_state_dict())
    assert restored.body_model == batch.body_model
    assert torch.equal(restored.body_pose_quat.contiguous(), batch.body_pose_quat.contiguous())


def test_body_model_fields_are_all_or_none_and_absent_by_default():
    plain = BatchMotionSequence.from_motion_sequences([_body_clip(5, None), _body_clip(6, None)])
    assert plain.body_pose_quat is None and plain.body_shape is None and plain.body_model is None
    with pytest.raises(ValueError, match="body_pose_quat.*all-or-none"):
        BatchMotionSequence.from_motion_sequences([_body_clip(5), _body_clip(6, None)])
    # No allocator invents them: their shape belongs to the body model.
    assert _batch().body_pose_quat is None
    assert MotionSequence.make_empty(4, num_dofs=2, num_links=1).body_pose_quat is None


def test_a_clip_with_half_of_the_body_model_fields_is_refused():
    clip = _body_clip(5)
    with pytest.raises(ValueError, match="all set or all None; got only body_pose_quat, body_model"):
        MotionSequence(body_pose_quat=clip.body_pose_quat, body_model="mano_right")
    with pytest.raises(ValueError, match="all set or all None; got only body_model"):
        MotionSequence(base_pos_w=torch.zeros(5, 3), body_model="mano_right")
