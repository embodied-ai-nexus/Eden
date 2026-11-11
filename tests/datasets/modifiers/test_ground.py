"""Tests for ``eden.datasets.modifiers.ground.GroundFirstFrame``, which grounds each clip in its own Genesis scene."""

from __future__ import annotations

import pytest
import torch

from eden.datasets.modifiers import GroundFirstFrame
from eden.options.robots.unitree_g1 import G1_IK_PROFILE, G1Dofs29

DEFAULT_Z = float(G1_IK_PROFILE.robot_options.default_root_pos[2])


def _ground(sequences, **options):
    return GroundFirstFrame(options=GroundFirstFrame.configure(robot_options=G1Dofs29(), **options)).apply(sequences)


@pytest.mark.parametrize(
    ("options", "sequences", "match"),
    [
        # It owns one robot: treating every key as its qpos would ground the wrong entity.
        ({}, {"g1": object(), "object": object()}, "exactly one motion-sequences key"),
        ({"safety_margin_m": -0.01}, {"g1": object()}, "safety_margin_m must be >= 0"),
        ({"chunk_size": 0}, {"g1": object()}, "chunk_size must be >= 1"),
        ({"chunk_size": -1}, {"g1": object()}, "chunk_size must be >= 1"),
    ],
)
def test_bad_options_fail_before_any_scene_is_built(options, sequences, match):
    with pytest.raises(ValueError, match=match):
        _ground(sequences, **options)


def test_only_penetrating_clips_are_lifted_and_chunking_changes_nothing(g1_clips):
    """Chunks of 2 over 5 clips (2 / 2 / 1, the last padded): padding must not leak into the per-clip shift."""
    base_zs = [DEFAULT_Z - 0.2, DEFAULT_Z + 0.3, DEFAULT_Z, DEFAULT_Z - 0.1, DEFAULT_Z + 0.1]
    whole, chunked = g1_clips(base_zs, num_links=2), g1_clips(base_zs, num_links=2)
    untouched = {
        name: getattr(whole, name).concatenated_tensor.clone()
        for name in ("base_lin_vel_w", "base_ang_vel_w", "dofs_vel", "links_pos_b")
    }
    z_before = whole.base_pos_w.concatenated_tensor[:, 2].clone()

    _ground({"g1": whole})
    _ground({"g1": chunked}, chunk_size=2)

    starts = whole.base_pos_w.batch_starts.tolist()
    dz = [float(whole.base_pos_w.concatenated_tensor[s, 2] - z_before[s]) for s in starts]
    # The collision AABB includes geom extents beyond the kinematic feet, hence the loose bounds.
    assert 0.15 < dz[0] < 0.30, dz
    assert abs(dz[1]) < 1e-6, dz  # above the floor: clamped to no shift
    assert abs(dz[2]) < 0.05, dz  # at the default pose, the feet are about on the floor
    for name, before in untouched.items():
        assert torch.equal(getattr(whole, name).concatenated_tensor, before), name  # a pure +z translation
    torch.testing.assert_close(whole.links_pos_w.concatenated_tensor[starts[0], :, 2], torch.full((2,), dz[0]))

    torch.testing.assert_close(
        chunked.base_pos_w.concatenated_tensor, whole.base_pos_w.concatenated_tensor, atol=1e-5, rtol=0
    )
    torch.testing.assert_close(
        chunked.links_pos_w.concatenated_tensor, whole.links_pos_w.concatenated_tensor, atol=1e-5, rtol=0
    )


def test_a_clip_without_links_is_still_lifted(g1_clips):
    clips = g1_clips([DEFAULT_Z - 0.2], num_links=0)
    z_before = clips.base_pos_w.concatenated_tensor[:, 2].clone()

    _ground({"g1": clips})

    assert (clips.base_pos_w.concatenated_tensor[:, 2] - z_before > 0.15).all()
    assert clips.links_pos_w.concatenated_tensor.shape == (8, 0, 3)
