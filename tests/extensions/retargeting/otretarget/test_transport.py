"""Tests for ``eden.extensions.retargeting.otretarget.transport``: entropic OT correspondences."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from eden.extensions.retargeting.otretarget.transport import (
    group_correspondences,
    sinkhorn,
    transport_correspondence,
)


def test_sinkhorn_plan_has_the_requested_marginals():
    rng = np.random.default_rng(0)
    cost = rng.random((40, 30))
    a = rng.random(40) + 0.1
    b = rng.random(30) + 0.1
    a, b = a / a.sum(), b / b.sum()
    plan = sinkhorn(cost, a, b, epsilon=0.05)
    np.testing.assert_allclose(plan.sum(axis=1), a, atol=1e-8)
    np.testing.assert_allclose(plan.sum(axis=0), b, atol=1e-8)


def test_correspondence_ignores_scale_and_offset_and_recovers_a_permutation():
    # A cloud matched to a shuffled, scaled and shifted copy of itself: the normalised clouds coincide,
    # so every target point must receive its own source point.
    rng = np.random.default_rng(1)
    source = rng.normal(size=(120, 3)) * np.array([0.3, 0.1, 0.05])
    perm = rng.permutation(len(source))
    target = source[perm] * 0.6 + np.array([2.0, -1.0, 0.5])
    src = transport_correspondence(source, target, epsilon=0.01)
    np.testing.assert_array_equal(src, perm)


def test_transport_covers_more_source_points_than_nearest_neighbour():
    # A long thin source against a stubby target: nearest-neighbour matching leaves the source's ends
    # unmatched, the transport plan's marginals do not (the paper's coverage argument, Sec. III-B).
    rng = np.random.default_rng(2)
    source = rng.normal(size=(300, 3)) * np.array([1.0, 0.2, 0.2])
    target = rng.normal(size=(300, 3)) * np.array([0.5, 0.3, 0.3])
    src = transport_correspondence(source, target, epsilon=0.1)
    from scipy.spatial import cKDTree

    from eden.extensions.retargeting.otretarget.transport import normalize_cloud

    nn = cKDTree(normalize_cloud(source)).query(normalize_cloud(target))[1]
    assert len(np.unique(src)) > 1.3 * len(np.unique(nn))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU to compare against")
def test_sinkhorn_runs_where_it_is_told_and_agrees_across_devices():
    # The device is the caller's: picked here whenever a GPU existed, it ignored `--cpu` and ran a shared GPU out of
    # memory during setup.
    rng = np.random.default_rng(1)
    cost, a, b = rng.random((50, 40)), np.full(50, 1 / 50), np.full(40, 1 / 40)
    np.testing.assert_allclose(sinkhorn(cost, a, b, 0.05), sinkhorn(cost, a, b, 0.05, device="cuda"), atol=1e-12)


def test_group_correspondences_match_within_groups_only():
    rng = np.random.default_rng(3)
    src_pts = np.concatenate([rng.normal(size=(50, 3)), rng.normal(size=(50, 3)) + 5.0])
    tgt_pts = np.concatenate([rng.normal(size=(40, 3)), rng.normal(size=(40, 3)) + 5.0, rng.normal(size=(5, 3))])
    src_labels = ["arm"] * 50 + ["leg"] * 50
    tgt_labels = ["a_link"] * 40 + ["l_link"] * 40 + ["extra"] * 5
    src, group = group_correspondences(
        src_pts, src_labels, tgt_pts, tgt_labels, [(("arm",), ("a_link",)), (("leg",), ("l_link",))]
    )
    assert (src[:40] < 50).all() and (src[40:80] >= 50).all()
    np.testing.assert_array_equal(group, [0] * 40 + [1] * 40 + [-1] * 5)
    np.testing.assert_array_equal(src[80:], -1)


@pytest.mark.parametrize(
    ("groups", "match"),
    [
        ([(("arm",), ("a_link",)), (("leg",), ("a_link",))], "a_link"),
        ([(("hand",), ("a_link",))], r"0 source.*Without samples: source parts \['hand'\], target parts \[\]"),
        ([(("arm",), ("b_link",))], r"target parts \['b_link'\].*sampled target parts: \['a_link'\]"),
    ],
)
def test_group_correspondences_refuses_bad_groups(groups, match):
    pts = np.random.default_rng(4).normal(size=(10, 3))
    with pytest.raises(ValueError, match=match):
        group_correspondences(pts, ["arm"] * 10, pts, ["a_link"] * 10, groups)


def test_normal_term_matches_like_facing_surfaces_of_shapes_with_other_proportions():
    # A round phalanx (the human's) against a flatter, longer box (a robot's): after normalisation position
    # alone sends ~7% of the box's points to a face of the cylinder that looks the other way; with the
    # normal term every point takes a like-facing source.
    import trimesh

    human = trimesh.creation.cylinder(radius=0.01, height=0.03, sections=32)
    robot = trimesh.creation.box((0.024, 0.014, 0.04))
    source, source_faces = trimesh.sample.sample_surface_even(human, 400, seed=1)
    target, target_faces = trimesh.sample.sample_surface_even(robot, 300, seed=2)
    source_n, target_n = human.face_normals[source_faces], robot.face_normals[target_faces]

    def like_facing(weight):
        src = transport_correspondence(source, target, 0.1, source_n, target_n, weight)
        return float(((source_n[src] * target_n).sum(1) > 0).mean())

    assert like_facing(0.0) < 0.95
    assert like_facing(0.5) > 0.99


def test_normal_term_needs_both_normals():
    pts = np.random.default_rng(6).normal(size=(10, 3))
    with pytest.raises(ValueError, match="normal_weight"):
        transport_correspondence(pts, pts, 0.1, None, pts, normal_weight=0.5)


def test_area_weights_keep_a_densely_sampled_part_from_taking_more_than_its_share():
    # One target half sampled 4x denser than the other, against a uniform source: with uniform marginals the
    # dense half takes ~80% of the source and the sparse half is matched to its far end only; weighting each
    # sample by the area it stands for restores the even split (sparse half <-> the source's whole right half).
    rng = np.random.default_rng(7)
    source = rng.uniform(-1.0, 1.0, (400, 3)) * np.array([1.0, 0.1, 0.1])
    left = rng.uniform(-1.0, 0.0, (400, 1))
    right = rng.uniform(0.0, 1.0, (100, 1))
    target = np.column_stack([np.concatenate([left, right]), rng.uniform(-0.1, 0.1, (500, 2))])
    area = np.concatenate([np.full(400, 1.0 / 400), np.full(100, 1.0 / 100)])

    def sparse_half_source_x(weights):
        src = transport_correspondence(source, target, 0.05, target_weights=weights)
        return float(source[src[400:], 0].mean())

    assert sparse_half_source_x(None) > 0.7
    assert abs(sparse_half_source_x(area) - 0.5) < 0.1
