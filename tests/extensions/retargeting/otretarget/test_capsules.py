"""Tests for ``eden.extensions.retargeting.otretarget.capsules``."""

from __future__ import annotations

import numpy as np
import trimesh

from eden.extensions.retargeting.otretarget.capsules import build_link_capsules, fit_capsule


def test_fit_capsule_recovers_a_cylinder_axis_and_radius():
    cyl = trimesh.creation.cylinder(radius=0.05, height=0.4, sections=64)
    pts, _ = trimesh.sample.sample_surface(cyl, 20000, seed=0)
    a, b, r = fit_capsule(np.asarray(pts), radius_quantile=0.95)
    axis = (b - a) / np.linalg.norm(b - a)
    assert abs(abs(axis[2]) - 1.0) < 1e-3
    np.testing.assert_allclose(r, 0.05, atol=2e-3)
    np.testing.assert_allclose(np.linalg.norm(b - a), 0.4 - 2 * r, atol=5e-3)


def test_touching_and_adjacent_links_are_not_checked():
    # Three links in a chain plus a far one: the chain's adjacent pairs and a pair touching at rest are
    # excluded, the far pair kept.
    rod = np.stack([np.linspace(0, 0.2, 50), np.zeros(50), np.zeros(50)], 1)
    pts = {0: rod, 1: rod, 2: rod, 3: rod}
    parent = np.array([-1, 0, 1, -1])
    pos = np.array([[0.0, 0, 0], [0.25, 0, 0], [0.5, 0, 0], [0.0, 1.0, 0]])
    caps = build_link_capsules(pts, parent, [(pos, np.tile(np.eye(3), (4, 1, 1)))], radius_quantile=1.0)
    kept = {tuple(sorted(caps.link[p])) for p in caps.pairs}
    assert (0, 1) not in kept and (1, 2) not in kept and (0, 2) not in kept
    assert (0, 3) in kept and (2, 3) in kept


def test_excluded_name_pairs_are_not_checked_in_either_order():
    rod = np.stack([np.linspace(0, 0.2, 50), np.zeros(50), np.zeros(50)], 1)
    parent = np.array([-1, -1, -1])
    pos = np.array([[0.0, 0, 0], [0.0, 0.5, 0], [0.0, 1.0, 0]])
    names = ["r_index_finger_distal", "r_middle_finger_distal", "r_thumb_distal"]
    caps = build_link_capsules(
        {0: rod, 1: rod, 2: rod},
        parent,
        [(pos, np.tile(np.eye(3), (3, 1, 1)))],
        radius_quantile=1.0,
        link_names=names,
        exclude=[(r"r_(index|middle)_finger_.*", r"r_(index|middle)_finger_.*")],
    )
    kept = {tuple(sorted(caps.link[p])) for p in caps.pairs}
    assert kept == {(0, 2), (1, 2)}  # finger-finger excluded, thumb-finger kept
