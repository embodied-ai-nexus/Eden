"""Tests for the art suite's plan format: the spline through a stroke's key points, and the tip's path through a plan."""

import numpy as np
import pytest

from eden.tasks.art.plans import STOP, TRAVEL, TRAVEL_HEIGHT, Stroke, catmull_rom, segment, tip_path


def _dash(u: float) -> Stroke:
    return Stroke("dash", ((u, 0.0, -0.003), (u + 0.01, 0.0, 0.002), (u + 0.02, 0.0, -0.003)))


def test_the_spline_passes_through_its_key_points_at_about_the_spacing():
    keys = np.array([[0.0, 0.0, -1.0], [0.01, 0.0, 1.0], [0.02, 0.01, 1.0], [0.02, 0.02, -1.0]])
    path = catmull_rom(keys, 0.00025)
    assert np.allclose(path[0], keys[0]) and np.allclose(path[-1], keys[-1])
    for key in keys[1:-1]:
        assert np.linalg.norm(path - key, axis=1).min() < 1e-9
    steps = np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1)
    assert steps.max() <= 0.0004


def test_the_tip_travels_high_between_strokes_and_owns_every_sample():
    path, owner = tip_path([_dash(0.0), _dash(0.05)])
    assert set(np.unique(owner)) == {TRAVEL, 0, 1}
    assert path[owner == TRAVEL, 2].min() == pytest.approx(-TRAVEL_HEIGHT)
    # The strokes are their own splines, in order, joined by travel.
    assert np.flatnonzero(owner == 0).max() < np.flatnonzero(owner == TRAVEL).min()
    assert np.flatnonzero(owner == TRAVEL).max() < np.flatnonzero(owner == 1).min()
    np.testing.assert_allclose(path[owner == 1][0], _dash(0.05).points[0])


def test_a_stop_is_visited_on_the_way_to_its_stroke():
    dip = segment((-0.1, 0.0, -0.01), (-0.1, 0.0, 0.004), 0.00025)
    path, owner = tip_path([_dash(0.0), _dash(0.05)], stops=[dip, dip])
    assert (owner[0] == STOP) and np.allclose(path[0], dip[0])  # the first stop is where the path starts
    runs = np.flatnonzero(np.diff(owner) != 0) + 1
    order = [int(owner[i]) for i in np.r_[0, runs]]
    assert order == [STOP, TRAVEL, 0, TRAVEL, STOP, TRAVEL, 1]
    assert path[owner == STOP, 2].max() == pytest.approx(0.004)


def test_a_stop_per_stroke_or_none_is_required():
    with pytest.raises(ValueError, match=r"1 stops for 2 strokes; give one \(or None\) per stroke"):
        tip_path([_dash(0.0), _dash(0.05)], stops=[None])
