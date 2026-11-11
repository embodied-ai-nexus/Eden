"""Tests for the art suite's plan timeline: careful (slow) in strokes and stops, quick between, home at both ends."""

import numpy as np

from eden.tasks.art.plans import STOP, TRAVEL, Stroke, segment, tip_path
from eden.tasks.art.scripted import plan_timeline


def test_a_stop_is_timed_as_carefully_as_a_stroke_and_the_travel_quickly():
    stroke = Stroke("dash", ((0.0, 0.0, -0.003), (0.01, 0.0, 0.002), (0.02, 0.0, -0.003)))
    dip = segment((-0.1, 0.0, -0.003), (-0.1, 0.0, 0.004), 0.00025)
    path, owner = tip_path([stroke], stops=[dip])
    goal, careful, step_owner = plan_timeline(
        path, owner, draw_speed=0.02, travel_speed=0.1, dt=0.05, home=(0.0, 0.0, -0.03)
    )
    assert np.allclose(goal[0], (0.0, 0.0, -0.03)) and np.allclose(goal[-1], (0.0, 0.0, -0.03))
    steps = np.linalg.norm(np.diff(goal, axis=0), axis=1)
    assert steps[careful[1:] & careful[:-1]].max() <= 0.02 * 0.05 + 1e-9
    assert steps.max() > 0.02 * 0.05, "travel goes faster than drawing"
    # The dip comes first, then the stroke; careful steps are exactly those in one or the other.
    runs = [int(o) for o in step_owner[np.r_[0, np.flatnonzero(np.diff(step_owner)) + 1]]]
    assert runs == [TRAVEL, STOP, TRAVEL, 0, TRAVEL]
    assert np.array_equal(careful, step_owner != TRAVEL)
