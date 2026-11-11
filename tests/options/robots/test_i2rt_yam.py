"""YAM's gains: Raibert defaults on the arm, a 5x stiffer gripper, and an action scale without it."""

from __future__ import annotations

import pytest

from eden.options.robots.i2rt_yam import YAM_ACTION_SCALE, Yam

_DM4340_KP = 126.33094  # 3.2e-2 armature at 10 Hz
_DM4310_KP = 7.1061152  # 1.8e-3 armature at 10 Hz


def test_gripper_runs_at_five_times_the_heuristic_stiffness():
    kp = Yam().default_dofs_kp
    assert kp["joint1"] == pytest.approx(_DM4340_KP, rel=1e-7)
    assert kp["joint4"] == pytest.approx(_DM4310_KP, rel=1e-7)
    assert kp["left_finger"] == kp["right_finger"] == pytest.approx(5 * _DM4310_KP, rel=1e-7)


def test_action_scale_leaves_out_the_fingers():
    """The scale is derived from the unscaled heuristic kp, so a finger entry would be 5x too large."""
    assert set(YAM_ACTION_SCALE) == {f"joint{i}" for i in range(1, 7)}
    assert YAM_ACTION_SCALE["joint1"] == pytest.approx(0.25 * 27.0 / _DM4340_KP, rel=1e-7)
