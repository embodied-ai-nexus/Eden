"""Keyframe plans and their evaluation: spline shape at stop / pass-through keyframes, retiming, compaction, wobble.

Pure torch on hand-computed values — no scene.
"""

from __future__ import annotations

# isort: off
import eden  # noqa: F401 - isolate the DDP rank before importing Torch
# isort: on

import pytest
import torch

from eden.extensions.scripted_demos.trajectory import Keyframes, KeyframeTrajectory, Wobble, smoothstep5

IDENTITY = (1.0, 0.0, 0.0, 0.0)


def _line(times, xs, *, stop=None, valid=None, grasp=None) -> Keyframes:
    """One plan moving along x through ``xs`` at ``times``."""
    k = len(times)
    pos = torch.zeros(1, k, 3)
    pos[0, :, 0] = torch.tensor(xs, dtype=torch.float32)
    return Keyframes(
        times=torch.tensor([times], dtype=torch.float32),
        pos=pos,
        quat=torch.tensor([IDENTITY] * k).unsqueeze(0),
        grasp=torch.tensor([grasp or [0.0] * k], dtype=torch.float32),
        stop=torch.tensor([stop or [False] * k]),
        valid=torch.tensor([valid or [True] * k]),
    )


def _x_at(trajectory: KeyframeTrajectory, t: float) -> float:
    pos, _, _ = trajectory.evaluate(torch.tensor([t]))
    return float(pos[0, 0])


def _assigned(keyframes: Keyframes) -> KeyframeTrajectory:
    trajectory = KeyframeTrajectory(num_envs=1, num_keyframes=keyframes.times.shape[1], device=torch.device("cpu"))
    trajectory.assign(torch.tensor([0]), keyframes)
    return trajectory


def test_smoothstep5_rests_at_both_ends_and_is_symmetric():
    u = torch.tensor([-1.0, 0.0, 0.5, 1.0, 2.0])
    torch.testing.assert_close(smoothstep5(u), torch.tensor([0.0, 0.0, 0.5, 1.0, 1.0]))


def test_a_segment_between_two_rests_starts_and_ends_at_rest():
    trajectory = _assigned(_line([0.0, 2.0], [0.0, 1.0]))
    assert _x_at(trajectory, 0.0) == pytest.approx(0.0)
    assert _x_at(trajectory, 1.0) == pytest.approx(0.5)
    assert _x_at(trajectory, 2.0) == pytest.approx(1.0)
    # Zero velocity at both keyframes: the first 10 ms cover far less than a constant-speed 5 mm.
    assert _x_at(trajectory, 0.01) < 1e-3
    assert 1.0 - _x_at(trajectory, 1.99) < 1e-3
    # Past the end it holds the last keyframe.
    assert _x_at(trajectory, 5.0) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("middle_stops", "x_at_half"),
    [
        # Hermite with m0 = 0 and m1 = central difference 1 m/s: h01 + h11 = 0.5 - 0.125 at u = 0.5.
        (False, 0.375),
        (True, 0.5),
    ],
)
def test_a_middle_keyframe_is_passed_through_unless_it_is_a_stop(middle_stops, x_at_half):
    trajectory = _assigned(_line([0.0, 1.0, 2.0], [0.0, 1.0, 2.0], stop=[True, middle_stops, True]))
    assert _x_at(trajectory, 0.5) == pytest.approx(x_at_half)
    assert _x_at(trajectory, 1.0) == pytest.approx(1.0)


def test_unused_keyframes_are_dropped_from_the_spline():
    with_unused = _assigned(_line([0.0, 1.0, 1.0, 2.0], [0.0, 1.0, 5.0, 2.0], valid=[True, True, False, True]))
    without = _assigned(_line([0.0, 1.0, 2.0], [0.0, 1.0, 2.0]))
    for t in (0.25, 0.5, 1.0, 1.5, 2.0):
        assert _x_at(with_unused, t) == pytest.approx(_x_at(without, t), abs=1e-6), t
    assert float(with_unused.duration[0]) == pytest.approx(2.0)


def test_filled_keyframes_inherit_the_last_valid_slot():
    filled = _line([0.0, 1.0, 9.0, 2.0], [0.0, 1.0, 5.0, 2.0], valid=[True, True, False, True]).filled()
    assert filled.times[0].tolist() == [0.0, 1.0, 1.0, 2.0]
    assert filled.pos[0, :, 0].tolist() == [0.0, 1.0, 1.0, 2.0]


def test_retiming_stretches_only_segments_too_fast_for_the_peak_speed():
    # 0.3 m in 1 s at a 0.15 m/s peak needs 1.875 * 0.3 / 0.15 = 3.75 s; 0.03 m in 1 s is slow enough.
    retimed = _line([0.0, 1.0, 2.0], [0.0, 0.3, 0.33]).retimed(peak_speed=torch.tensor([0.15]))
    torch.testing.assert_close(retimed.times, torch.tensor([[0.0, 3.75, 4.75]]))


def test_the_gripper_command_eases_between_keyframes():
    trajectory = _assigned(_line([0.0, 1.0], [0.0, 0.0], grasp=[0.0, 1.0]))
    _, _, grasp = trajectory.evaluate(torch.tensor([0.5]))
    assert float(grasp[0]) == pytest.approx(0.5)


def test_assign_rejects_a_plan_whose_first_keyframe_is_unused():
    trajectory = KeyframeTrajectory(num_envs=1, num_keyframes=2, device=torch.device("cpu"))
    with pytest.raises(ValueError, match="first keyframe"):
        trajectory.assign(torch.tensor([0]), _line([0.0, 1.0], [0.0, 1.0], valid=[False, True]))


def test_wobble_bulges_inside_its_window_only():
    wobble = Wobble(num_envs=1, num_windows=1, device=torch.device("cpu"))
    curves = torch.tensor([[[[0.02, 0.0], [0.04, 0.0]]]])  # c1 = (0.02, 0), c2 = (0.04, 0)
    wobble.assign(torch.tensor([0]), torch.tensor([[[1.0, 3.0]]]), curves)
    # u = 0.5: 16 u^2 (1-u)^2 ((1-u) c1 + u c2) = 0.5 (c1 + c2).
    torch.testing.assert_close(wobble.offset(torch.tensor([2.0])), torch.tensor([[0.03, 0.0]]))
    for t in (0.5, 1.0, 3.0, 4.0):
        torch.testing.assert_close(wobble.offset(torch.tensor([t])), torch.zeros(1, 2))
