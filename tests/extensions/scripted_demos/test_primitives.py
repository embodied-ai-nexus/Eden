"""The pick-and-place keyframe sampler: plan structure, speed limit, headings the wrist can reach, retries.

Pure torch over a random batch — every assertion is an invariant that must hold for every sample.
"""

from __future__ import annotations

# isort: off
import eden  # noqa: F401 - isolate the DDP rank before importing Torch
# isort: on

import math

import pytest
import torch

from eden.extensions.scripted_demos.primitives import (
    PickPlaceParams,
    PickPlaceRequest,
    Slot,
    effector_rot,
    fit_yaw,
    sample_pick_place,
)
from eden.extensions.scripted_demos.trajectory import MIN_JERK_PEAK_TO_MEAN
from eden.utils.geom import transform_by_quat, yaw_from_quat

BATCH = 256
WORKSPACE = (torch.tensor([-0.4, -0.4, 0.81]), torch.tensor([0.4, 0.4, 1.3]))


def _request(*, release=True, retries=0, yaw_window=(-math.inf, math.inf), goal_turn=None, seed=0) -> PickPlaceRequest:
    g = torch.Generator().manual_seed(seed)

    def u(lo, hi, *shape):
        return lo + (hi - lo) * torch.rand(BATCH, *shape, generator=g)

    object_yaw = u(-math.pi, math.pi)
    return PickPlaceRequest(
        tool_pos=torch.stack([u(-0.1, 0.1), u(-0.1, 0.1), u(1.0, 1.1)], dim=-1),
        tool_rot=effector_rot(u(-0.5, 0.5)),
        object_pos=torch.stack([u(-0.08, 0.08), u(-0.08, 0.08), torch.full((BATCH,), 0.82)], dim=-1),
        object_yaw=object_yaw,
        goal_pos=torch.stack([u(-0.1, 0.1), u(-0.1, 0.1), torch.full((BATCH,), 0.865)], dim=-1),
        goal_yaw=u(-math.pi, math.pi) if goal_turn is None else object_yaw + goal_turn,
        final_pos=torch.stack([u(-0.2, 0.0), u(-0.15, 0.15), u(0.92, 1.1)], dim=-1),
        final_yaw=u(-1.5, 1.5),
        release=torch.full((BATCH,), release),
        segment_dt=u(0.7, 1.5),
        num_retries=torch.full((BATCH,), retries, dtype=torch.long),
        yaw_window=torch.tensor([yaw_window]).expand(BATCH, 2),
    )


def _plan(request: PickPlaceRequest, params: PickPlaceParams | None = None, seed: int = 1):
    return sample_pick_place(request, generator=torch.Generator().manual_seed(seed), workspace=WORKSPACE, params=params)


def test_a_plan_starts_at_the_hand_and_grasps_once_before_releasing_at_the_place():
    request = _request()
    kf = _plan(request).keyframes
    torch.testing.assert_close(kf.pos[:, Slot.INITIAL], request.tool_pos)
    assert bool((kf.times.diff(dim=1) >= 0).all())
    for row in range(BATCH):
        used = kf.grasp[row][kf.valid[row]]
        closes = (used.diff() > 0).sum().item()
        opens = (used.diff() < 0).sum().item()
        assert (closes, opens) == (1, 1), f"row {row}: gripper {used.tolist()}"
    # Open at release, closed while carrying.
    assert bool((kf.grasp[:, Slot.CLEARANCE] == 1).all())
    assert bool((kf.grasp[:, Slot.PLACE_END] == 0).all())
    torch.testing.assert_close(kf.pos[:, Slot.FINAL], request.final_pos)


def test_a_lift_ends_holding_the_object_at_the_goal():
    request = _request(release=False)
    kf = _plan(request).keyframes
    for slot in (Slot.PLACE, Slot.PLACE_END, Slot.POSTPLACE, Slot.FINAL):
        assert not bool(kf.valid[:, slot].any()), slot.name
    torch.testing.assert_close(kf.pos[:, Slot.PLACE_START], request.goal_pos)
    assert bool((kf.grasp[:, Slot.PLACE_START] == 1).all())


def test_no_segment_is_faster_than_the_peak_speed():
    request = _request()
    params = PickPlaceParams(peak_speed=0.12)
    kf = _plan(request, params).keyframes
    dt = kf.times.diff(dim=1)
    dist = (kf.pos[:, 1:] - kf.pos[:, :-1]).norm(dim=-1)
    moving = dist > 1e-6
    peak = MIN_JERK_PEAK_TO_MEAN * dist / dt.clamp(min=1e-9)
    limit = (params.peak_speed / request.segment_dt).unsqueeze(1).expand_as(peak)
    assert bool((peak[moving] <= limit[moving] * (1 + 1e-4)).all())


def test_the_object_turns_by_the_nearest_equivalent_heading():
    """A cube asked to turn 3/2 pi is already there a quarter turn round: the hand must not turn it at all."""
    kf = _plan(_request(goal_turn=1.5 * math.pi)).keyframes
    pick_yaw = yaw_from_quat(kf.quat[:, Slot.PICK_START])
    place_yaw = yaw_from_quat(kf.quat[:, Slot.PLACE_START])
    turn = torch.remainder(place_yaw - pick_yaw + math.pi, 2 * math.pi) - math.pi
    assert float(turn.abs().max()) < 1e-4


def test_every_planned_heading_stays_inside_the_wrist_window():
    window = (-1.6, 1.8)  # wider than a half turn (the gripper period), as a real wrist's is (~5 rad on a Panda)
    kf = _plan(_request(yaw_window=window)).keyframes
    for slot in (Slot.PICK, Slot.PLACE_START, Slot.FINAL):
        yaw = yaw_from_quat(kf.quat[:, slot])
        assert bool(((yaw >= window[0] - 1e-4) & (yaw <= window[1] + 1e-4)).all()), slot.name


def test_retries_shrink_every_random_offset():
    """After many failed picks the hand comes straight down on the object, untilted."""
    request = _request(retries=40)
    kf = _plan(request).keyframes
    hover = kf.pos[:, Slot.PICK]
    torch.testing.assert_close(hover[:, :2], request.object_pos[:, :2], atol=1e-3, rtol=0)
    up = torch.tensor([0.0, 0.0, 1.0])
    axis = transform_by_quat(up.expand(BATCH, 3), kf.quat[:, Slot.PICK_START])
    assert float((1 - axis[:, 2]).max()) < 1e-4


@pytest.mark.parametrize(
    ("yaw", "window", "period", "expected"),
    [
        (0.3, (-1.0, 1.0), math.pi / 2, 0.3),  # already inside: unchanged
        (2.0, (-1.0, 1.0), math.pi / 2, 2.0 - math.pi / 2),  # one face back
        (-2.9, (-1.0, 1.0), math.pi, -2.9 + math.pi),  # a gripper half turn
        (0.3, (1.5, 1.6), math.pi / 2, 0.3 + math.pi / 2),  # window narrower than the period: nearest candidate
    ],
)
def test_fit_yaw_moves_a_heading_into_the_window_by_whole_periods(yaw, window, period, expected):
    out = fit_yaw(torch.tensor([yaw]), torch.tensor([window]), period)
    assert float(out[0]) == pytest.approx(expected)
