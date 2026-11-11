"""The put-on routine's geometry: its goals, its grasps and its aim, on footprints made up for each test.

Its real runs are SIMPLER's task tests (``tests/tasks/simpler/test_scripted_demos.py``); the Polaris
demonstrators run only under ``eden datagen`` (``docs/guides/scripted-demos.md``), their batched scenes
being too heavy to build in a test.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from eden.extensions.scripted_demos.put_on_routine import (
    CONTAINER_WALL,
    RIM_INSET,
    Goal,
    Hand,
    _aim,
    _grasp,
    _over,
    _placed,
    _rim_grasp,
    body_centre,
    put_on_routine,
)
from eden.extensions.scripted_demos.tools import Footprint

#: The WidowX's (SIMPLER's) fingers.
HAND = Hand(finger_reach=(0.027, 0.037), clear_view_pos=(0.15, 0.0, 0.18))
#: A box goal, basket-sized.
BOX = ((0.16, 0.01, 0.07), (0.35, 0.13, 0.13))


def _footprint(x, y, *, yaw=0.0, half=(0.05, 0.05), bottom=0.0, height=0.02, centroid=None) -> Footprint:
    return Footprint(
        center=np.array([x, y, bottom + height / 2]),
        yaw=yaw,
        half_length=half[0],
        half_width=half[1],
        top_z=bottom + height,
        bottom_z=bottom,
        top_center=np.array([x, y, bottom + height]),
        num_points=100,
        centroid=None if centroid is None else np.asarray(centroid),
    )


class _Tools:
    """The routine's ``tools`` as far as the grasps read them: the WidowX's finger geometry and a random stream."""

    tip_reach = 0.0187
    finger_axis_yaw = math.pi

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)
        self.notes = []

    def note(self, text):
        self.notes.append(text)


# --- the goal ---------------------------------------------------------------------------------------


def test_a_goal_takes_exactly_one_of_a_target_and_a_region():
    with pytest.raises(ValueError, match="exactly one of target and region"):
        Goal()
    with pytest.raises(ValueError, match="exactly one of target and region"):
        Goal(target="target", region=BOX)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"region": BOX, "container": True}, "Name the container as target="),
        ({"region": BOX, "target_body": True}, "Name the container as target="),
        ({"target": "pan", "max_reach": 0.7}, "container=False"),
        ({"target": "tray", "avoid": ("tomato",)}, "container=False"),
        ({"target": "tray", "container": True, "max_reach": 0.0}, "must be positive"),
        ({"target": "pan", "scrub": -0.01}, "must be >= 0"),
    ],
)
def test_a_goal_refuses_options_it_would_ignore(kwargs, match):
    with pytest.raises(ValueError, match=match):
        Goal(**kwargs)


def test_a_rim_grasp_refuses_clutter_it_would_ignore():
    with pytest.raises(ValueError, match="grasp='rim' with clutter=\\('plate',\\)"):
        put_on_routine(
            "cup", Goal(target="board"), hand=HAND, p_mistake=0.0, camera=None, grasp="rim", clutter=("plate",)
        )


def test_the_routine_lists_every_entity_it_looks_at():
    goal = Goal(target="tray", container=True, avoid=("tomato",))
    routine = put_on_routine("cube", goal, hand=HAND, p_mistake=0.0, camera=None, clutter=("plate",))
    assert routine.entity_names == ("cube", "tray", "tomato", "plate")
    assert put_on_routine("cube", Goal(region=BOX), hand=HAND, p_mistake=0.0, camera=None).entity_names == ("cube",)


def test_an_object_is_on_a_target_when_centred_over_it_and_resting_on_it():
    goal, centre = Goal(target="target"), np.array([0.3, 0.0])
    assert _placed(_footprint(0.31, 0.0, bottom=0.02), goal, centre, top=0.02)
    assert not _placed(_footprint(0.33, 0.0, bottom=0.02), goal, centre, top=0.02)  # 3 cm off
    assert not _placed(_footprint(0.30, 0.0, bottom=0.08), goal, centre, top=0.02)  # still held above it


def test_an_object_is_in_a_box_goal_anywhere_inside_it():
    """A let-go eggplant rolls a few cm in the basket: inside is placed, not only near the middle."""
    goal = Goal(region=BOX)
    low, high = BOX
    floor = low[2]
    corner = _footprint(low[0] + 0.01, high[1] - 0.01, bottom=floor)
    assert _placed(corner, goal, np.array([0.3, 0.045]), top=floor)
    assert not _placed(_footprint(low[0] - 0.01, 0.045, bottom=floor), goal, np.array([0.3, 0.045]), top=floor)


# --- the grasp --------------------------------------------------------------------------------------


def test_a_point_is_over_a_footprint_widened_by_the_finger_clearance():
    cloth = _footprint(0.3, 0.0, yaw=math.pi / 4)
    assert _over(cloth, np.array([0.3, 0.0]))
    corner_out = np.array([0.3 + 0.1, 0.0])  # 10 cm along x, past the turned square's corner
    assert not _over(cloth, corner_out)


def test_a_grasp_closes_across_the_short_side_without_an_obstacle():
    spoon = _footprint(0.3, 0.1, half=(0.07, 0.018), height=0.026)
    yaw, grasp_xy, grasp_z = _grasp(_Tools(), spoon, [], HAND)
    # Fingers close along the effector heading + finger_axis_yaw: across a spoon lying along x.
    assert math.cos(yaw + math.pi) == pytest.approx(0.0, abs=1e-6)
    assert abs(grasp_xy[1] - 0.1) < 1e-9
    # Floor-limited: the fingertips stop just above the table.
    assert grasp_z == pytest.approx(0.0187 + 0.002)


def test_a_grasp_slides_along_the_object_until_no_finger_comes_down_on_an_obstacle():
    """A spoon lying along the folded cloth's edge: a centred grasp puts a finger on the cloth."""
    cloth = _footprint(0.31, 0.06, half=(0.052, 0.052), height=0.026)
    spoon = _footprint(0.27, 0.12, half=(0.07, 0.018), height=0.026)
    for seed in range(10):
        _, grasp_xy, _ = _grasp(_Tools(seed), spoon, [cloth], HAND)
        for side in (-1, 1):
            for reach in HAND.finger_reach:
                assert not _over(cloth, grasp_xy + np.array([0.0, side * reach]))
        assert grasp_xy[0] < 0.27  # away from the cloth, along the spoon


def test_an_obstacle_lower_than_the_fingertips_is_no_obstacle():
    plate = _footprint(0.31, 0.06, half=(0.055, 0.055), height=0.007)
    carrot = _footprint(0.27, 0.12, half=(0.06, 0.017), height=0.047)
    tools = _Tools()
    _grasp(tools, carrot, [plate], HAND)
    assert tools.notes == []


def test_a_rim_grasp_straddles_the_wall_on_the_side_facing_the_base():
    """A 13 cm roll of tape (too wide for any pinch) at (0.4, 0.3): gripped by its wall, one finger in the hole."""
    tape = _footprint(0.4, 0.3, half=(0.067, 0.066), height=0.046)
    yaw, grasp_xy, grasp_z = _rim_grasp(_Tools(), tape)
    radial = grasp_xy - tape.center[:2]
    assert np.linalg.norm(radial) == pytest.approx(0.066 - RIM_INSET)
    assert np.linalg.norm(grasp_xy) < np.linalg.norm(tape.center[:2])  # the near side
    closing = np.array([math.cos(yaw + math.pi), math.sin(yaw + math.pi)])  # finger_axis_yaw = pi
    assert abs(closing @ radial) == pytest.approx(np.linalg.norm(radial))  # the fingers close radially
    assert grasp_z == pytest.approx(0.046 - 0.02)


def test_the_body_of_a_handled_object_is_the_end_its_points_crowd():
    """A pan: a 21 cm bowl at x 0.45-0.66 and a handle out to x 0.80; its points crowd the bowl."""
    pan = _footprint(0.625, 0.2, half=(0.175, 0.105), centroid=(0.53, 0.2))
    centre, radius = body_centre(pan)
    np.testing.assert_allclose(centre, [0.555, 0.2])
    assert radius == 0.105
    round_bowl = _footprint(0.4, 0.0, half=(0.09, 0.088), centroid=(0.38, 0.0))
    np.testing.assert_allclose(body_centre(round_bowl)[0], [0.4, 0.0])


def test_an_object_is_in_a_container_anywhere_its_centre_clears_the_walls():
    tray = _footprint(0.87, -0.2, half=(0.2, 0.13), height=0.09)
    goal = Goal(target="tray", container=True)
    centre = tray.center[:2]
    near_wall = _footprint(0.87 + 0.2 - 0.04, -0.2, half=(0.1, 0.03), bottom=0.01)
    assert _placed(near_wall, goal, centre, top=0.09, target=tray)
    over_wall = _footprint(0.87 + 0.2 - 0.02, -0.2, half=(0.1, 0.03), bottom=0.01)
    assert not _placed(over_wall, goal, centre, top=0.09, target=tray)


def test_the_aim_is_pulled_in_to_the_reach_while_the_object_stays_over_the_container():
    tray = _footprint(0.87, -0.2, half=(0.2, 0.13), height=0.09)
    scissor = _footprint(0.47, 0.0, half=(0.115, 0.05))
    goal = Goal(target="tray", container=True, max_reach=0.6)
    aim = _aim(goal, tray, scissor, tray.center[:2])
    # Not to 0.6 m: the scissor's half length and the wall keep it 13.5 cm inside the tray's near edge.
    assert np.linalg.norm(aim) > 0.6
    assert aim[0] == pytest.approx(0.87 - 0.2 + 0.115 + CONTAINER_WALL, abs=0.01)
    np.testing.assert_array_equal(
        _aim(Goal(target="tray", container=True), tray, scissor, tray.center[:2]), tray.center[:2]
    )


def test_the_aim_keeps_clear_of_what_already_lies_in_the_container():
    """A tomato in the tray's -x half: the cube goes to a far corner of the tray, inset by its half size and the wall."""
    tray = _footprint(0.5, 0.0, half=(0.15, 0.1), height=0.03)
    cube = _footprint(0.3, 0.2, half=(0.025, 0.025))
    tomato = _footprint(0.42, 0.0, half=(0.03, 0.03))
    aim = _aim(Goal(target="tray", container=True, avoid=("tomato",)), tray, cube, tray.center[:2], [tomato])
    inset = 0.025 + CONTAINER_WALL
    assert aim[0] == pytest.approx(0.5 + 0.15 - inset)
    assert abs(aim[1]) == pytest.approx(0.1 - inset)


def test_a_square_object_is_closed_across_its_other_side_when_one_side_has_clutter():
    """A cube with a tilted plate along its +y side: closing across y puts a finger on the plate; across x does not."""
    cube = _footprint(0.42, -0.10, half=(0.03, 0.03), height=0.06)
    plate = _footprint(0.42, -0.03, half=(0.06, 0.03), height=0.05)
    tools = _Tools()
    yaw, grasp_xy, _ = _grasp(tools, cube, [plate], HAND)
    closing = np.array([math.cos(yaw + math.pi), math.sin(yaw + math.pi)])  # finger_axis_yaw = pi
    assert abs(closing[0]) == pytest.approx(1.0, abs=1e-6)  # the fingers close along x
    assert tools.notes == []
