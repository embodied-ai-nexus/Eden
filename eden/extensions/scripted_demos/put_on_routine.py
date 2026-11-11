"""A tool routine that puts an object on or into its target (``eden datagen --task simpler_put_carrot_on_plate``, …).

Written against :class:`~eden.extensions.scripted_demos.tools.Tools`, for one robot, as the
routines of :mod:`eden.extensions.scripted_demos.routines` are: look at the object, grip it from
above, carry it over the goal and lower it on, then look again. The loop is goal-directed, so the
episode's deliberate mistakes (the object put down off the goal first, with probability
``p_mistake``) and a missed or dropped pick are recovered from by looking again.

The goal is where the object's own centre must come to rest, from :class:`Goal`: over a target
entity's footprint (the towel, the plate), into a target container (a bowl, a tray), or inside a
fixed box (the sink's basket, which is part of the fixed sink). The same routine runs privileged
(footprints off the collision meshes) or camera-only
(:meth:`~eden.extensions.scripted_demos.tools.Tools.locate`); only :func:`_look` chooses.

What differs between robots is the :class:`Hand`: where its open fingers reach and where it moves to
clear the camera's view. The SIMPLER Bridge tasks (WidowX250) and the Polaris DROID tasks (Franka +
Robotiq 2F85) each pass their own.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Generator

    from eden.extensions.scripted_demos.routines import Routine
    from eden.extensions.scripted_demos.tools import Footprint, Motion, Outcome, Tools

#: An object is looked at until it has settled: two looks ``SETTLE_STEP_S`` apart whose centres are
#: within ``SETTLE_DRIFT`` [m], for at most ``SETTLE_MAX_S``. A third of the reset eggplants roll 4-6 cm across the
#: basin over the first seconds, and one let go in the basket rolls a few cm.
SETTLE_STEP_S = 0.4
SETTLE_DRIFT = 0.003
SETTLE_MAX_S = 4.0
#: An object seen again from the hover pose this far [m] from where it was is picked from where it is now.
REAIM_DISTANCE = 0.008
#: How far [m] above the object's top the hand hovers before the descent, drawn per pick.
HOVER_ABOVE = (0.04, 0.08)
#: Fingertip clearance [m] of the empty hand over the obstacles between the object and the goal.
HOVER_CLEARANCE = 0.02
#: Clearance [m] of a carried object's bottom over the obstacles and the goal's top, drawn per carry.
CARRY_CLEARANCE = (0.03, 0.05)
#: How deep [m] below the object's top the pads close — at most half its height.
GRASP_DEPTH = 0.02
#: Gap [m] the fingertips keep above the surface the object rests on: the hand stalls on it otherwise.
FLOOR_GAP = 0.002
#: Grasp offset along the object's long axis, drawn per grasp as a fraction of how much longer than wide it is.
GRASP_SLIDE = 0.15
#: Tolerance [m] of the descent to the grasp pose. The WidowX's open fingers clear a 3.7 cm eggplant by
#: under a centimetre a side, and the default 1 cm let the hand stop 7 mm high: the pads closed above
#: the eggplant's equator and squeezed it out.
GRASP_TOLERANCE = 0.005
#: Largest grasp offset along the object's long axis, as a fraction of how much longer than wide it is,
#: when the grasp slides to keep the fingers off an obstacle.
MAX_GRASP_SLIDE = 0.8
#: Clearance [m] a finger keeps from an obstacle's footprint: half a finger's width, and some.
FINGER_CLEARANCE = 0.012
#: How long [s] the hand pauses at the grasp pose before closing, drawn per pick (the IK trails the move).
GRASP_PAUSE = (0.2, 0.5)
#: Gap [m] between the object's bottom and the goal's top when it is let go.
RELEASE_GAP = 0.01
#: An object is placed when its centre rests within this [m] of the goal centre (in xy) …
PLACED_TOLERANCE = 0.02
#: … and its bottom no higher than this [m] above the goal's top.
PLACED_HEIGHT = 0.02
#: Spread [m] of where on the goal the object is aimed: the task's own check allows 3 cm.
PLACE_JITTER = 0.008
#: Tolerance [m] of the carry and release moves: the WidowX's damped IK stops ~1 cm short of some.
PLACE_TOLERANCE = 0.015
#: Picks before a placement attempt is given up, and placement attempts before the episode is.
MAX_PICK_ATTEMPTS = 3
MAX_PLACEMENTS = 4
#: Depth [m] below a rim that a rim grasp closes at.
RIM_DEPTH = 0.02
#: How far [m] inside a hollow object's outer edge a rim grasp is centred: about half a cup's wall.
RIM_INSET = 0.004
#: An elongated footprint whose long side exceeds its short side by less than this [m] is taken as round.
ROUND_SLACK = 0.01
#: Back-and-forth strokes of a scrub, each :attr:`Goal.scrub` [m] to either side of the place.
SCRUB_STROKES = 3
#: Room [m] a container aim leaves between the object and the container's outer edge: its wall and rim.
CONTAINER_WALL = 0.02
#: How far [m] a scrub lowers the object to find what it rubs on: the goal's top is a pan's rim, its floor lower.
SCRUB_REACH = 0.06


@dataclass(frozen=True)
class Hand:
    """What the routine knows about the robot's hand that the rig does not measure.

    Attributes
    ----------
    finger_reach : tuple[float, float]
        Distances [m] from the grasp point, along the closing axis, of an open finger's inner and
        outer faces.
    clear_view_pos : tuple[float, float, float]
        Where the hand goes to clear the camera's view when the object cannot be seen.
    """

    finger_reach: tuple[float, float]
    clear_view_pos: tuple[float, float, float]


@dataclass(frozen=True)
class Goal:
    """Where the object's centre must come to rest.

    Exactly one of ``target`` and ``region`` is given.

    A target is something to put the object *on* (the object counts as placed with its centre within
    :data:`PLACED_TOLERANCE` of the target's) unless ``container``: then the object is let go above
    the target's rim and counts as placed with its centre anywhere over the target's footprint, inset
    by the object's half width — where a bowl or a tray holds it.

    Attributes
    ----------
    target : str | None
        Scene entity to put the object on: the goal is its footprint's centre and top.
    region : tuple[tuple[float, float, float], tuple[float, float, float]] | None
        Low / high corners of a fixed box the object goes into: the goal is its centre in xy, and
        its low ``z`` is the floor the object is lowered onto.
    object_yaw : float | None
        Heading [rad] the object's long axis is turned to (modulo pi) while carried; None keeps the
        heading it was picked at.
    obstacle_top : float
        Highest point [m] between the object and the goal — a rim the hand and the object pass over.
    mistake_bounds : tuple[tuple[float, float], tuple[float, float]]
        Low / high xy corners of where a deliberate mistake puts the object down instead.
    mistake_floor : float
        Height [m] of the surface inside ``mistake_bounds``.
    container : bool
        The target is a container the object is dropped into (see above).
    target_body : bool
        Aim at the round body of a target with a handle (a frying pan) instead of its footprint's
        centre, which the handle drags off it: the end of the footprint's long side its points crowd.
    max_reach : float | None
        Container goals: the aim stays within this horizontal distance [m] of the robot's base (the
        world origin) where the container allows — the far side of a large one is out of the arm's
        reach — and otherwise comes as close to it as the container allows.
    avoid : tuple[str, ...]
        Container goals: entities already in the container (a tomato in the tray) the aim keeps
        clear of.
    scrub : float
        Lower the object onto the goal and rub it back and forth this far [m] to either side,
        :data:`SCRUB_STROKES` times, before letting it go (a sponge on a pan); 0 does not.
    """

    target: str | None = None
    region: tuple[tuple[float, float, float], tuple[float, float, float]] | None = None
    object_yaw: float | None = None
    obstacle_top: float = 0.0
    mistake_bounds: tuple[tuple[float, float], tuple[float, float]] = ((0.2, -0.15), (0.35, 0.15))
    mistake_floor: float = 0.0
    container: bool = False
    target_body: bool = False
    max_reach: float | None = None
    avoid: tuple[str, ...] = ()
    scrub: float = 0.0

    def __post_init__(self):
        if (self.target is None) == (self.region is None):
            raise ValueError(
                f"A Goal takes exactly one of target and region, got target={self.target!r} and region={self.region!r}."
            )
        if self.region is not None and (self.container or self.target_body):
            raise ValueError(
                f"container and target_body describe a target entity, and this Goal has a region; got "
                f"container={self.container}, target_body={self.target_body}. Name the container as target= instead."
            )
        if not self.container and (self.max_reach is not None or self.avoid):
            raise ValueError(
                f"max_reach and avoid steer the aim inside a container, and this Goal has container=False; got "
                f"max_reach={self.max_reach}, avoid={self.avoid}. Pass container=True, or drop them."
            )
        if self.max_reach is not None and self.max_reach <= 0:
            raise ValueError(
                f"max_reach is a distance [m] from the robot's base and must be positive, got {self.max_reach}."
            )
        if self.scrub < 0:
            raise ValueError(f"scrub is a stroke length [m] and must be >= 0 (0 does not scrub), got {self.scrub}.")


def put_on_routine(
    source: str,
    goal: Goal,
    *,
    hand: Hand,
    p_mistake: float,
    camera: str | None,
    grasp: str = "pinch",
    clutter: tuple[str, ...] = (),
) -> Routine:
    """The routine that puts ``source`` at ``goal``; camera-only when ``camera`` is given.

    ``grasp`` is how the object is gripped from above: ``"pinch"`` across its short side, or ``"rim"``
    by the wall of a hollow object too wide to pinch (a cup, a roll of tape), one finger inside it.
    A pinch keeps the open fingers off the goal's target and off the ``clutter`` entities around the
    object (a tilted plate next to a cube), which the hand would come down on and stop short.
    """
    if grasp not in ("pinch", "rim"):
        raise ValueError(f"put_on_routine grasp must be 'pinch' or 'rim', got {grasp!r}.")
    if clutter and grasp == "rim":
        raise ValueError(
            f"clutter keeps a pinch's open fingers off nearby objects, and a rim grasp does not use it; got "
            f"grasp='rim' with clutter={clutter}. Drop clutter, or use grasp='pinch'."
        )

    def put_on(tools: Tools) -> Generator[Motion, Outcome, None]:
        mistake = tools.rng.random() < p_mistake
        for _ in range(MAX_PLACEMENTS):
            item = yield from _settled_look(tools, source, camera, hand)
            if item is None:
                tools.note(f"{source}: not seen, gave up")
                return
            target = _look(tools, goal.target, camera) if goal.target is not None else None
            if goal.target is not None and target is None:
                tools.note(f"{goal.target}: not seen, gave up")
                return
            centre, top = _goal_centre_and_top(goal, target)
            if _placed(item, goal, centre, top, target):
                return
            others = [other for other in (_look(tools, name, camera) for name in goal.avoid) if other is not None]
            centre = _aim(goal, target, item, centre, others)
            if mistake:
                mistake = False
                centre, top = _mistake_spot(tools, goal, item, centre), goal.mistake_floor
                tools.note(f"{source} -> mistake spot ({centre[0]:.3f}, {centre[1]:.3f})")
            else:
                centre = centre + tools.rng.uniform(-PLACE_JITTER, PLACE_JITTER, size=2)
                tools.note(f"{source} -> goal ({centre[0]:.3f}, {centre[1]:.3f})")
            if not (yield from _pick_and_place(tools, source, item, centre, top, goal, camera, hand, grasp, clutter)):
                tools.note(f"{source}: every pick missed")
        tools.note(f"gave up after {MAX_PLACEMENTS} placements")

    put_on.entity_names = (source, *((goal.target,) if goal.target is not None else ()), *goal.avoid, *clutter)
    return put_on


def _look(tools: Tools, name: str, camera: str | None) -> Footprint | None:
    """The object's footprint: off the simulator when privileged, through ``camera`` otherwise."""
    return tools.object_footprint(name) if camera is None else tools.locate(name, camera)


def _look_or_clear_view(
    tools: Tools, name: str, camera: str | None, hand: Hand
) -> Generator[Motion, Outcome, Footprint | None]:
    """The object's footprint, moving the hand out of the camera's view first if it hides the object."""
    item = _look(tools, name, camera)
    if item is None:
        yield tools.move_to(hand.clear_view_pos)
        item = _look(tools, name, camera)
    return item


def _settled_look(
    tools: Tools, name: str, camera: str | None, hand: Hand
) -> Generator[Motion, Outcome, Footprint | None]:
    """The object's footprint once it has stopped moving (or :data:`SETTLE_MAX_S` has passed)."""
    item = yield from _look_or_clear_view(tools, name, camera, hand)
    waited = 0.0
    while item is not None and waited < SETTLE_MAX_S:
        yield tools.wait(SETTLE_STEP_S)
        waited += SETTLE_STEP_S
        again = _look(tools, name, camera)
        if again is None:
            break
        drift = float(np.linalg.norm(again.center - item.center))
        item = again
        if drift < SETTLE_DRIFT:
            break
    return item


def _goal_centre_and_top(goal: Goal, target: Footprint | None) -> tuple[np.ndarray, float]:
    """The ``(2,)`` xy the object's centre goes to, and the height [m] of the surface it rests on there.

    For a container that surface is its rim: the object is let go above it and drops in.
    """
    if goal.region is not None:
        low, high = np.asarray(goal.region[0]), np.asarray(goal.region[1])
        return (low[:2] + high[:2]) / 2, float(low[2])
    centre = body_centre(target)[0] if goal.target_body else target.center[:2]
    return np.asarray(centre), target.top_z


def body_centre(footprint: Footprint) -> tuple[np.ndarray, float]:
    """The ``(2,)`` centre and radius [m] of the round body of an object with a handle (a mug, a pan).

    The body is the end of the footprint's long side its points crowd (a handle is thin), as wide as
    the footprint's short side. A footprint about as long as wide is all body.
    """
    spare = footprint.half_length - footprint.half_width
    if spare < ROUND_SLACK or footprint.centroid is None:
        return np.asarray(footprint.center[:2]), footprint.half_width
    along = np.array([math.cos(footprint.yaw), math.sin(footprint.yaw)])
    side = 1.0 if (footprint.centroid - footprint.center[:2]) @ along >= 0 else -1.0
    return footprint.center[:2] + side * spare * along, footprint.half_width


def _placed(item: Footprint, goal: Goal, centre: np.ndarray, top: float, target: Footprint | None = None) -> bool:
    """The object rests at the goal: its bottom down on ``top``, its centre inside the goal's box or over ``centre``.

    A box goal takes the object anywhere inside it, as the task's own check does: an object let go
    in the basket rolls a few centimetres, and picking it up again to centre it is no demonstration.
    A container goal likewise takes it anywhere over the container's footprint, inset by its half width.
    """
    if item.bottom_z > top + PLACED_HEIGHT:
        return False
    if goal.region is not None:
        low, high = np.asarray(goal.region[0][:2]), np.asarray(goal.region[1][:2])
        return bool(np.all(item.center[:2] > low) and np.all(item.center[:2] < high))
    if goal.container:
        local = _local(target, item.center[:2])
        return bool(
            abs(local[0]) < target.half_length - item.half_width and abs(local[1]) < target.half_width - item.half_width
        )
    return bool(np.linalg.norm(item.center[:2] - centre) < PLACED_TOLERANCE)


def _aim(
    goal: Goal, target: Footprint | None, item: Footprint, centre: np.ndarray, others: list[Footprint] = ()
) -> np.ndarray:
    """Where in a container goal the object's centre goes: ``centre`` for any other goal.

    The spot keeps the object inside the container's footprint (inset by its half length, whichever
    way it comes to lie, and :data:`CONTAINER_WALL`), within :attr:`Goal.max_reach` of the base if
    any spot is (the nearest to the base if none is), as far as it can from ``others`` — what
    already lies in the container — and otherwise as near the middle as it can.
    """
    if not goal.container or (goal.max_reach is None and not others):
        return centre
    along = np.array([math.cos(target.yaw), math.sin(target.yaw)])
    across = np.array([-along[1], along[0]])
    inset = item.half_length + CONTAINER_WALL
    room = np.maximum([target.half_length - inset, target.half_width - inset], 0.0)
    grid = np.linspace(-1.0, 1.0, 9)
    spots = np.array([centre + a * room[0] * along + b * room[1] * across for a in grid for b in grid])
    reach = np.linalg.norm(spots, axis=1)
    if goal.max_reach is not None:
        within = reach <= goal.max_reach
        spots = spots[within] if within.any() else spots[reach <= reach.min() + 1e-6]
    clearance = np.full(len(spots), np.inf)
    for other in others:
        clearance = np.minimum(clearance, np.linalg.norm(spots - other.center[:2], axis=1) - other.half_length)
    off_middle = np.linalg.norm(spots - centre, axis=1)
    best = np.lexsort((off_middle, -np.round(clearance, 3)))[0]  # most clearance (to the mm), then the middle
    return spots[best]


def _local(footprint: Footprint, xy: np.ndarray) -> np.ndarray:
    """``xy`` in ``footprint``'s frame: along its long side, then across it."""
    along = np.array([math.cos(footprint.yaw), math.sin(footprint.yaw)])
    offset = xy - footprint.center[:2]
    return np.array([offset @ along, offset[0] * -along[1] + offset[1] * along[0]])


def _mistake_spot(tools: Tools, goal: Goal, item: Footprint, centre: np.ndarray) -> np.ndarray:
    """A spot in ``goal.mistake_bounds`` clear of the goal by the object's length, and of where it lies now."""
    low, high = np.asarray(goal.mistake_bounds[0]), np.asarray(goal.mistake_bounds[1])
    clearance = 2 * item.half_length + 0.02
    spot = tools.rng.uniform(low, high)
    for _ in range(50):
        if np.linalg.norm(spot - centre) > clearance and np.linalg.norm(spot - item.center[:2]) > item.half_length:
            break
        spot = tools.rng.uniform(low, high)
    return spot


def _grasp(tools: Tools, item: Footprint, obstacles: list[Footprint], hand: Hand) -> tuple[float, np.ndarray, float]:
    """Heading, ``(2,)`` xy and height of a top-down grasp across ``item``'s short side, drawn a little off centre.

    The grasp slides along the long side, as little as it can, until neither open finger comes down
    over an ``obstacle`` taller than the fingertips there: a spoon lying against the folded cloth is
    picked from its far end, or one finger lands on the cloth and the hand stops short of the spoon.
    """
    grasp_z = max(item.top_z - min(GRASP_DEPTH, item.height / 2), item.bottom_z + tools.tip_reach + FLOOR_GAP)
    tip_z = grasp_z - tools.tip_reach
    spare = item.half_length - item.half_width
    drawn = tools.rng.uniform(-GRASP_SLIDE, GRASP_SLIDE) * spare
    candidates = sorted(np.linspace(-MAX_GRASP_SLIDE * spare, MAX_GRASP_SLIDE * spare, 9), key=lambda s: abs(s - drawn))
    # A square object (a cube) closes as well across either side: the second heading is tried too.
    headings = [item.yaw] if spare >= ROUND_SLACK else [item.yaw, item.yaw + math.pi / 2]
    for heading in headings:
        along = np.array([math.cos(heading), math.sin(heading)])
        across = np.array([-along[1], along[0]])
        for slide in [drawn, *candidates]:
            grasp_xy = item.center[:2] + slide * along
            fingers = [grasp_xy + side * reach * across for side in (-1, 1) for reach in hand.finger_reach]
            if not any(
                _over(obstacle, finger) for obstacle in obstacles if obstacle.top_z > tip_z for finger in fingers
            ):
                return heading + math.pi / 2 - tools.finger_axis_yaw, grasp_xy, grasp_z
    tools.note("every grasp along the object has a finger over an obstacle")
    along = np.array([math.cos(item.yaw), math.sin(item.yaw)])
    return item.yaw + math.pi / 2 - tools.finger_axis_yaw, item.center[:2] + drawn * along, grasp_z


def _rim_grasp(tools: Tools, item: Footprint) -> tuple[float, np.ndarray, float]:
    """Heading, ``(2,)`` xy and height of a top-down grasp on the wall of a hollow ``item``, one finger inside it.

    On the body's side nearer the robot's base, the fingers closing radially: they straddle the wall
    whatever its thickness, since the inner one only has to come down inside the opening.
    """
    centre, radius = body_centre(item)
    along = np.array([math.cos(item.yaw), math.sin(item.yaw)])
    across = np.array([-along[1], along[0]])
    if across @ centre > 0:  # the side of the body facing the base
        across = -across
    grasp_xy = centre + (radius - RIM_INSET) * across
    yaw = math.atan2(across[1], across[0]) - tools.finger_axis_yaw
    grasp_z = max(item.top_z - RIM_DEPTH, item.bottom_z + tools.tip_reach + FLOOR_GAP)
    return yaw, grasp_xy, grasp_z


def _over(footprint: Footprint, xy: np.ndarray) -> bool:
    """``xy`` lies over ``footprint``'s rectangle, widened by :data:`FINGER_CLEARANCE`."""
    local = _local(footprint, xy)
    return bool(
        abs(local[0]) < footprint.half_length + FINGER_CLEARANCE
        and abs(local[1]) < footprint.half_width + FINGER_CLEARANCE
    )


def _pick_and_place(
    tools: Tools,
    name: str,
    item: Footprint,
    centre: np.ndarray,
    top: float,
    goal: Goal,
    camera: str | None,
    hand: Hand,
    grasp: str,
    clutter: tuple[str, ...],
) -> Generator[Motion, Outcome, bool]:
    """Pick ``name`` from where ``item`` says it is and lower its centre onto ``centre`` at height ``top``.

    False if every pick missed.
    """
    rng = tools.rng
    # What the open fingers must not come down on: the goal's own object, which a pick can lie against.
    names = ((goal.target,) if goal.target is not None else ()) + clutter
    obstacles = [obstacle for obstacle in (_look(tools, n, camera) for n in names) if obstacle is not None]
    for attempt in range(MAX_PICK_ATTEMPTS):
        yield tools.open_gripper()
        # Seen again from the hover pose: a pick aimed where the object was before it rolled closes on air.
        for _ in range(2):
            yaw, grasp_xy, grasp_z = (
                _grasp(tools, item, obstacles, hand) if grasp == "pinch" else _rim_grasp(tools, item)
            )
            hover_z = max(item.top_z + rng.uniform(*HOVER_ABOVE), goal.obstacle_top + HOVER_CLEARANCE) + tools.tip_reach
            yield tools.move_to((grasp_xy[0], grasp_xy[1], hover_z), yaw)
            again = _look(tools, name, camera)
            if again is None or np.linalg.norm(again.center - item.center) < REAIM_DISTANCE:
                break
            tools.note(f"{name}: moved {1000 * np.linalg.norm(again.center - item.center):.0f} mm, re-aiming")
            item = again
        yield tools.move_to((grasp_xy[0], grasp_xy[1], grasp_z), yaw, tolerance=GRASP_TOLERANCE)
        yield tools.wait(rng.uniform(*GRASP_PAUSE))
        yield tools.close_gripper()
        hang = grasp_z - item.bottom_z
        carry_z = max(goal.obstacle_top, top) + rng.uniform(*CARRY_CLEARANCE) + hang
        yield tools.move_to((grasp_xy[0], grasp_xy[1], carry_z), yaw)
        if tools.holding():
            break
        tools.note(f"{name}: pick {attempt + 1} missed (closure {tools.gripper_closure():.2f})")
        item = yield from _settled_look(tools, name, camera, hand)
        if item is None:
            return False
    else:
        return False

    # Where the object sits in the hand, as it is carried: looked at again, since it may have shifted.
    held = _look(tools, name, camera) or item
    offset = held.center[:2] - tools.tool_pos()[:2]
    hang = tools.tool_pos()[2] - held.bottom_z
    turn = 0.0
    if goal.object_yaw is not None:
        turn = (goal.object_yaw - held.yaw + math.pi / 2) % math.pi - math.pi / 2
    cos, sin = math.cos(turn), math.sin(turn)
    offset = np.array([cos * offset[0] - sin * offset[1], sin * offset[0] + cos * offset[1]])
    hand_xy = centre - offset
    release_z = top + RELEASE_GAP + hang
    yield tools.move_to((hand_xy[0], hand_xy[1], carry_z), yaw + turn, tolerance=PLACE_TOLERANCE)
    yield tools.move_to((hand_xy[0], hand_xy[1], release_z), tolerance=PLACE_TOLERANCE)
    if goal.scrub > 0:
        yield from _scrub(tools, goal.scrub)
    yield tools.open_gripper()
    yield tools.move_by((0.0, 0.0, carry_z - release_z), tolerance=PLACE_TOLERANCE)
    return True


def _scrub(tools: Tools, amplitude: float) -> Generator[Motion, Outcome, None]:
    """Press the held object onto what is under it and rub it ``amplitude`` [m] to either side, then return."""
    yield tools.move_until_contact((0.0, 0.0, -1.0), max_distance=SCRUB_REACH)
    turn = tools.rng.uniform(-math.pi, math.pi)
    stroke = amplitude * np.array([math.cos(turn), math.sin(turn), 0.0])
    yield tools.move_by(stroke, tolerance=PLACE_TOLERANCE)
    for _ in range(SCRUB_STROKES - 1):
        yield tools.move_by(-2 * stroke, tolerance=PLACE_TOLERANCE)
        yield tools.move_by(2 * stroke, tolerance=PLACE_TOLERANCE)
    yield tools.move_by(-stroke, tolerance=PLACE_TOLERANCE)
