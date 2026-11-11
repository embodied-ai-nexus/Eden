"""A tool routine that sorts robosuite's groceries into their bin2 quadrants.

Run by ``eden datagen --task robosuite_pick_place``.

Written against :class:`~eden.extensions.scripted_demos.tools.Tools`, for one robot, as the
routines of :mod:`eden.extensions.scripted_demos.routines` are: look at every object, pick one that
is not in its quadrant, carry it over the bin walls and lower it in. The loop is goal-directed, so
the episode's deliberate mistakes (an object put in the wrong quadrant, with probability
``p_mistake``) and a dropped object are recovered from by looking again.

The same routine runs privileged (object footprints off the collision meshes) or camera-only
(:meth:`~eden.extensions.scripted_demos.tools.Tools.locate` through ``agentview``): the two give a
:class:`~eden.extensions.scripted_demos.tools.Footprint` alike, and only :func:`_look` chooses.
The bins are fixtures, so their geometry is a constant here in either mode.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from eden.tasks.robosuite.pick_place.config import BIN_TOP_Z, OBJECT_NAMES, target_region

if TYPE_CHECKING:
    from collections.abc import Generator

    from eden.extensions.scripted_demos.routines import Routine
    from eden.extensions.scripted_demos.tools import Footprint, Motion, Outcome, Tools

#: Top of the bins' walls [m] (their collision meshes' highest point): a carried object's bottom stays above it.
BIN_RIM_Z = 0.90
#: Clearance [m] of a carried object's bottom over the rims and the other objects, drawn per carry.
CARRY_CLEARANCE = (0.03, 0.07)
#: Fingertip clearance [m] of the empty hand over the bin rims and the other objects, and how far
#: above the object's own top it hovers before the descent (drawn per pick).
HOVER_CLEARANCE = 0.02
HOVER_ABOVE = (0.05, 0.09)
#: How deep [m] below an object's top the pads close — at most half its height. Deep enough to
#: hold the milk carton below its gable top, which slips out of a 3 cm grasp and topples the carton.
GRASP_DEPTH = 0.045
#: Gap [m] between the object's bottom and the bin floor when it is let go.
RELEASE_GAP = 0.015
#: How long [s] the hand pauses at the grasp pose before closing, drawn per pick: the IK still trails
#: the move by 7-9 mm when it first comes within a centimetre, and settles in the pause. A tighter
#: move tolerance instead timed out at a steady ~9 mm in a third of the picks.
GRASP_PAUSE = (0.2, 0.6)
#: Grasp offset along the object's long axis, drawn per grasp as a fraction of how much longer than wide it is.
GRASP_SLIDE = 0.3
#: Clearance [m] between a placed object's footprint and its cell's near walls, and how far further in it may land.
PLACE_MARGIN = 0.015
PLACE_JITTER = 0.015
#: How long [s] the routine waits at the start of an episode for the dropped objects to settle.
SETTLE_S = 0.5
#: Tolerance [m] of the moves over bin2. Its far cells are at the edge of the arm's reach with the
#: hand pointing down, where the arm stops 1-2 cm short of a tighter target and waits out the timeout.
PLACE_TOLERANCE = 0.02
#: Picks of one object before the routine moves on to another, and pick cycles before it gives the object up.
MAX_PICK_ATTEMPTS = 3
MAX_PICK_CYCLES = 2
#: Widest [m] an object may be across its short side to be gripped from above: a Panda hand opens
#: 8 cm, a Robotiq 2F-85 8.5 cm, less the hand's tracking error. A cereal box knocked flat is 7.5 cm.
MAX_GRASP_WIDTH = 0.065
#: Pick-and-places per episode before the routine gives up (two per object, plus mistakes).
MAX_PLACEMENTS = 12
#: Where the hand goes to clear the camera's view of the bins when an object cannot be seen.
CLEAR_VIEW_POS = (-0.3, -0.1, 1.15)


def pick_place_routine(objects: tuple[str, ...], *, p_mistake: float, camera: str | None) -> Routine:
    """The routine for the grocery ``objects`` present in the scene; camera-only when ``camera`` is given."""
    unknown = sorted(set(objects) - set(OBJECT_NAMES))
    if unknown:
        raise KeyError(f"No bin2 quadrant for objects {unknown}; robosuite's PickPlace objects: {list(OBJECT_NAMES)}.")

    def sort_groceries(tools: Tools) -> Generator[Motion, Outcome, None]:
        # The reset drops every object from 1 cm up; look once they have landed.
        yield tools.wait(SETTLE_S)
        failed_cycles = dict.fromkeys(objects, 0)
        for _ in range(MAX_PLACEMENTS):
            seen = {}
            for name in objects:
                if failed_cycles[name] >= MAX_PICK_CYCLES:
                    continue
                footprint = _look(tools, name, camera)
                if footprint is None:
                    yield tools.move_to(CLEAR_VIEW_POS)
                    footprint = _look(tools, name, camera)
                if footprint is None:
                    tools.note(f"{name}: not seen, left out")
                    continue
                if 2 * footprint.half_width > MAX_GRASP_WIDTH:
                    tools.note(f"{name}: {200 * footprint.half_width:.1f} cm across, too wide to grip from above")
                    failed_cycles[name] = MAX_PICK_CYCLES
                    continue
                seen[name] = footprint
            occupant = {name: _quadrant_of(footprint) for name, footprint in seen.items()}
            misplaced = [name for name in seen if occupant[name] != name]
            if not misplaced:
                return
            # An object sitting in another's cell goes first: its owner would be lowered onto it.
            intruders = [name for name in misplaced if occupant[name] is not None]
            candidates = intruders or misplaced
            name = candidates[int(tools.rng.integers(len(candidates)))]
            quadrant, kind = name, "goal"
            empty = [cell for cell in OBJECT_NAMES if cell != name and cell not in occupant.values()]
            if empty and tools.rng.random() < p_mistake:
                quadrant, kind = empty[int(tools.rng.integers(len(empty)))], "mistake"
            goal = _place_xy(tools, quadrant, seen[name])
            tools.note(f"{name} -> {kind} quadrant ({goal[0]:.3f}, {goal[1]:.3f})")
            # A carried object passes over the others: the cereal box stands 7 cm above the bin rims.
            obstacle_top = max([BIN_RIM_Z] + [footprint.top_z for other, footprint in seen.items() if other != name])
            placed = yield from _pick_and_place(tools, name, seen[name], goal, camera, obstacle_top)
            if not placed:
                failed_cycles[name] += 1
        tools.note(f"gave up after {MAX_PLACEMENTS} placements")

    return sort_groceries


def _look(tools: Tools, name: str, camera: str | None) -> Footprint | None:
    """The object's footprint: off the simulator when privileged, through ``camera`` otherwise."""
    return tools.object_footprint(name) if camera is None else tools.locate(name, camera)


def _place_xy(tools: Tools, quadrant: str, footprint: Footprint) -> tuple[float, float]:
    """Where in ``quadrant``'s bin2 cell to lower an object of ``footprint``: on the robot's side of it.

    The quadrant centres of the far column are at the edge of the Panda's reach with the hand
    pointing down (robosuite's own target, the centre, timed out 5 cm short of it), so the object
    goes to the cell's corner nearest the robot, inset by its own extent, then a few cm in at random.
    """
    low, high = target_region(quadrant)
    inset = math.hypot(footprint.half_length, footprint.half_width) + PLACE_MARGIN
    jitter = tools.rng.uniform(0.0, PLACE_JITTER, size=2)
    x = low[0] + inset + jitter[0]
    y = low[1] + inset + jitter[1]
    return min(x, (low[0] + high[0]) / 2), min(y, (low[1] + high[1]) / 2)


def _quadrant_of(footprint: Footprint) -> str | None:
    """Whose bin2 cell the object rests in (the task's in-bin box, in xy, and down on the floor); None outside bin2."""
    x, y = footprint.center[:2]
    for cell in OBJECT_NAMES:
        low, high = target_region(cell)
        if low[0] <= x <= high[0] and low[1] <= y <= high[1] and footprint.bottom_z < BIN_TOP_Z + 0.04:
            return cell
    return None


def _pick_and_place(
    tools: Tools,
    name: str,
    footprint: Footprint,
    goal: tuple[float, float],
    camera: str | None,
    obstacle_top: float,
) -> Generator[Motion, Outcome, bool]:
    """Pick ``name`` from where ``footprint`` says it is and lower it at ``goal`` in bin2; False if every pick missed.

    It is carried with its bottom above ``obstacle_top`` [m] — the bin rims and every other object.
    """
    rng = tools.rng
    for attempt in range(MAX_PICK_ATTEMPTS):
        # Close across the short side, a little off centre along the long one.
        yaw = footprint.yaw + math.pi / 2 - tools.finger_axis_yaw
        # Only along an elongated object: off the axis of a can, the pads squeeze it out.
        slide = rng.uniform(-GRASP_SLIDE, GRASP_SLIDE) * (footprint.half_length - footprint.half_width)
        along = np.array([math.cos(footprint.yaw), math.sin(footprint.yaw), 0.0])
        grasp_z = footprint.top_z - min(GRASP_DEPTH, footprint.height / 2)
        hang = grasp_z - footprint.bottom_z  # tool point above the object's bottom while carried
        carry_z = obstacle_top + rng.uniform(*CARRY_CLEARANCE) + hang
        grasp = footprint.top_center[:2] + slide * along[:2]

        # The empty hand only has to clear the others with its fingertips; it lifts to carry_z once loaded.
        hover_z = max(obstacle_top + tools.tip_reach + HOVER_CLEARANCE, footprint.top_z + rng.uniform(*HOVER_ABOVE))
        yield tools.open_gripper()
        yield tools.move_to((grasp[0], grasp[1], hover_z), yaw)
        yield tools.move_to((grasp[0], grasp[1], grasp_z), yaw)
        yield tools.wait(rng.uniform(*GRASP_PAUSE))
        yield tools.close_gripper()
        yield tools.move_to((grasp[0], grasp[1], carry_z), yaw)
        if tools.holding():
            break
        tools.note(f"{name}: pick {attempt + 1} missed (closure {tools.gripper_closure():.2f})")
        yield tools.open_gripper()
        footprint = _look(tools, name, camera)
        if footprint is None:
            return False
    else:
        return False

    release_z = BIN_TOP_Z + RELEASE_GAP + hang
    yield tools.move_to((goal[0], goal[1], carry_z), tolerance=PLACE_TOLERANCE)
    yield tools.move_to((goal[0], goal[1], release_z), tolerance=PLACE_TOLERANCE)
    yield tools.open_gripper()
    yield tools.move_by((0.0, 0.0, carry_z - release_z), tolerance=PLACE_TOLERANCE)
    return True
