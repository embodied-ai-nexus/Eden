"""Two tool routines that do robosuite's TwoArmTransport together (``eden datagen --task robosuite_two_arm_transport``).

Written against :class:`~eden.extensions.scripted_demos.tools.Tools`, one routine per arm, run side
by side by :class:`~eden.extensions.scripted_demos.multi_arm.MultiArmDemonstrator`:

* ``robot0`` (behind the start bin) lifts the lid off the start bin by its handle and sets it down
  beside the bin, takes the hammer out by its balance point and holds it out over the gap between
  the tables, its free end toward ``robot1``;
* ``robot1`` (behind the target and trash bins) moves the trash cube from the target bin into the
  trash bin, then takes the hammer from ``robot0`` by the free end and lowers it into the target bin.

The hand-off goes through the env's blackboard (:attr:`Tools.shared`): ``robot0`` writes ``offered``
once the hammer is held out, ``robot1`` writes ``taken`` once its fingers are closed on it too, and
only then does ``robot0`` open its fingers and write ``released``. The hammer is turned to point
across the gap first, by as much of the hand's own quarter turn as the wrist allows
(:func:`_turn_toward_robot0`).

Privileged: the routines read object poses off the simulator (:meth:`Tools.link_pose`) — the
handle is a fixed segment of the hammer's frame, which the footprint of the whole hammer does not
locate. The bins are fixtures; their floor and rim are constants of ``bin.xml`` relative to the bin's
frame.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from eden.utils.geom import heading_from_quat, transform_by_quat, wrap_to_pi

if TYPE_CHECKING:
    from collections.abc import Generator

    from eden.extensions.scripted_demos.routines import Routine
    from eden.extensions.scripted_demos.tools import Motion, Outcome, Tools

#: ``bin.xml``: the floor (top of the base slab) and the rim above the bin's frame, which sits at the
#: bin's half height above the table [m].
BIN_FLOOR_ABOVE_ROOT = -0.065
BIN_RIM_ABOVE_ROOT = 0.075
BIN_HALF_HEIGHT = 0.075
#: ``lid.xml``: the handle's centre above the lid's frame (it is 3 cm tall, 2 cm across, 8 cm long
#: along the lid's y), and the lid's bottom below the frame [m].
LID_HANDLE_ABOVE_ROOT = 0.005
LID_BOTTOM_BELOW_ROOT = 0.02
#: How far above the handle's centre the pads close [m]: the fingertips stay clear of the plate.
LID_GRASP_ABOVE_CENTRE = 0.004
#: Where the lid is set down: this far along x from the start bin, on the free half of the same table
#: (the bin spans the table's whole width) [m].
LID_DROP_OFFSET_X = -0.4
#: ``hammer_payload.xml``: the handle runs along the hammer frame's z over [-0.1, 0.1], the head at +z,
#: and the centre of mass is at s = 0.054 (measured on the built hammer). robot0 holds it there, so
#: it stays level in the fingers and robot1 can come down onto the free end [m along the handle].
GIVER_GRIP_S = 0.054
TAKER_GRIP_S = -0.06
#: The 4 cm trash cube's half size [m].
TRASH_HALF = 0.02
#: Where the point midway between the two grips is held for the hand-off, over the gap between the
#: tables [m]. The hammer is turned to point across the gap so each hand is ~0.76 m from its arm's
#: base: pointing along x, both hands would sit at y = 0, 0.81 m out, past where robot0's hand
#: reaches (it stopped 5 cm short).
HANDOVER_MIDPOINT = (0.0, -0.005, 1.15)
#: Heading [rad] the hammer's head is turned toward for the hand-off (robot0), and the least turn
#: [rad] that still gets both hands within reach. The hand turns as far toward it as its last joint
#: allows (:meth:`Tools.heading_window`), 1.0-1.57 rad on the robosuite Panda's grasp pose.
HANDOVER_HEAD_YAW = -math.pi / 2
MIN_HANDOVER_TURN = 0.9
#: Clearance [m] of robot1's hand over the held-out hammer's top on its way in.
HAND_OVER_CLEARANCE = 0.05
#: How long [s] robot1 lets the hammer settle once robot0 has let go: held by its free end, it swings
#: head-down.
SWING_SETTLE_S = 1.0
#: Clearance [m] of a carried object's lowest point over the bin rims (drawn per carry).
CARRY_CLEARANCE = (0.05, 0.08)
#: Clearance [m] of the hammer's lowest point over the start bin's rim on the way to the hand-off.
TRANSIT_CLEARANCE = 0.1
#: How far above a grasp [m] the hand hovers before descending onto it (drawn per grasp).
HOVER_ABOVE = (0.05, 0.08)
#: Gap [m] between a released object's lowest point and the floor it is set down on.
RELEASE_GAP = 0.012
#: How long [s] the hand pauses at a grasp pose before closing (drawn per grasp), and how long the
#: routines wait for the reset's drop to settle.
GRASP_PAUSE = (0.2, 0.5)
SETTLE_S = 0.5
#: How often [s] a routine looks at the blackboard while it waits for the other arm, and for how long at most.
POLL_S = 0.1
MAX_WAIT_S = 60.0
#: Grasp attempts per object before the routine gives it up.
MAX_ATTEMPTS = 3
#: Where each arm's hand goes when its part is done: back over its own table, out of the other arm's way.
RETREAT = {"robot0": (0.0, -0.35, 1.1), "robot1": (0.0, 0.35, 1.1)}

#: The blackboard keys of the hand-off.
OFFERED, TAKEN, RELEASED, GAVE_UP = "offered", "taken", "released", "gave_up"


def two_arm_transport_routines() -> dict[str, Routine]:
    """The two routines, keyed by the arm (scene entry) each drives."""
    return {"robot0": _giver, "robot1": _taker}


def _giver(tools: Tools) -> Generator[Motion, Outcome, None]:
    """robot0: the lid off the start bin, the hammer out of it, held out until robot1 has it."""
    yield tools.wait(SETTLE_S)
    if not (yield from _remove_lid(tools)):
        tools.shared[GAVE_UP] = True
        return
    if not (yield from _pick_hammer(tools)):
        tools.shared[GAVE_UP] = True
        return
    if not (yield from _turn_toward_robot0(tools)):
        tools.shared[GAVE_UP] = True
        return
    # Over the bin's wall with room to spare: the head hangs 5 cm below the handle.
    yield from _clear(tools, "payload", _bin_z(tools, "start_bin", BIN_RIM_ABOVE_ROOT) + TRANSIT_CLEARANCE)
    target = _handover_tool_target(tools)
    yield tools.move_to((target[0], target[1], tools.tool_pos()[2]), tolerance=0.01, timeout_s=4.0)
    yield tools.move_to(target, tolerance=0.005, timeout_s=3.0)
    yield tools.wait(0.3)
    if not tools.holding():
        tools.note("dropped the hammer on the way to the hand-off")
        tools.shared[GAVE_UP] = True
        return
    tools.note("hammer held out for the hand-off")
    tools.shared[OFFERED] = True
    if not (yield from _wait_for(tools, TAKEN)):
        return
    yield tools.open_gripper()
    tools.shared[RELEASED] = True
    yield tools.move_by((0.0, 0.0, 0.06))
    yield tools.move_to(RETREAT["robot0"])


def _taker(tools: Tools) -> Generator[Motion, Outcome, None]:
    """robot1: the trash into the trash bin, then the hammer from robot0 into the target bin."""
    yield tools.wait(SETTLE_S)
    if not (yield from _move_trash(tools)):
        # The episode cannot succeed with the trash still in the target bin: end both routines here.
        tools.shared[GAVE_UP] = True
        yield tools.move_to(RETREAT["robot1"])
        return
    yield tools.move_to(RETREAT["robot1"])
    if not (yield from _wait_for(tools, OFFERED)):
        return
    if not (yield from _take_over(tools)):
        tools.shared[GAVE_UP] = True
        yield tools.move_to(RETREAT["robot1"])
        return
    if not (yield from _wait_for(tools, RELEASED)):
        return
    yield tools.move_by((0.0, 0.0, 0.04))
    yield tools.wait(SWING_SETTLE_S)
    yield from _lower_into_bin(tools, "payload", "target_bin")
    tools.note("hammer in the target bin")
    yield tools.move_to(RETREAT["robot1"])


# --- the parts ------------------------------------------------------------------------------------


def _remove_lid(tools: Tools) -> Generator[Motion, Outcome, bool]:
    """Lift the lid off the start bin by its handle and set it down beside the bin; False if every grasp missed."""
    rng = tools.rng
    rim_z = _bin_z(tools, "start_bin", BIN_RIM_ABOVE_ROOT)
    for attempt in range(MAX_ATTEMPTS):
        pos, quat = tools.link_pose("lid")
        # The handle is long along the lid's y: close across it, along the lid's x.
        heading = _nearest_equivalent(_yaw(quat) - tools.finger_axis_yaw, tools.tool_yaw(), math.pi)
        grasp = pos + _rotate(quat, (0.0, 0.0, LID_HANDLE_ABOVE_ROOT + LID_GRASP_ABOVE_CENTRE))
        yield tools.open_gripper()
        yield from _travel(tools, grasp + (0.0, 0.0, rng.uniform(*HOVER_ABOVE)), heading)
        yield tools.move_to(grasp, heading, tolerance=0.005)
        yield tools.wait(rng.uniform(*GRASP_PAUSE))
        yield tools.close_gripper()
        carry_z = grasp[2] + (rim_z - (pos[2] - LID_BOTTOM_BELOW_ROOT)) + rng.uniform(*CARRY_CLEARANCE)
        yield tools.move_to((grasp[0], grasp[1], carry_z))
        if tools.holding():
            break
        tools.note(f"lid: grasp {attempt + 1} missed (closure {tools.gripper_closure():.2f})")
    else:
        return False
    bin_pos, _ = tools.link_pose("start_bin")
    drop = bin_pos[:2] + (LID_DROP_OFFSET_X + rng.uniform(-0.02, 0.02), rng.uniform(-0.02, 0.02))
    lid_pos, _ = tools.link_pose("lid")
    hand_above_bottom = tools.tool_pos()[2] - (lid_pos[2] - LID_BOTTOM_BELOW_ROOT)
    release_z = bin_pos[2] - BIN_HALF_HEIGHT + RELEASE_GAP + hand_above_bottom
    yield tools.move_to((drop[0], drop[1], carry_z), tolerance=0.015)
    yield tools.move_to((drop[0], drop[1], release_z), tolerance=0.01)
    yield tools.open_gripper()
    yield tools.move_to((drop[0], drop[1], carry_z), tolerance=0.015)
    tools.note("lid off the bin")
    return True


def _pick_hammer(tools: Tools) -> Generator[Motion, Outcome, bool]:
    """Take the hammer out of the start bin by its balance point; False if every grasp missed."""
    rng = tools.rng
    rim_z = _bin_z(tools, "start_bin", BIN_RIM_ABOVE_ROOT)
    for attempt in range(MAX_ATTEMPTS):
        grasp = _handle_point(tools, GIVER_GRIP_S)
        heading = _across_handle(tools)
        yield tools.open_gripper()
        yield from _travel(tools, (grasp[0], grasp[1], rim_z + rng.uniform(*HOVER_ABOVE)), heading)
        yield tools.move_to(grasp, heading, tolerance=0.005)
        yield tools.wait(rng.uniform(*GRASP_PAUSE))
        yield tools.close_gripper()
        hang = tools.tool_pos()[2] - tools.object_footprint("payload").bottom_z
        clearance = rng.uniform(*CARRY_CLEARANCE)
        yield tools.move_to((grasp[0], grasp[1], rim_z + clearance + hang))
        if tools.holding():
            yield from _clear(tools, "payload", rim_z + clearance)
            tools.note("hammer out of the start bin")
            return True
        tools.note(f"hammer: grasp {attempt + 1} missed (closure {tools.gripper_closure():.2f})")
    return False


def _turn_toward_robot0(tools: Tools) -> Generator[Motion, Outcome, bool]:
    """Turn the held hammer's head toward robot0 as far as the wrist allows; False if that is too little for the hand-off.

    The hammer turns with the hand, so the turn must be the hand's own: :meth:`Tools.move_to` would
    realize a heading past the last joint's stop as the equivalent heading the other way, turning the
    hammer's head toward robot1. A flipped wrist (the same grasp a half turn round) has the room, but
    from it the arm no longer reaches the gap. The turn goes in two halves, each with one nearest
    equivalent.
    """
    low, high = tools.heading_window()
    turn = turn_within_window(wrap_to_pi(HANDOVER_HEAD_YAW - _head_yaw(tools)), tools.tool_yaw(), low, high)
    if abs(turn) < MIN_HANDOVER_TURN:
        tools.note(f"hammer: the wrist turns it only {turn:+.2f} rad toward robot0, too little for the hand-off")
        return False
    for _ in range(2):
        yield tools.move_to(tools.tool_pos(), tools.tool_yaw() + turn / 2, tolerance=0.01)
    tools.note(f"hammer turned {turn:+.2f} rad, head at {_head_yaw(tools):+.2f}")
    return True


def turn_within_window(turn: float, heading: float, low: float, high: float) -> float:
    """``turn`` [rad] cut to the room the wrist has in its direction, from ``heading`` within ``[low, high]``.

    The cut never reverses the turn: a wrist already past that end of the window has no room, and
    the turn is 0, not a turn the other way.
    """
    room = max(0.0, high - heading) if turn > 0 else max(0.0, heading - low)
    return math.copysign(min(abs(turn), room), turn)


def _handover_tool_target(tools: Tools) -> np.ndarray:
    """Where robot0's hand goes so the point midway between its grip and robot1's lies at :data:`HANDOVER_MIDPOINT`."""
    grip = _handle_point(tools, GIVER_GRIP_S)
    span = _handle_point(tools, TAKER_GRIP_S) - grip
    span[2] = 0.0
    return np.asarray(HANDOVER_MIDPOINT) - span / 2 + (tools.tool_pos() - grip) * (1.0, 1.0, 0.0)


def _take_over(tools: Tools) -> Generator[Motion, Outcome, bool]:
    """Close on the free end of the hammer robot0 holds out; True once it is held.

    The hand comes in above the hammer and descends onto the handle: a hand swept in at the hammer's
    height would knock it out of robot0's fingers.
    """
    rng = tools.rng
    s = TAKER_GRIP_S
    for attempt in range(MAX_ATTEMPTS):
        grasp = _handle_point(tools, s)
        heading = _across_handle(tools)
        above = max(grasp[2] + rng.uniform(*HOVER_ABOVE), tools.object_footprint("payload").top_z + HAND_OVER_CLEARANCE)
        yield tools.open_gripper()
        yield tools.move_to((*tools.tool_pos()[:2], max(above, tools.tool_pos()[2])))
        yield tools.move_to((grasp[0], grasp[1], above), heading)
        grasp = _handle_point(tools, s)  # robot0's hand may have settled since
        yield tools.move_to(grasp, heading, tolerance=0.005)
        yield tools.wait(rng.uniform(*GRASP_PAUSE))
        yield tools.close_gripper()
        if tools.holding():
            tools.note("took the hammer from robot0")
            tools.shared[TAKEN] = True
            return True
        tools.note(f"hand-off: grasp {attempt + 1} missed (closure {tools.gripper_closure():.2f})")
        yield tools.open_gripper()
        yield tools.move_by((0.0, 0.0, 0.06))
    return False


def _move_trash(tools: Tools) -> Generator[Motion, Outcome, bool]:
    """Pick the trash cube out of the target bin and lower it into the trash bin; False if every grasp missed."""
    rng = tools.rng
    rim_z = _bin_z(tools, "target_bin", BIN_RIM_ABOVE_ROOT)
    for attempt in range(MAX_ATTEMPTS):
        pos, quat = tools.link_pose("trash")
        # A cube: close across whichever face pair is nearest the hand's current heading.
        heading = _nearest_equivalent(_yaw(quat) - tools.finger_axis_yaw, tools.tool_yaw(), math.pi / 2)
        grasp = pos + (0.0, 0.0, 0.25 * TRASH_HALF)
        yield tools.open_gripper()
        yield from _travel(tools, (grasp[0], grasp[1], rim_z + rng.uniform(*HOVER_ABOVE)), heading)
        yield tools.move_to(grasp, heading, tolerance=0.005)
        yield tools.wait(rng.uniform(*GRASP_PAUSE))
        yield tools.close_gripper()
        hang = tools.tool_pos()[2] - tools.object_footprint("trash").bottom_z
        clearance = rng.uniform(*CARRY_CLEARANCE)
        yield tools.move_to((grasp[0], grasp[1], rim_z + clearance + hang))
        if tools.holding():
            yield from _clear(tools, "trash", rim_z + clearance)
            break
        tools.note(f"trash: grasp {attempt + 1} missed (closure {tools.gripper_closure():.2f})")
    else:
        return False
    yield from _lower_into_bin(tools, "trash", "trash_bin")
    tools.note("trash in the trash bin")
    return True


def _lower_into_bin(tools: Tools, name: str, bin_name: str) -> Generator[Motion, Outcome, None]:
    """Carry the held ``name`` over the middle of ``bin_name`` and let it go just above the floor."""
    rng = tools.rng
    bin_pos, _ = tools.link_pose(bin_name)
    rim_z = bin_pos[2] + BIN_RIM_ABOVE_ROOT
    held = tools.object_footprint(name)
    hang = tools.tool_pos()[2] - held.bottom_z
    offset = held.center - tools.tool_pos()
    carry_z = max(tools.tool_pos()[2], rim_z + rng.uniform(*CARRY_CLEARANCE) + hang)
    target = bin_pos[:2] - offset[:2] + rng.uniform(-0.015, 0.015, size=2)
    release_z = bin_pos[2] + BIN_FLOOR_ABOVE_ROOT + RELEASE_GAP + hang
    yield tools.move_to((tools.tool_pos()[0], tools.tool_pos()[1], carry_z), tolerance=0.015)
    yield tools.move_to((target[0], target[1], carry_z), tolerance=0.015)
    yield tools.move_to((target[0], target[1], release_z), tolerance=0.01)
    yield tools.open_gripper()
    yield tools.move_by((0.0, 0.0, carry_z - release_z))


def _travel(tools: Tools, target, heading: float | None = None) -> Generator[Motion, Outcome, None]:
    """Go to ``target`` over the top: up to its height first (if it is higher), then across, then down.

    A straight move from a hand low beside a bin to a point over it sweeps the fingers through the
    bin's wall, and the arm stays stuck against it.
    """
    target = np.asarray(target, dtype=np.float64)
    here = tools.tool_pos()
    if target[2] > here[2] + 0.01:
        yield tools.move_to((here[0], here[1], target[2]), tolerance=0.015)
    yield tools.move_to(target, heading)


def _clear(tools: Tools, name: str, floor_z: float) -> Generator[Motion, Outcome, None]:
    """Raise the hand until the held ``name``'s lowest point is above ``floor_z``.

    The hang measured at the grasp misses how a held object settles in the fingers once lifted, and
    an object that clips a bin rim on the way out is knocked out of the fingers.
    """
    shortfall = floor_z - tools.object_footprint(name).bottom_z
    if shortfall > 0.0:
        yield tools.move_by((0.0, 0.0, shortfall + 0.01))


# --- readings -------------------------------------------------------------------------------------


def _wait_for(tools: Tools, key: str) -> Generator[Motion, Outcome, bool]:
    """Hold still until the other arm writes ``key``; False (with a note) if it gave up or never did."""
    waited = 0.0
    while not tools.shared.get(key):
        if tools.shared.get(GAVE_UP):
            tools.note(f"the other arm gave up while waiting for '{key}'")
            return False
        if waited >= MAX_WAIT_S:
            tools.note(f"waited {MAX_WAIT_S:.0f} s for '{key}', gave up")
            tools.shared[GAVE_UP] = True
            return False
        yield tools.wait(POLL_S)
        waited += POLL_S
    return True


def _handle_point(tools: Tools, s: float) -> np.ndarray:
    """``(3,)`` world position of the point ``s`` along the hammer's handle axis (0 is its centre)."""
    pos, quat = tools.link_pose("payload")
    return pos + _rotate(quat, (0.0, 0.0, s))


def _across_handle(tools: Tools) -> float:
    """The effector heading that closes the fingers across the hammer's handle, nearest the hand's own."""
    return _nearest_equivalent(_head_yaw(tools) + math.pi / 2 - tools.finger_axis_yaw, tools.tool_yaw(), math.pi)


def _head_yaw(tools: Tools) -> float:
    """Heading [rad] the hammer's head points at: its handle axis (the frame's +z)."""
    _, quat = tools.link_pose("payload")
    return float(heading_from_quat(np.asarray(quat, dtype=np.float64), np.array((0.0, 0.0, 1.0))))


def _bin_z(tools: Tools, bin_name: str, above_root: float) -> float:
    return float(tools.link_pose(bin_name)[0][2] + above_root)


def _rotate(quat: np.ndarray, vector: tuple[float, float, float]) -> np.ndarray:
    return transform_by_quat(np.asarray(vector, dtype=np.float64), np.asarray(quat, dtype=np.float64))


def _yaw(quat: np.ndarray) -> float:
    """Heading [rad] of the frame's x axis."""
    return float(heading_from_quat(np.asarray(quat, dtype=np.float64)))


def _nearest_equivalent(angle: float, reference: float, period: float) -> float:
    """``angle`` shifted by a multiple of ``period`` to lie nearest ``reference``."""
    return angle + period * round((reference - angle) / period)


_giver.entity_names = ("lid", "payload", "start_bin")
_taker.entity_names = ("trash", "payload", "target_bin", "trash_bin")
