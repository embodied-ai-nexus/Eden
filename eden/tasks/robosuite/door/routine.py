"""A tool routine that opens robosuite's door (``eden datagen --task robosuite_door``).

Written against :class:`~eden.extensions.scripted_demos.tools.Tools`, for the Panda, as the
routines of :mod:`eden.extensions.scripted_demos.routines` are: lower the open fingers over the
handle lever, one each side of it, press it down with the palm until the latch clears the frame
(``--latch latch``), and swing the door open about its hinge with the back finger, turning the hand
with it — then lift off the lever.

What decides that shape, all found in rollouts:

* **The hand leans away from the panel.** The lever runs parallel to the panel 8 cm off its face,
  and a hand pointing straight down with its fingers either side of the lever has its 20 cm-wide
  palm in the panel. Tilted 30 degrees about the lever, the palm leans out
  (:meth:`~eden.extensions.scripted_demos.tools.Tools.aim`) and clears it by ~1.5 cm, with the back
  finger in the 6.5 cm gap behind the lever. Turning the handle is a turn of the hand about the
  hub's axis, which keeps every point of the hand at its distance from the panel.
* **It pushes with rigid parts, it does not pinch.** The palm presses the lever down, and the back
  finger, open at its stop, is rigid toward the panel: it pulls the door. A pinch depends on grip
  force -- squeezed with ~2 N, the latch spring (1 N m rad^-1 at 10 cm) pries the fingers open within
  half a turn -- while the push needs no grip at all.
* **The latch is a real lock, turned down.** Its bolt meets the frame post after ~0.02 rad of swing
  unless the handle is turned past ~1.25 rad (1.15 still catches it), either way. Down, because
  the hand then ends the turn pointing away from the robot; turned up it would end pointing back at
  the arm's own base. The hand along the lever, from its free end, is out of reach for the same
  reason: the lever points back at the robot.

The door is re-posed every reset, so every point is read off it — the lever in the ``latch`` link's
frame, the hinge in the ``door`` link's — through the privileged
:meth:`~eden.extensions.scripted_demos.tools.Tools.link_pose`.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
from genesis.utils.geom import axis_angle_to_quat, transform_quat_by_quat

from eden.utils.geom import transform_by_quat, wrap_to_pi, yaw_from_quat

if TYPE_CHECKING:
    from collections.abc import Generator

    from eden.extensions.scripted_demos.routines import Routine
    from eden.extensions.scripted_demos.tools import Motion, Outcome, Tools

#: The door entity the routine reads.
DOOR = "door"
#: Hinge axis in the ``door`` link frame (``door_lock.xml`` / ``door.xml``: ``<joint name="hinge" pos="0.255 0 0">``).
HINGE_IN_DOOR = (0.255, 0.0, 0.0)
# The lever in the ``latch`` link frame: from the hub along +x for 15 cm, its centre line 10 cm off
# the panel's mid-plane (``y = -0.10``); 3 cm thick along the panel normal (back face 8.5 cm off the
# mid-plane, the panel's face 2 cm), 4 cm tall.
#: Where along the lever [m from the hub] the tool point (the pad midpoint) sits, drawn per attempt:
#: the palm, 2.6 cm either side of it along the lever, clears the hub's 2.5 cm cylinder.
HOOK_ALONG = (0.08, 0.12)
#: How far the hand leans [rad] away from the panel, about the lever.
LEAN = math.radians(30.0)
#: Where the tool point sits off the panel's mid-plane, and below the lever's centre [m]. The palm's
#: underside then rests 2-3 mm above the lever's top, the back finger stands ~1 cm behind the lever's
#: back face and the palm's edge ~1.5 cm off the panel.
HOOK_OFF_PANEL = 0.105
HOOK_BELOW = 0.012
#: How far above [m] the hand lines up before lowering the fingers either side of the lever, drawn per attempt.
LINE_UP = (0.07, 0.11)
#: How far [m] off the panel the hand turns into its leaning pose before moving in, drawn per attempt.
STAND_OFF = (0.08, 0.12)
#: A deliberate mistake: the hand lines up this far [m] too far out from the panel, both fingers in
#: front of the lever. It presses the handle down but pulls nothing, sees the door has not moved,
#: and lines up again.
MISTAKE_OUT = 0.075
#: How far [rad] the handle is turned down, and the steps the turn is made in; the latch's stop is at 1.57.
LATCH_TURN = 1.45
TURN_STEP = 0.25
#: Angle [rad] the hand swings through about the hinge, in steps of at most :data:`PULL_STEP`. The
#: hinge stops at 0.4 rad and success is past 0.3.
PULL_ANGLE = 0.38
PULL_STEP = 0.1
#: Tolerance [m] of the turn's and the pull's moves: the lever and the door hold the hand back a little.
HELD_TOLERANCE = 0.015
#: Peak tool speed [m/s] of the turn and the pull.
TURN_SPEED = 0.03
PULL_SPEED = 0.05
#: How long [s] a released lever is given to spring back before the next attempt.
SETTLE_S = 1.0
#: The door counts as open past this angle [rad] — robosuite's success threshold.
OPEN_ANGLE = 0.3
MAX_ATTEMPTS = 3


def door_routine(*, latch: bool, p_mistake: float) -> Routine:
    """The routine for the door; ``latch`` turns the handle down before the pull (robosuite ``use_latch``)."""

    def open_door(tools: Tools) -> Generator[Motion, Outcome, None]:
        yield tools.open_gripper()
        miss = tools.rng.random() < p_mistake
        for attempt in range(MAX_ATTEMPTS):
            if miss:
                tools.note("mistake: lining up in front of the lever")
            rot = yield from _hook(tools, out=MISTAKE_OUT if miss else 0.0)
            miss = False
            if latch:
                rot = yield from _turn_handle(tools, rot)
            yield from _pull(tools, rot)
            angle = door_angle(tools)
            yield from _lift_off(tools)
            if angle > OPEN_ANGLE:
                tools.note(f"door open at {angle:.2f} rad")
                return
            tools.note(f"attempt {attempt + 1}: door only at {angle:.2f} rad")
            yield tools.wait(SETTLE_S)
        tools.note(f"gave up after {MAX_ATTEMPTS} attempts")

    open_door.entity_names = (DOOR,)
    return open_door


def lever_frame(tools: Tools) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The hub ``(3,)``, and the lever's direction, the panel's outward normal and the lever's up, in world."""
    hub, quat = tools.link_pose(DOOR, "latch")
    along, outward, up = (_rotate(quat, axis) for axis in ((1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, 1.0)))
    return hub, along, outward, up


def hinge_point(tools: Tools) -> np.ndarray:
    """World position of the hinge axis at the ``door`` link's origin height."""
    pos, quat = tools.link_pose(DOOR, "door")
    return pos + _rotate(quat, HINGE_IN_DOOR)


def door_angle(tools: Tools) -> float:
    """How far [rad] the panel has swung open: the ``door`` link's heading relative to the fixed ``frame``'s."""
    _, door = tools.link_pose(DOOR, "door")
    _, frame = tools.link_pose(DOOR, "frame")
    return wrap_to_pi(float(yaw_from_quat(np.asarray(door)) - yaw_from_quat(np.asarray(frame))))


def swing(point: np.ndarray, centre: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    """``point`` turned by ``angle`` [rad] about the line through ``centre`` along ``axis``."""
    v = np.asarray(point, dtype=np.float64) - centre
    return centre + transform_by_quat(v, _axis_angle(axis, angle))


def arc(
    point: np.ndarray, rot: tuple, centre: np.ndarray, axis: np.ndarray, angle: float, step: float
) -> list[tuple[np.ndarray, tuple]]:
    """Waypoints that carry a hand at ``point`` / ``rot`` about the axis by ``angle``, in equal steps of at most
    ``step``.

    Each is ``(position, effector rotation)``: the position swung about the line, the rotation turned
    with it. The chord of a 0.1 rad step on the door's 31 cm radius sags 0.4 mm off the arc.
    """
    count = max(1, math.ceil(abs(angle) / step - 1e-9))
    waypoints = []
    for k in range(1, count + 1):
        theta = angle * k / count
        waypoints.append((swing(point, centre, axis, theta), _turned(rot, axis, theta)))
    return waypoints


def _hook(tools: Tools, *, out: float) -> Generator[Motion, Outcome, tuple]:
    """Lower the open fingers either side of the lever, the palm onto its top; returns the hand's rotation.

    ``out`` moves the whole hand that much farther from the panel (the deliberate miss).
    """
    rng = tools.rng
    hub, along, outward, up = lever_frame(tools)
    rot = tools.aim(-math.sin(LEAN) * outward - math.cos(LEAN) * up, math.cos(LEAN) * outward - math.sin(LEAN) * up)
    hook = hub + rng.uniform(*HOOK_ALONG) * along + (HOOK_OFF_PANEL + out) * outward - HOOK_BELOW * up
    lined_up = hook + rng.uniform(*LINE_UP) * up
    here = tools.tool_pos()
    if here[2] < lined_up[2]:
        # Rise first: a retry starts beside the lever, and a straight line to above it crosses it.
        yield tools.move_to((here[0], here[1], lined_up[2]))
    # Turn the hand while it stands off the panel: turning near it swung the palm into it.
    yield tools.move_to(lined_up + rng.uniform(*STAND_OFF) * outward, rot=rot, timeout_s=3.0)
    yield tools.move_to(lined_up, rot=rot)
    yield tools.move_to(hook, rot=rot)
    return rot


def _turn_handle(tools: Tools, rot: tuple) -> Generator[Motion, Outcome, tuple]:
    """Press the lever down about the hub's axis by :data:`LATCH_TURN`, turning the hand with it.

    Returns the hand's rotation.
    """
    hub, along, outward, _ = lever_frame(tools)
    # The turn about the hub's axis (the panel normal) that lowers the lever's free end.
    sign = 1.0 if swing(hub + along, hub, outward, 0.1)[2] < hub[2] + along[2] else -1.0
    for point, rot in arc(tools.tool_pos(), rot, hub, outward, sign * LATCH_TURN, TURN_STEP):
        yield tools.move_to(point, rot=rot, speed=TURN_SPEED, tolerance=HELD_TOLERANCE, timeout_s=1.0)
    return rot


def _pull(tools: Tools, rot: tuple) -> Generator[Motion, Outcome, None]:
    """Swing the hand about the hinge by :data:`PULL_ANGLE`, turning it with the door."""
    vertical = np.array([0.0, 0.0, 1.0])
    for point, rot in arc(tools.tool_pos(), rot, hinge_point(tools), vertical, PULL_ANGLE, PULL_STEP):
        yield tools.move_to(point, rot=rot, speed=PULL_SPEED, tolerance=HELD_TOLERANCE, timeout_s=1.0)


def _lift_off(tools: Tools) -> Generator[Motion, Outcome, None]:
    """Lift the fingers off the lever the way they came on: along the lever's (turned) up."""
    _, _, _, up = lever_frame(tools)
    yield tools.move_by(LINE_UP[1] * up)


def _axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    """The wxyz quaternion of a turn by ``angle`` [rad] about ``axis`` (signed; zero is the identity)."""
    return axis_angle_to_quat(np.array(angle, dtype=np.float64), np.asarray(axis, dtype=np.float64))


def _turned(rot: tuple, axis: np.ndarray, angle: float) -> tuple:
    """The effector rotation ``rot`` turned by ``angle`` [rad] about the world ``axis``."""
    turned = transform_quat_by_quat(np.asarray(rot, dtype=np.float64), _axis_angle(axis, angle))
    return tuple(float(v) for v in turned)


def _rotate(quat: np.ndarray, vector) -> np.ndarray:
    return transform_by_quat(np.asarray(vector, dtype=np.float64), np.asarray(quat, dtype=np.float64))
