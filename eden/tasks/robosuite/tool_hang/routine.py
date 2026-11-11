"""A tool routine that assembles robosuite's ToolHang (``eden datagen --task robosuite_tool_hang``).

Written against :class:`~eden.extensions.scripted_demos.tools.Tools`, for one robot, as the
routines of :mod:`eden.extensions.scripted_demos.routines` are: pick the frame lying on the table,
stand it up with a tilted hand and lower its rod into the stand's socket; then pick the wrench,
turn it so its ``hole1`` ring faces along the frame's hook bar, hold it leaned over the bar's free
end and let it drop onto the bar.

Both objects are pinched from above and then turned with the hand (``move_to(rot=...)``). The
routine records the object's pose relative to the hand once it is lifted (:class:`Grasp`) and from
that solves for the hand rotation that holds the object in a target orientation
(:meth:`Grasp.hand_for`). Every placement that has to be precise is closed loop: the hand moves, the
routine reads where the object's site actually is, and moves by the remainder.

The poses come off the simulator (:meth:`~eden.extensions.scripted_demos.tools.Tools.link_pose`,
privileged), and the sites are placed with the asset offsets the task's success terms use
(:mod:`eden.tasks.robosuite.tool_hang.custom_terms`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from genesis.utils.geom import R_to_quat, R_to_rotvec, axis_angle_to_quat, quat_to_R

from eden.tasks.robosuite.tool_hang.custom_terms import FrameAssembled, ToolHoleOnBar

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

    from eden.extensions.scripted_demos.routines import Routine
    from eden.extensions.scripted_demos.tools import Motion, Outcome, Tools

#: Scene entities the routine reads.
FRAME, WRENCH, STAND = "frame", "tool", "stand"
DOWN = np.array((0.0, 0.0, -1.0))
UP = -DOWN

#: Where the frame is pinched, in its root frame (``tool_hang_frame.xml``): on the rod's axis in the
#: upper part of the 2.54 cm grip block, 2.2 cm above the frame's centre of mass (4.3 cm up the rod,
#: measured on the built asset). A pinch is held by the pads' friction: should the frame turn in it,
#: it swings rod-down from there rather than rod-up.
FRAME_GRIP = np.array((0.04375, 0.0, 0.065))
#: Half the grip block's width [m]: its top is this far above the rod's axis while the frame lies flat.
FRAME_GRIP_HALF = 0.0127
#: Frame sites, in its root frame: the rod's tip, the bar's free end and where the bar meets the rod.
FRAME_TIP = np.array(FrameAssembled.tip_offset)
BAR_FREE_END = np.array(ToolHoleOnBar.hang_offset)
BAR_ROD_END = np.array(ToolHoleOnBar.intersection_offset)
#: Stand sites, in its root frame: the socket's axis (the centre of its four walls), the walls' top and
#: the base slab's top.
SOCKET_XY = np.mean(np.array(FrameAssembled.wall_offsets)[:, :2], axis=0)
SOCKET_TOP_Z = 0.08
SLAB_TOP_Z = FrameAssembled.base_offset[2] + 0.005
#: Half the wrench's grip block height [m], and where it is pinched, in its root frame: 2 mm past its
#: centre of mass (2 mm off the root toward the hole1 ring), so that should it turn in the pinch, its
#: heavier half takes the handle down and the ring up — pinched on the ring's side of it, the ring
#: drooped a little on every correction until the wrench fell out.
WRENCH_GRIP_HALF = 0.015
WRENCH_PINCH = np.array((-0.004, 0.0, 0.0))
WRENCH_HOLE = np.array(ToolHoleOnBar.hole_offset)

#: The frame once upright, as world axes of its body axes (columns): rod (+z) up, the bar's free end
#: (-x) at heading :data:`BAR_FREE_END_HEADING` — toward the robot and its right. The hand, which
#: pinched it from above along its +y, then points level along :data:`FRAME_APPROACH` (heading 150
#: degrees). The wrench is let go with the hand pointing along the bar from its free end, the frame is
#: held with the hand across the bar, and of the headings measured (180, -150, -135, -120 and -90
#: degrees) -120 is the one where the Panda reaches both: the frame over the socket and backing off
#: it, and the wrench at its drop point. Toward +x (away from the robot) it cannot point back at
#: itself to reach the wrench's drop point; at -90 it cannot back off the frame's grip block.
BAR_FREE_END_HEADING = math.radians(-120.0)
FRAME_UPRIGHT = quat_to_R(axis_angle_to_quat(np.array(BAR_FREE_END_HEADING + math.pi), np.array((0.0, 0.0, 1.0))))
FRAME_APPROACH = np.cross(np.array((0.0, 0.0, 1.0)), FRAME_UPRIGHT[:, 0])

#: How long [s] the routine waits for the reset's objects to settle.
SETTLE_S = 0.5
#: Height [m] above an object's top the hand hovers before descending, drawn per pick.
HOVER_ABOVE = (0.05, 0.08)
#: How long [s] the hand pauses at the grasp pose before closing, drawn per pick.
GRASP_PAUSE = (0.2, 0.5)
#: Height [m] of the tool point while a held object is turned: the frame's rod hangs 20 cm below it.
TURN_Z = 1.17
#: How far [m] from the socket, back along the hand's level approach, the frame is turned upright.
FRAME_TURN_OFFSET = 0.10
#: Clearance [m] of the frame's tip over the socket's walls once carried over it; height over the rim it
#: is centred at before it goes in; how deep it goes in at first; the socket's inner half width (a tip
#: further off its axis has come down on the rim); and how many times the entry is tried.
SOCKET_CLEARANCE = 0.015
RIM_CLEARANCE = 0.005
SOCKET_ENTRY = 0.03
SOCKET_HALF = 0.0055
ENTRY_ATTEMPTS = 3
#: The rest of the way down goes in this many stages, the grip re-centred over the socket before each,
#: to this height [m] of the tip over the base slab, where the frame is let go.
INSERT_STAGES = 3
RELEASE_ABOVE_SLAB = 0.004
#: Speed [m/s] of a carry and of the insertion and threading.
CARRY_SPEED = 0.06
SLOW_SPEED = 0.03
#: Tolerance [m] of a fine placement, how many closed-loop corrections it gets, and how long [s] the
#: held object is left to settle after each.
FINE_TOLERANCE = 0.002
CORRECTIONS = 4
SETTLE_AFTER_MOVE_S = 0.4
#: A held object that ends a turn more than this [deg] off is re-measured in the hand and turned again,
#: up to :data:`REAIMS` times.
ROTATION_SLACK = 3.0
REAIMS = 2
#: How far [m] the hand backs out along its approach axis after letting the frame go (off its grip
#: block), then rises by as much, and where it goes, pointing down, between the two stages.
FRAME_RETREAT = 0.05
NEUTRAL_POS = (-0.12, -0.08, 1.12)
#: Picks of an object before the routine gives up on it.
MAX_PICK_ATTEMPTS = 3

#: Threading the ring over the bar's free end. The hook's lip stands 1.2 cm above the bar there, so the
#: bar and lip together are 19.5 mm tall, and the ring's hole is 19.4 mm across: a ring kept square to
#: the bar cannot pass over it. Leaned back by :data:`RING_LEAN` (its top toward the rod), the ring is
#: held over the bar's end with its top segment past the lip and its bottom segment still short of the
#: end, and let go: it drops onto the bar, swings upright, and its bottom passes under the bar.
#: :data:`RING_DROP` is where its top segment's centre is let go, along the bar from its free end and
#: above the bar's axis [m]. It is reached from above: coming in along the bar from short of its end
#: puts the hand, 9 cm behind the ring, too close to the robot's base to reach. Lowering it further
#: before letting go only lost it: held through a second placement, the wrench slipped out of the
#: fingers in half the episodes.
RING_LEAN = math.radians(62.0)
RING_DROP = (0.014, 0.04)
#: Distance [m] from the ring's centre to the middle of its top segment (``tool_hole1_hc_*``: 12.9 mm),
#: the bottom segment's centre (``tool_hole1_hc_4``) in the wrench's root frame, and how far short of
#: the bar's end it must be when the wrench is let go (its own half thickness and a margin).
RING_SEGMENT_RADIUS = 0.012934
WRENCH_BOTTOM_SEGMENT = WRENCH_HOLE + np.array((RING_SEGMENT_RADIUS, 0.0, 0.0))
BOTTOM_CLEARANCE = 0.004
#: How far [m] the open hand backs off the let-go wrench along its axis, then rises; and how long [s]
#: the wrench is given to swing onto the bar.
RELEASE_BACKOFF = 0.06
RELEASE_LIFT = 0.06
SWING_S = 1.0


def tool_hang_routine() -> Routine:
    """The routine for the tool hang: stand the frame in the socket, then hang the wrench on it."""

    def hang_tool(tools: Tools) -> Generator[Motion, Outcome, None]:
        yield tools.wait(SETTLE_S)
        if not (yield from _assemble_frame(tools)):
            return
        yield from _hang_wrench(tools)

    hang_tool.entity_names = (FRAME, WRENCH, STAND)
    return hang_tool


# --- geometry -------------------------------------------------------------------------------------


def _matrix(quat: np.ndarray) -> np.ndarray:
    return quat_to_R(np.asarray(quat, dtype=np.float64))


def _quat(rotation: np.ndarray) -> tuple[float, float, float, float]:
    return tuple(float(v) for v in R_to_quat(np.asarray(rotation, dtype=np.float64)))


def _point(tools: Tools, name: str, offset: np.ndarray) -> np.ndarray:
    """World position of ``offset`` in entity ``name``'s root frame."""
    pos, quat = tools.link_pose(name)
    return pos + _matrix(quat) @ offset


@dataclass(frozen=True)
class Grasp:
    """An object held rigidly in the hand.

    Its rotation and position relative to the tool point, in the effector frame.
    """

    name: str
    rotation: np.ndarray
    position: np.ndarray

    @classmethod
    def measure(cls, tools: Tools, name: str) -> Grasp:
        """Read the held ``name``'s pose against the hand's."""
        hand = _matrix(tools.tool_rotation())
        pos, quat = tools.link_pose(name)
        return cls(name, hand.T @ _matrix(quat), hand.T @ (pos - tools.tool_pos()))

    def hand_for(self, rotation: np.ndarray) -> np.ndarray:
        """The effector rotation that holds the object at world ``rotation``."""
        return rotation @ self.rotation.T

    def tool_for(self, hand: np.ndarray, site: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Where the tool point goes for the object's ``site`` (root frame) to reach ``target``.

        The hand is held at effector rotation ``hand``.
        """
        return target - hand @ (self.position + self.rotation @ site)


def wrench_hanging(along: np.ndarray) -> np.ndarray:
    """The wrench hanging on a bar that runs (level) along ``along``, from its free end toward the rod.

    Handle (+x) down, so the hole1 ring (-x) is on top; the ring's axis (+z) back along the bar. The
    hand, which pinched it from above along -z, then points along ``along``: in from the bar's free end.
    """
    ring_axis = -np.array((along[0], along[1], 0.0)) / np.linalg.norm(along[:2])
    return np.stack([DOWN, np.cross(ring_axis, DOWN), ring_axis], axis=1)


def lean(rotation: np.ndarray, toward: np.ndarray, angle: float) -> np.ndarray:
    """``rotation`` turned so the world up axis tips by ``angle`` toward the level direction ``toward``."""
    # Not ``axis_angle_to_R``: it takes the sine from the cosine, so a negative angle turns the positive way.
    return quat_to_R(axis_angle_to_quat(np.array(angle), np.cross(UP, toward))) @ rotation


def _closing_of(tools: Tools, hand: np.ndarray) -> np.ndarray:
    """World direction the fingers close along with the hand at effector rotation ``hand``."""
    fa = tools.finger_axis_yaw
    return hand @ np.array((math.cos(fa), math.sin(fa), 0.0))


# --- motions --------------------------------------------------------------------------------------


def _pick(tools: Tools, grip: np.ndarray, closing: np.ndarray, top_z: float) -> Generator[Motion, Outcome, bool]:
    """Pinch at ``grip`` from above, the fingers closing along the level direction ``closing``; lift to :data:`TURN_Z`.

    The closing direction is taken as given (``move_to(rot=...)``), not the half turn the wrist might
    prefer: which way round the object sits in the hand decides how it turns with it. ``top_z`` is
    the object's top, which the hand hovers over first.
    """
    rng = tools.rng
    rot = tools.aim(DOWN, closing)
    yield tools.open_gripper()
    yield tools.move_to((grip[0], grip[1], top_z + rng.uniform(*HOVER_ABOVE)), rot=rot)
    yield tools.move_to(grip, rot=rot, tolerance=0.005)
    yield tools.wait(rng.uniform(*GRASP_PAUSE))
    yield tools.close_gripper()
    yield tools.move_to((grip[0], grip[1], TURN_Z))
    return tools.holding()


def _pick_with_retries(
    tools: Tools, name: str, locate: Callable[[], tuple[np.ndarray, np.ndarray, float]]
) -> Generator[Motion, Outcome, Grasp | None]:
    """Pick ``name`` where ``locate()`` says (grip point, closing direction, top).

    Tries up to :data:`MAX_PICK_ATTEMPTS` times; returns ``None`` if every pick misses.
    """
    for attempt in range(MAX_PICK_ATTEMPTS):
        grip, closing, top_z = locate()
        if (yield from _pick(tools, grip, closing, top_z)):
            return Grasp.measure(tools, name)
        tools.note(f"{name}: pick {attempt + 1} missed (closure {tools.gripper_closure():.2f})")
        yield tools.open_gripper()
    tools.note(f"{name}: gave up after {MAX_PICK_ATTEMPTS} picks")
    return None


def _place_site_of(
    tools: Tools, site: Callable[[], np.ndarray], target: np.ndarray, *, speed: float | None = None
) -> Generator[Motion, Outcome, float]:
    """Move the hand (keeping its rotation) until ``site()`` is at ``target``, correcting from where it lands.

    Returns the remaining miss [m].
    """
    miss = target - site()
    for _ in range(1 + CORRECTIONS):
        if np.linalg.norm(miss) < FINE_TOLERANCE:
            break
        yield tools.move_by(miss, speed=speed, tolerance=FINE_TOLERANCE / 2)
        yield tools.wait(SETTLE_AFTER_MOVE_S)
        miss = target - site()
        speed = SLOW_SPEED
    return float(np.linalg.norm(miss))


def _hold_at(
    tools: Tools, grasp: Grasp, rotation: np.ndarray, site: np.ndarray, target: np.ndarray, *, speed: float
) -> Generator[Motion, Outcome, tuple[Grasp, float]]:
    """Turn the held object to ``rotation`` with its ``site`` at ``target``, correcting from where it lands.

    The object can turn in the pinch on the way (a long wrench swung through a quarter turn does): if it
    ends more than :data:`ROTATION_SLACK` off, its grasp is measured again and the hand re-aimed. Returns
    the grasp as last measured and the site's remaining miss [m].
    """
    for _ in range(1 + REAIMS):
        hand = grasp.hand_for(rotation)
        yield tools.move_to(grasp.tool_for(hand, site, target), rot=_quat(hand), speed=speed, timeout_s=4.0)
        yield tools.wait(SETTLE_AFTER_MOVE_S)
        if _angle_deg(_matrix(tools.link_pose(grasp.name)[1]), rotation) < ROTATION_SLACK:
            break
        grasp = Grasp.measure(tools, grasp.name)
        speed = SLOW_SPEED
    miss = yield from _place_site_of(tools, lambda: _point(tools, grasp.name, site), target, speed=SLOW_SPEED)
    return grasp, miss


def _locate_frame(tools: Tools) -> tuple[np.ndarray, np.ndarray, float]:
    pos, quat = tools.link_pose(FRAME)
    rot = _matrix(quat)
    grip = pos + rot @ FRAME_GRIP
    # Across the rod, along the bar (body x): the closing line the turn to FRAME_UPRIGHT keeps. Its
    # direction (against the body's x) is the way round of the two that leaves the arm's last joint
    # room for the quarter turn about the vertical that follows.
    return grip, -rot[:, 0], grip[2] + FRAME_GRIP_HALF


def _locate_wrench(tools: Tools) -> tuple[np.ndarray, np.ndarray, float]:
    pos, quat = tools.link_pose(WRENCH)
    rot = _matrix(quat)
    # Across the handle, along its +y: of the two ways round, the one the Panda reaches at the drop point.
    return pos + rot @ WRENCH_PINCH, rot[:, 1], pos[2] + WRENCH_GRIP_HALF


def _assemble_frame(tools: Tools) -> Generator[Motion, Outcome, bool]:
    stand_pos, stand_quat = tools.link_pose(STAND)
    socket = stand_pos + _matrix(stand_quat) @ np.array((*SOCKET_XY, 0.0))
    above = np.array((socket[0], socket[1], stand_pos[2] + SOCKET_TOP_Z + SOCKET_CLEARANCE))
    grasp = yield from _pick_with_retries(tools, FRAME, lambda: _locate_frame(tools))
    if grasp is None:
        return False
    # Carry it beside the socket, turn it about the vertical there with the hand still pointing down,
    # then tip the hand level about the closing line: the rod comes up straight, swinging clear of the
    # stand. (Turned where it was picked, the arm then could not carry it over the socket.)
    upright = grasp.hand_for(FRAME_UPRIGHT)
    closing = _closing_of(tools, upright)
    beside = np.array((socket[0], socket[1], TURN_Z)) - FRAME_TURN_OFFSET * FRAME_APPROACH
    yield tools.move_to(beside, speed=CARRY_SPEED)
    yield tools.move_to(tools.tool_pos(), rot=tools.aim(DOWN, closing), speed=CARRY_SPEED, timeout_s=4.0)
    yield tools.move_to(tools.tool_pos(), rot=_quat(upright), speed=CARRY_SPEED, timeout_s=4.0)
    yield tools.wait(SETTLE_AFTER_MOVE_S)
    tip = _point(tools, FRAME, FRAME_TIP)
    yield tools.move_to(tools.tool_pos() + (above - tip), speed=CARRY_SPEED)
    miss = yield from _place_site_of(tools, lambda: _point(tools, FRAME, FRAME_TIP), above)
    tools.note(
        f"frame: tip over the socket, {1000 * miss:.1f} mm off, rod {_tilt_deg(tools, FRAME):.1f} deg off vertical"
    )
    # Tip into the socket's mouth: centred just over its rim while it touches nothing, then straight
    # down — once a wall touches it, the frame turns on its pinch and a correction of the hand turns it
    # further. A tip that comes down on the rim instead is lifted and centred again.
    rim = np.array((socket[0], socket[1], stand_pos[2] + SOCKET_TOP_Z + RIM_CLEARANCE))
    for attempt in range(ENTRY_ATTEMPTS):
        yield from _place_site_of(tools, lambda: _point(tools, FRAME, FRAME_TIP), rim, speed=SLOW_SPEED)
        yield tools.move_by((0.0, 0.0, -(RIM_CLEARANCE + SOCKET_ENTRY)), speed=SLOW_SPEED, tolerance=FINE_TOLERANCE)
        tip = _point(tools, FRAME, FRAME_TIP)
        if tip[2] < rim[2] - RIM_CLEARANCE - SOCKET_ENTRY / 2 and np.linalg.norm(tip[:2] - socket[:2]) < SOCKET_HALF:
            break
        tools.note(f"frame: tip on the socket's rim, {1000 * np.linalg.norm(tip[:2] - socket[:2]):.0f} mm off its axis")
        yield tools.move_by((0.0, 0.0, RIM_CLEARANCE + SOCKET_ENTRY), speed=SLOW_SPEED)
    bottom_z = stand_pos[2] + SLAB_TOP_Z + RELEASE_ABOVE_SLAB
    start_z = _point(tools, FRAME, FRAME_TIP)[2]
    for stage in range(1, INSERT_STAGES + 1):
        # The grip over the socket's axis, so the walls stand the rod straight; then down.
        pinch_z = _point(tools, FRAME, FRAME_GRIP)[2]
        pinch_target = np.array((socket[0], socket[1], pinch_z))
        yield from _place_site_of(tools, lambda: _point(tools, FRAME, FRAME_GRIP), pinch_target, speed=SLOW_SPEED)
        tip_z = start_z + (bottom_z - start_z) * stage / INSERT_STAGES
        drop = tip_z - _point(tools, FRAME, FRAME_TIP)[2]
        yield tools.move_by((0.0, 0.0, drop), speed=SLOW_SPEED, tolerance=FINE_TOLERANCE)
    tools.note(f"frame: tip in the socket, rod {_tilt_deg(tools, FRAME):.1f} deg off vertical")
    yield tools.open_gripper()
    tip = _point(tools, FRAME, FRAME_TIP)
    tools.note(f"frame: let go, its tip {1000 * (tip[2] - bottom_z + RELEASE_ABOVE_SLAB):.0f} mm above the slab")
    base = stand_pos + _matrix(stand_quat) @ np.array(FrameAssembled.base_offset)
    seated = float(np.linalg.norm(tip - base)) < FrameAssembled.insertion_tolerance
    # Back off the grip block along the hand's axis, up, and point the hand down again over the middle
    # of the table: from the level hand a far, low target leaves the arm on its joint stops.
    yield tools.move_by(-FRAME_RETREAT * FRAME_APPROACH)
    yield tools.move_by(FRAME_RETREAT * UP)
    yield tools.move_to(tools.tool_pos(), rot=tools.aim(DOWN, closing), timeout_s=4.0)
    yield tools.move_to(NEUTRAL_POS, timeout_s=4.0)
    if not seated:
        tools.note("frame: jammed short of the slab; the wrench has nothing to hang on")
    return seated


def _hang_wrench(tools: Tools) -> Generator[Motion, Outcome, None]:
    grasp = yield from _pick_with_retries(tools, WRENCH, lambda: _locate_wrench(tools))
    if grasp is None:
        return
    free_end = _point(tools, FRAME, BAR_FREE_END)
    along = _point(tools, FRAME, BAR_ROD_END) - free_end
    along /= np.linalg.norm(along)
    leaned = lean(wrench_hanging(along), along, RING_LEAN)
    ring_up = math.cos(RING_LEAN) * UP + math.sin(RING_LEAN) * along
    # Turn it where it was lifted, clear of everything; it turns a little in the pinch, so measure it again.
    hand = grasp.hand_for(leaned)
    yield tools.move_to(tools.tool_pos(), rot=_quat(hand), speed=SLOW_SPEED, timeout_s=4.0)
    yield tools.wait(SETTLE_AFTER_MOVE_S)
    grasp = Grasp.measure(tools, WRENCH)
    centre = free_end + RING_DROP[0] * along + RING_DROP[1] * UP - RING_SEGMENT_RADIUS * ring_up
    grasp, miss = yield from _hold_at(tools, grasp, leaned, WRENCH_HOLE, centre, speed=CARRY_SPEED)
    if not tools.holding():
        tools.note("wrench: slipped out of the fingers on the way over the bar")
        return
    tools.note(f"wrench: ring over the bar's end, {1000 * miss:.1f} mm off")
    for _ in range(REAIMS):
        # Its bottom segment must still be short of the bar's end, or it lands on the end, not under it.
        bottom = float((_point(tools, WRENCH, WRENCH_BOTTOM_SEGMENT) - free_end) @ along)
        if bottom < -BOTTOM_CLEARANCE:
            break
        tools.note(
            f"wrench: the ring's bottom is {1000 * bottom:.0f} mm along the bar, not short of its end; turning it back"
        )
        grasp = Grasp.measure(tools, WRENCH)
        grasp, _ = yield from _hold_at(tools, grasp, leaned, WRENCH_HOLE, centre, speed=SLOW_SPEED)
    bottom = float((_point(tools, WRENCH, WRENCH_BOTTOM_SEGMENT) - free_end) @ along)
    tools.note(f"wrench: let go, the ring's bottom {1000 * bottom:.0f} mm along the bar")
    yield tools.open_gripper()
    approach = _matrix(tools.tool_rotation()) @ DOWN
    yield tools.move_by(-RELEASE_BACKOFF * approach, speed=CARRY_SPEED)
    yield tools.move_by(RELEASE_LIFT * UP, speed=CARRY_SPEED)
    yield tools.wait(SWING_S)


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    """Angle [deg] of the rotation between rotation matrices ``a`` and ``b``."""
    return math.degrees(np.linalg.norm(R_to_rotvec(a.T @ b)))


def _tilt_deg(tools: Tools, name: str) -> float:
    """Angle [deg] of entity ``name``'s +z axis off vertical."""
    _, quat = tools.link_pose(name)
    return math.degrees(math.acos(min(1.0, _matrix(quat)[2, 2])))
