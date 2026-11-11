"""Robot tools: the calls a scripted routine makes — motions it waits on, and readings it takes at once.

A routine (:mod:`~eden.extensions.scripted_demos.routines`) is written for one environment, as a
plain Python generator, against a :class:`Tools` handle:

* **Motions** (:meth:`Tools.move_to`, :meth:`Tools.move_until_contact`, :meth:`Tools.close_gripper`,
  ...) return a :class:`Motion`. The routine yields it and gets an :class:`Outcome` back once
  the motion has ended — reached, timed out, stopped by contact. Every env's motion runs batched on
  the env device, so a routine reads like a script for one robot and runs on thousands.
* **Readings** return at once. The robot's own state (:meth:`Tools.tool_pos`, :meth:`Tools.holding`)
  is always available. What it knows about objects comes either from the simulator
  (:meth:`Tools.object_footprint`, :meth:`Tools.grasped`, :meth:`Tools.link_pose` — *privileged*) or from a camera
  (:meth:`Tools.locate`). A routine run with ``privileged=False`` is refused the first kind, by name,
  so a camera-only demonstrator cannot read the simulator state by accident.

Positions are world-frame metres. Headings are effector headings: 0 is the rig's straight-down pose
with heading 0 (:attr:`~eden.extensions.scripted_demos.rig.RigDriver.down_quat`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING

import numpy as np
import torch

from eden.utils.geom import R_to_quat

if TYPE_CHECKING:
    from collections.abc import Sequence

    from eden.extensions.scripted_demos.routines import RoutineDemonstrator


class MotionKind(IntEnum):
    """What a :class:`Motion` does; the executor dispatches on it, batched."""

    MOVE = 0
    GUARDED = 1
    GRIPPER = 2
    WAIT = 3


#: Outcome statuses. A motion that did what it was asked ends ``REACHED`` (a move), ``CONTACT`` (a
#: guarded move that touched something) or ``DONE`` (gripper, wait); the others are how it can fail.
REACHED, CONTACT, DONE = "reached", "contact", "done"
TIMEOUT, BLOCKED, NO_CONTACT = "timeout", "blocked", "no_contact"
#: Status codes the executor writes per env; :data:`STATUSES` names them.
STATUSES = (REACHED, CONTACT, DONE, TIMEOUT, BLOCKED, NO_CONTACT)


@dataclass(frozen=True)
class Motion:
    """One motion a routine yields. Build it with a :class:`Tools` method, not by hand.

    Attributes
    ----------
    kind : MotionKind
        What the executor does with it.
    label : str
        How the trace names it (``move_to(0.10, -0.25, 0.95)``).
    target : tuple[float, float, float] | None
        Tool-point position to reach (``MOVE``).
    yaw : float | None
        Effector heading to reach (``MOVE``); ``None`` keeps the current one.
    rot : tuple[float, float, float, float] | None
        Full effector rotation to reach (``MOVE``), wxyz relative to the straight-down pose — a tilted
        hand; set instead of ``yaw``.
    direction : tuple[float, float, float] | None
        Unit direction of a guarded move.
    distance : float
        How far [m] a guarded move goes before giving up.
    speed : float
        Peak tool speed [m/s] of a move, or the constant speed of a guarded one.
    tolerance : float
        A move has reached its target once the tool point is this close [m].
    duration_s : float
        How long a gripper motion or a wait lasts [s].
    timeout_s : float
        How long past its own duration a move may run before it ends ``TIMEOUT`` [s].
    grasp : float | None
        Gripper value from this motion on, 0 open / 1 closed; ``None`` leaves it.
    """

    kind: MotionKind
    label: str
    target: tuple[float, float, float] | None = None
    yaw: float | None = None
    rot: tuple[float, float, float, float] | None = None
    direction: tuple[float, float, float] | None = None
    distance: float = 0.0
    speed: float = 0.0
    tolerance: float = 0.01
    duration_s: float = 0.0
    timeout_s: float = 2.0
    grasp: float | None = None


@dataclass(frozen=True)
class Outcome:
    """What a motion reports when it ends — the value of the routine's ``yield``.

    Attributes
    ----------
    motion : str
        The motion's label.
    status : str
        One of :data:`STATUSES`.
    elapsed_s : float
        How long the motion ran [s].
    error_m : float
        Distance [m] from the tool point to the motion's target when it ended (0 for gripper / wait).
    """

    motion: str
    status: str
    elapsed_s: float
    error_m: float

    @property
    def ok(self) -> bool:
        """The motion did what it was asked: a move reached, a guarded move touched, a gripper / wait finished."""
        return self.status in (REACHED, CONTACT, DONE)


#: Height band [m] below an object's top whose points give :attr:`Footprint.top_center`.
TOP_BAND = 0.02


@dataclass(frozen=True)
class Footprint:
    """Where an object is and how it lies, from points on it — its collision mesh, or what a camera saw of it.

    Attributes
    ----------
    center : np.ndarray
        ``(3,)``: the centre of the horizontal footprint, at mid-height.
    yaw : float
        Heading [rad] of the footprint's long axis, in ``(-pi/2, pi/2]``.
    half_length, half_width : float
        Half extents [m] along the long and the short horizontal axis.
    top_z, bottom_z : float
        Highest and lowest point [m].
    top_center : np.ndarray
        ``(3,)``: centroid of the points within :data:`TOP_BAND` of the top — where a hand coming from
        above meets the object, which is off :attr:`center` when the object leans.
    num_points : int
        How many points it was measured from.
    centroid : np.ndarray | None
        ``(2,)``: mean xy of the points — off :attr:`center` toward where they crowd, the body of a mug
        rather than its handle (:func:`~eden.extensions.scripted_demos.put_on_routine.body_centre`).
    """

    center: np.ndarray
    yaw: float
    half_length: float
    half_width: float
    top_z: float
    bottom_z: float
    top_center: np.ndarray
    num_points: int
    centroid: np.ndarray | None = None

    @property
    def height(self) -> float:
        return self.top_z - self.bottom_z


def footprint_from_points(points: np.ndarray) -> Footprint:
    """The :class:`Footprint` of ``(P, 3)`` points: principal horizontal axes, and the extents along them.

    Raises
    ------
    ValueError
        Fewer than 3 points — nothing to measure an orientation from.
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(points) < 3:
        raise ValueError(f"A footprint needs at least 3 points to measure; got {len(points)}.")
    xy = points[:, :2]
    mean = xy.mean(axis=0)
    _, _, vt = np.linalg.svd(xy - mean, full_matrices=False)
    long_axis = vt[0]
    yaw = math.atan2(long_axis[1], long_axis[0])
    # Fold into (-pi/2, pi/2]: the long axis has no direction.
    if yaw <= -math.pi / 2:
        yaw += math.pi
    elif yaw > math.pi / 2:
        yaw -= math.pi
    axes = np.array([[math.cos(yaw), math.sin(yaw)], [-math.sin(yaw), math.cos(yaw)]])
    local = (xy - mean) @ axes.T
    low, high = local.min(axis=0), local.max(axis=0)
    if high[1] - low[1] > high[0] - low[0]:
        # The axis of most variance need not be the longest (uneven vertices, a partial camera view):
        # the long axis is the one of larger extent.
        yaw = yaw + math.pi / 2 if yaw <= 0 else yaw - math.pi / 2
        axes = np.array([[math.cos(yaw), math.sin(yaw)], [-math.sin(yaw), math.cos(yaw)]])
        local = (xy - mean) @ axes.T
        low, high = local.min(axis=0), local.max(axis=0)
    centre_xy = mean + ((low + high) / 2) @ axes
    top, bottom = float(points[:, 2].max()), float(points[:, 2].min())
    half = (high - low) / 2
    top_center = points[points[:, 2] >= top - TOP_BAND].mean(axis=0)
    return Footprint(
        center=np.array([centre_xy[0], centre_xy[1], (top + bottom) / 2]),
        yaw=yaw,
        half_length=float(half[0]),
        half_width=float(half[1]),
        top_z=top,
        bottom_z=bottom,
        top_center=top_center,
        num_points=len(points),
        centroid=mean,
    )


class PrivilegedToolError(PermissionError):
    """A camera-only routine (``privileged=False``) called a tool that reads the simulator's state."""


class Tools:
    """The tools one environment's routine calls. Handed to the routine by the executor; one per env.

    Parameters
    ----------
    executor : RoutineDemonstrator
        Runs the motions and serves the readings, batched over envs.
    env_id : int
        The environment this handle speaks for.
    rng : np.random.Generator
        This env's random stream, re-seeded per episode: draw a routine's variations from it.
    """

    def __init__(self, executor: RoutineDemonstrator, env_id: int, rng: np.random.Generator):
        self._executor = executor
        self.env_id = env_id
        self.rng = rng

    # --- facts about the rig and the episode ------------------------------------------------------

    @property
    def privileged(self) -> bool:
        """Whether this routine may read the simulator's state (:meth:`object_footprint`, :meth:`grasped`,
        :meth:`link_pose`).
        """
        return self._executor.privileged

    @property
    def speed(self) -> float:
        """This episode's peak tool speed [m/s]: the demonstrator's ``peak_speed`` times a per-episode draw."""
        return self._executor.episode_speed(self.env_id)

    @property
    def tip_reach(self) -> float:
        """How far [m] the fingertips reach past the tool point (the pad midpoint) along the approach axis."""
        return self._executor.driver.tip_reach

    @property
    def finger_axis_yaw(self) -> float:
        """Heading [rad] the fingers close along, relative to the effector heading (modulo pi)."""
        return self._executor.driver.finger_axis_yaw

    @property
    def time_s(self) -> float:
        """Time [s] since this env's episode started."""
        return self._executor.episode_time(self.env_id)

    @property
    def shared(self) -> dict:
        """A dict this env's routine keeps for the episode, shared with the other arm's routine.

        Under a :class:`~eden.extensions.scripted_demos.multi_arm.MultiArmDemonstrator` both routines of
        an env get the same one, which is how two arms hand off: one writes a key, the other waits for
        it (``while not tools.shared.get(key): yield tools.wait(0.1)``).
        """
        return self._executor.blackboard(self.env_id)

    def note(self, text: str) -> None:
        """Add a line to this env's trace (:meth:`RoutineDemonstrator.trace`): what the routine decided and why."""
        self._executor.add_note(self.env_id, text)

    # --- the robot's own state: always available ------------------------------------------------

    def tool_pos(self) -> np.ndarray:
        """``(3,)`` position of the tool point (the pad midpoint)."""
        return self._executor.read("tool_pos")[self.env_id]

    def tool_yaw(self) -> float:
        """Current effector heading [rad]."""
        return float(self._executor.read("tool_yaw")[self.env_id])

    def heading_window(self) -> tuple[float, float]:
        """Lowest and highest effector heading [rad] the arm's last joint can turn the hand to from here.

        Linearized about the current pose (:meth:`~eden.extensions.scripted_demos.rig.RigDriver.yaw_window`);
        re-read it after a move. A routine about to turn something it holds by a quarter turn reads it
        to pick the direction there is room for — :meth:`move_to` would otherwise realize a turn past
        the joint's stop as the equivalent turn the other way, which a two-finger grasp allows but the
        held object does not.
        """
        low, high = self._executor.read("yaw_window")[self.env_id]
        return float(low), float(high)

    def tool_rotation(self) -> np.ndarray:
        """``(4,)`` current effector rotation, wxyz relative to the straight-down pose — what ``move_to(rot=...)``
        asks for.

        The whole rotation, where :meth:`tool_yaw` is its heading only: what a held object's pose is
        measured against to turn it with the hand.
        """
        return self._executor.read("tool_rot")[self.env_id]

    def gripper_closure(self) -> float:
        """How far the fingers are from open (0) toward fully closed (1)."""
        return float(self._executor.read("gripper_closure")[self.env_id])

    def holding(self) -> bool:
        """The fingers are commanded closed and stopped short of closing: something is between them.

        Proprioceptive — what the robot feels, not what the simulator knows — so camera-only routines
        may call it. Ask it after :meth:`close_gripper` has finished.
        """
        closure = self.gripper_closure()
        return self._executor.grasp_command(self.env_id) > 0.5 and closure < self._executor.holding_closure

    # --- object state from the simulator: privileged --------------------------------------------

    def object_footprint(self, name: str) -> Footprint:
        """Footprint of entity ``name`` from its collision geometry (privileged)."""
        self._require_privileged("object_footprint", f"locate('{name}', camera=...)")
        return self._executor.object_footprint(name, self.env_id)

    def grasped(self, name: str) -> bool:
        """Fingers on both sides of the closing line touch entity ``name`` (privileged)."""
        self._require_privileged("grasped", "holding()")
        return bool(self._executor.grasped(name)[self.env_id])

    def link_pose(self, name: str, link: str | None = None) -> tuple[np.ndarray, np.ndarray]:
        """World position ``(3,)`` and wxyz orientation ``(4,)`` of entity ``name``'s ``link`` (privileged).

        ``None`` reads the entity's root. Poses are in the frame the asset was authored in
        (``relative=True``), so a point given in an MJCF body's frame — a site — is
        ``pos + R(quat) @ offset``. For articulated fixtures (a door's panel and handle) whose pose a
        footprint cannot tell.
        """
        self._require_privileged("link_pose", f"locate('{name}', camera=...)")
        return self._executor.link_pose(name, link, self.env_id)

    # --- object state from a camera ---------------------------------------------------------------

    def locate(self, name: str, camera: str | None = None) -> Footprint | None:
        """Footprint of what ``camera`` sees of entity ``name``; ``None`` if it sees too little of it.

        The camera's segmentation stands in for a perfect text-prompted segmenter: the points are the
        depth of the pixels showing ``name``, so an occluded side or a hidden object reads as it would
        to a real camera. ``camera`` defaults to the demonstrator's first camera.
        """
        return self._executor.locate(name, camera, self.env_id)

    # --- motions: yield them -------------------------------------------------------------------

    def move_to(
        self,
        pos: Sequence[float],
        yaw: float | None = None,
        *,
        rot: Sequence[float] | None = None,
        speed: float | None = None,
        tolerance: float = 0.01,
        timeout_s: float = 2.0,
    ) -> Motion:
        """Move the tool point to ``pos`` (and heading ``yaw``) along a straight, minimum-jerk path.

        ``yaw`` is turned by a half turn (a two-finger gripper's symmetry) into the range the arm's
        last joint can reach. ``rot`` instead asks for a full effector rotation (wxyz, relative to the
        straight-down pose; :meth:`aim` builds one), reached as given. Ends ``REACHED`` within
        ``tolerance`` [m], or ``TIMEOUT`` ``timeout_s`` after the path's own duration.
        """
        if tolerance <= 0 or timeout_s < 0:
            raise ValueError(f"move_to needs tolerance > 0 and timeout_s >= 0, got {tolerance} and {timeout_s}.")
        if yaw is not None and rot is not None:
            raise ValueError(f"move_to takes a heading (yaw={yaw}) or a full rotation (rot={tuple(rot)}), not both.")
        target = _vector(pos, "pos")
        rotation = None if rot is None else _quaternion(rot)
        label = f"move_to({_fmt(target)}" + (f", yaw={yaw:.2f}" if yaw is not None else "")
        label += (f", rot=({_fmt(rotation)})" if rotation is not None else "") + ")"
        return Motion(
            kind=MotionKind.MOVE,
            label=label,
            target=target,
            yaw=None if yaw is None else float(yaw),
            rot=rotation,
            speed=self._speed(speed),
            tolerance=tolerance,
            timeout_s=timeout_s,
        )

    def aim(self, approach: Sequence[float], closing: Sequence[float]) -> tuple[float, float, float, float]:
        """The effector rotation (for :meth:`move_to`'s ``rot``) pointing the hand along ``approach`` and closing
        along ``closing``.

        ``closing`` is made perpendicular to ``approach``; both are world-frame directions.
        """
        return aim_rotation(approach, closing, self.finger_axis_yaw)

    def move_by(self, offset: Sequence[float], *, speed: float | None = None, tolerance: float = 0.01) -> Motion:
        """Move the tool point by ``offset`` from where it is now, keeping its heading."""
        target = tuple(float(p + o) for p, o in zip(self.tool_pos(), _vector(offset, "offset")))
        motion = self.move_to(target, speed=speed, tolerance=tolerance)
        return Motion(**{**motion.__dict__, "label": f"move_by({_fmt(_vector(offset, 'offset'))})"})

    def move_until_contact(
        self, direction: Sequence[float] = (0.0, 0.0, -1.0), max_distance: float = 0.1, *, speed: float = 0.03
    ) -> Motion:
        """Move the tool point along ``direction`` at ``speed`` until a finger touches something.

        Ends ``CONTACT`` when a finger link newly presses on anything that is not the robot (a touch
        already there when it starts — a held object — does not count; lowering a held object until it
        lands ends ``BLOCKED``), ``BLOCKED`` when the hand stops short of the command without a finger
        touching (the palm, the arm, the workspace floor), ``NO_CONTACT`` after ``max_distance`` [m]
        with nothing touched.
        """
        d = np.asarray(_vector(direction, "direction"))
        norm = float(np.linalg.norm(d))
        if norm < 1e-9:
            raise ValueError(f"move_until_contact needs a non-zero direction, got {tuple(direction)}.")
        if max_distance <= 0 or speed <= 0:
            raise ValueError(f"move_until_contact needs max_distance > 0 and speed > 0, got {max_distance}, {speed}.")
        unit = tuple(float(v) for v in d / norm)
        return Motion(
            kind=MotionKind.GUARDED,
            label=f"move_until_contact({_fmt(unit)}, max {max_distance:.3f} m)",
            direction=unit,
            distance=float(max_distance),
            speed=float(speed),
        )

    def close_gripper(self, duration_s: float = 0.6) -> Motion:
        """Close the fingers and hold the arm still for ``duration_s`` [s]."""
        return Motion(kind=MotionKind.GRIPPER, label="close_gripper()", duration_s=_positive(duration_s), grasp=1.0)

    def open_gripper(self, duration_s: float = 0.5) -> Motion:
        """Open the fingers and hold the arm still for ``duration_s`` [s]."""
        return Motion(kind=MotionKind.GRIPPER, label="open_gripper()", duration_s=_positive(duration_s), grasp=0.0)

    def wait(self, duration_s: float) -> Motion:
        """Hold the arm still for ``duration_s`` [s]."""
        return Motion(kind=MotionKind.WAIT, label=f"wait({duration_s:.2f})", duration_s=_positive(duration_s))

    # --- helpers ----------------------------------------------------------------------------------

    def _speed(self, speed: float | None) -> float:
        if speed is not None and speed <= 0:
            raise ValueError(f"speed must be positive [m/s], got {speed}.")
        return self.speed if speed is None else float(speed)

    def _require_privileged(self, tool: str, alternative: str) -> None:
        if not self.privileged:
            raise PrivilegedToolError(
                f"{tool}() reads the simulator's state, which this camera-only routine (privileged=False) may not. "
                f"Measure it instead: {alternative}. Do not flip privileged=True to get past this — the "
                f"demonstrations would then rely on state a deployed robot does not have."
            )


def aim_rotation(
    approach: Sequence[float], closing: Sequence[float], finger_axis_yaw: float
) -> tuple[float, float, float, float]:
    """Effector rotation (wxyz, relative to the straight-down pose) whose approach axis is ``approach`` and closing
    line ``closing``.

    In the straight-down pose with heading 0 the approach axis is world ``-z`` and the fingers close
    along heading ``finger_axis_yaw``; the rotation maps that pair onto the requested one.

    Raises
    ------
    ValueError
        A zero direction, or ``closing`` parallel to ``approach``.
    """
    a = np.asarray(_vector(approach, "approach"))
    c = np.asarray(_vector(closing, "closing"))
    if np.linalg.norm(a) < 1e-9:
        raise ValueError(f"aim needs a non-zero approach direction, got {tuple(approach)}.")
    a = a / np.linalg.norm(a)
    c = c - (c @ a) * a
    if np.linalg.norm(c) < 1e-6:
        raise ValueError(f"aim's closing direction {tuple(closing)} is parallel to its approach {tuple(approach)}.")
    c = c / np.linalg.norm(c)
    source_a = np.array([0.0, 0.0, -1.0])
    source_c = np.array([math.cos(finger_axis_yaw), math.sin(finger_axis_yaw), 0.0])
    source = np.stack([source_a, source_c, np.cross(source_a, source_c)], axis=1)
    target = np.stack([a, c, np.cross(a, c)], axis=1)
    rotation = torch.as_tensor(target @ source.T, dtype=torch.float64)
    return _quaternion(R_to_quat(rotation).tolist())


def _quaternion(values: Sequence[float]) -> tuple[float, float, float, float]:
    q = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(q) != 4 or not np.linalg.norm(q) > 1e-9:
        raise ValueError(f"A rotation is a non-zero wxyz quaternion of 4 components, got {tuple(q)}.")
    q = q / np.linalg.norm(q)
    return tuple(float(v) for v in q)


def _positive(duration_s: float) -> float:
    if not duration_s > 0:
        raise ValueError(f"A gripper motion or a wait needs a positive duration_s [s], got {duration_s}.")
    return float(duration_s)


def _vector(values: Sequence[float], what: str) -> tuple[float, float, float]:
    values = tuple(float(v) for v in np.asarray(values, dtype=np.float64).reshape(-1))
    if len(values) != 3:
        raise ValueError(f"{what} must have 3 components (x, y, z), got {len(values)}: {values}.")
    return values


def _fmt(values: Sequence[float]) -> str:
    return ", ".join(f"{v:.3f}" for v in values)
