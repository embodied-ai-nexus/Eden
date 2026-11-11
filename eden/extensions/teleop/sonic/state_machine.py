"""Operator state machine for SONIC teleoperation (pure logic, no I/O).

Port of the reference manager's mode table. Two chains share the VR_3PT
sub-mode, which remembers its parent:

    chain 1:  POSE <--(B+Y)--> PLANNER_FROZEN_UPPER_BODY <--(L-stick click)--> PLANNER_VR_3PT
    chain 2:  POSE <--(A+X)--> PLANNER                   <--(L-stick click)--> PLANNER_VR_3PT
    VR_3PT --(A+X or B+Y)--> POSE;  A+B+X+Y engages from OFF / disengages from anywhere
    POSE --(menu held)--> POSE_PAUSE --(menu released)--> POSE  (level-triggered)

Button combos are rising-edge except the menu pause. Each transition carries
the side effects the manager must run before streaming the new mode's first
tick, plus the ``command`` message to send after it.

Disengaging is an Eden deviation: the reference's A+B+X+Y from a live mode
sends ``stop``, which ends the robot process — on hardware a damping-only
command, and an unsupported robot falls. Here it hands the robot to the
planner's IDLE instead, so it keeps balancing in place; the robot session
ends only from the robot side.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class StreamMode(Enum):
    OFF = 0
    POSE = 1
    PLANNER = 2
    PLANNER_FROZEN_UPPER_BODY = 3
    POSE_PAUSE = 4
    PLANNER_VR_3PT = 5


PLANNER_MODES = (StreamMode.PLANNER, StreamMode.PLANNER_FROZEN_UPPER_BODY, StreamMode.PLANNER_VR_3PT)


class Effect(Enum):
    """Side effects to run before the new mode's first streamed tick."""

    CALIBRATE_FULL = "calibrate_full"  # every engage (OFF -> PLANNER): zero-pose calibration
    RECALIBRATE_WRISTS = "recalibrate_wrists"  # every VR_3PT entry
    RESET_POSE_YAW = "reset_pose_yaw"
    RESET_PLANNER_YAW = "reset_planner_yaw"
    SAVE_FROZEN_TARGETS = "save_frozen_targets"  # grab measured upper body via g1_debug
    POSE_EXIT = "pose_exit"  # clear the pose stream window
    RELEASE_TO_PLANNER = "release_to_planner"  # disengage: the robot idles on the planner


@dataclass
class ButtonSnapshot:
    """Controller levels for one manager tick (A/B = right, X/Y = left)."""

    a: bool = False
    b: bool = False
    x: bool = False
    y: bool = False
    left_axis_click: bool = False
    left_menu: bool = False


@dataclass
class Transition:
    """Outcome of one state-machine step."""

    previous: StreamMode
    mode: StreamMode
    changed: bool
    effects: list[Effect] = field(default_factory=list)
    # Command to send AFTER the new mode's first tick (reference ordering):
    # ("start", planner: bool), or None.
    command: tuple[str, bool] | None = None


class OperatorStateMachine:
    """Rising-edge mode table with VR_3PT parent tracking."""

    def __init__(self) -> None:
        self.mode = StreamMode.OFF
        self.vr3pt_parent = StreamMode.PLANNER
        self._prev_ax = False
        self._prev_by = False
        self._prev_start = False
        self._prev_stick = False

    def step(self, buttons: ButtonSnapshot) -> Transition:
        """Advance one tick; returns the (possibly unchanged) mode and its effects."""
        ax = buttons.a and buttons.x
        by = buttons.b and buttons.y
        start_combo = buttons.a and buttons.b and buttons.x and buttons.y
        stick = buttons.left_axis_click

        ax_edge = ax and not self._prev_ax
        by_edge = by and not self._prev_by
        start_edge = start_combo and not self._prev_start
        stick_edge = stick and not self._prev_stick

        current = self.mode
        new_mode = current
        if current == StreamMode.OFF:
            if start_edge:
                new_mode = StreamMode.PLANNER
        elif current == StreamMode.PLANNER:
            if start_edge:
                new_mode = StreamMode.OFF
            elif ax_edge:
                new_mode = StreamMode.POSE
            elif stick_edge:
                new_mode = StreamMode.PLANNER_VR_3PT
        elif current == StreamMode.POSE:
            if start_edge:
                new_mode = StreamMode.OFF
            elif ax_edge:
                new_mode = StreamMode.PLANNER
            elif by_edge:
                new_mode = StreamMode.PLANNER_FROZEN_UPPER_BODY
            elif buttons.left_menu:  # level, not edge
                new_mode = StreamMode.POSE_PAUSE
        elif current == StreamMode.PLANNER_FROZEN_UPPER_BODY:
            if start_edge:
                new_mode = StreamMode.OFF
            elif by_edge:
                new_mode = StreamMode.POSE
            elif stick_edge:
                new_mode = StreamMode.PLANNER_VR_3PT
        elif current == StreamMode.POSE_PAUSE:
            if start_edge:
                new_mode = StreamMode.OFF
            elif not buttons.left_menu:
                new_mode = StreamMode.POSE
        elif current == StreamMode.PLANNER_VR_3PT:
            if start_edge:
                new_mode = StreamMode.OFF
            elif stick_edge:
                new_mode = self.vr3pt_parent
            elif ax_edge or by_edge:
                new_mode = StreamMode.POSE

        self._prev_ax = ax
        self._prev_by = by
        self._prev_start = start_combo
        self._prev_stick = stick

        transition = Transition(previous=current, mode=new_mode, changed=new_mode != current)
        if transition.changed:
            self._collect_effects(transition)
        self.mode = new_mode
        return transition

    def force_off(self) -> Transition:
        """External force-off (robot fall auto-reset, input loss): back to OFF, no effects but the pose exit.

        The caller decides what the robot needs: nothing after a fall auto-reset
        (it already reset itself), a planner hand-off after an input loss. The
        next A+B+X+Y re-engages through the normal OFF -> PLANNER edge with full
        calibration.
        """
        previous = self.mode
        self.mode = StreamMode.OFF
        transition = Transition(previous=previous, mode=StreamMode.OFF, changed=previous != StreamMode.OFF)
        if previous in (StreamMode.POSE, StreamMode.POSE_PAUSE):
            transition.effects.append(Effect.POSE_EXIT)
        return transition

    def _collect_effects(self, t: Transition) -> None:
        current, new_mode = t.previous, t.mode
        if current == StreamMode.POSE:
            t.effects.append(Effect.POSE_EXIT)
        if new_mode == StreamMode.PLANNER_VR_3PT:
            self.vr3pt_parent = current

        if current == StreamMode.OFF and new_mode == StreamMode.PLANNER:
            t.effects.append(Effect.CALIBRATE_FULL)

        if new_mode == StreamMode.POSE:
            t.effects.append(Effect.RESET_POSE_YAW)
        elif new_mode == StreamMode.PLANNER and current != StreamMode.PLANNER_VR_3PT:
            # Returning from the VR_3PT sub-mode keeps the accumulated heading.
            t.effects.append(Effect.RESET_PLANNER_YAW)
        elif new_mode == StreamMode.PLANNER_FROZEN_UPPER_BODY:
            if current != StreamMode.PLANNER_VR_3PT:
                t.effects.append(Effect.RESET_PLANNER_YAW)
            # Always re-grab the frozen targets — after VR_3PT the old ones are stale.
            t.effects.append(Effect.SAVE_FROZEN_TARGETS)
        elif new_mode == StreamMode.PLANNER_VR_3PT:
            t.effects.append(Effect.RECALIBRATE_WRISTS)

        if new_mode == StreamMode.OFF:
            t.effects.append(Effect.RELEASE_TO_PLANNER)
        elif new_mode in PLANNER_MODES:
            t.command = ("start", True)
        elif new_mode == StreamMode.POSE:
            t.command = ("start", False)
