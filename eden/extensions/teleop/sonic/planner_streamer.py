"""Planner-mode streamer: joystick -> locomotion commands (~20 Hz).

Port of the reference ``PlannerStreamer``. The left stick sets the movement
direction/speed (deadzone-rescaled, rotated into the world by the accumulated
facing), the right stick turns, and A+B / X+Y cycle the locomotion mode.
PLANNER_FROZEN_UPPER_BODY attaches the frozen upper-body targets grabbed from
the robot's feedback; PLANNER_VR_3PT attaches the calibrated 3-point pose and
trigger-driven hand targets.
"""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np

from eden.extensions.policies.sonic.planner import LocomotionMode
from eden.extensions.policies.sonic.wire import build_planner_message
from eden.extensions.teleop.base import TeleopStreamer
from eden.extensions.teleop.sonic.feedback import FeedbackReader
from eden.extensions.teleop.sonic.hands import hand_targets_from_triggers
from eden.extensions.teleop.sonic.inputs import BodySample, ControllerSample
from eden.extensions.teleop.sonic.pose_streamer import IDLE_POLL_S, JOYSTICK_DEADZONE, YawAccumulator
from eden.extensions.teleop.sonic.state_machine import StreamMode
from eden.extensions.teleop.sonic.three_point import ThreePointPose

_FEEDBACK = "robot feedback (g1_debug)"
_FEEDBACK_HINT = "check that `eden teleop sonic robot` is running and --robot-host / --feedback-port point at it"


class PlannerStreamer(TeleopStreamer):
    """One planner-mode streaming iteration per manager tick."""

    def __init__(
        self,
        send: Callable[[bytes], None],
        three_point: ThreePointPose,
        feedback: FeedbackReader,
        poll_hz: int = 20,
        on_frame: Callable[[np.ndarray, np.ndarray | None], None] | None = None,
    ) -> None:
        self._send = send
        self._three_point = three_point
        self.feedback = feedback
        self._on_frame = on_frame
        self.dt = 1.0 / max(1, poll_hz)
        self.mode = LocomotionMode.IDLE
        self.yaw_accumulator = YawAccumulator()
        self._prev_ab = False
        self._prev_xy = False
        self._last_send = time.time()
        self._last_timestamp_ns: int | None = None
        self._sleep = time.sleep
        # The mode whose entry needed robot feedback it did not get; retried every tick.
        self._feedback_pending: StreamMode | None = None

    def reset_yaw(self) -> None:
        self.yaw_accumulator.reset()

    def send_idle(self) -> None:
        """Send one IDLE planner message, holding the current facing, with no optional fields.

        No upper-body, hand or VR 3-point fields: the robot reads their absence as "not
        operator-driven", so this drops VR 3-point control (back to the g1 encoder) and the
        frozen upper body. Used when the operator hands the robot back to the planner.
        """
        self._send(
            build_planner_message(
                mode=int(LocomotionMode.IDLE),
                movement=[0.0, 0.0, 0.0],
                facing=np.asarray(self.yaw_accumulator.heading),
                speed=-1.0,
                height=-1.0,
            )
        )

    def save_upper_body_position_target(self) -> None:
        """Grab the robot's measured upper body as the frozen targets (retried until it arrives)."""
        self._feedback_pending = None
        if not self.feedback.poll_feedback():
            self._feedback_pending = StreamMode.PLANNER_FROZEN_UPPER_BODY
            print(
                f"[PlannerStreamer] WARNING: no {_FEEDBACK} to freeze the upper body at — it stays on "
                f"the planner until that arrives; {_FEEDBACK_HINT}"
            )

    def recalibrate_for_vr3pt(self) -> None:
        """Schedule wrist recalibration against the robot's measured joints (retried until they arrive).

        Until then no 3-point targets are sent, so the robot keeps the g1 encoder: calibrating
        against an all-zero configuration instead (the arms are not at zero) would produce silently
        wrong wrist offsets and a wrist jump.
        """
        self._feedback_pending = None
        if not self._try_recalibrate_for_vr3pt(quiet=False):
            self._feedback_pending = StreamMode.PLANNER_VR_3PT
            print(
                f"[PlannerStreamer] WARNING: no {_FEEDBACK} for the VR 3PT wrist calibration — 3-point "
                f"targets are held back until it arrives; {_FEEDBACK_HINT}"
            )

    def _try_recalibrate_for_vr3pt(self, quiet: bool) -> bool:
        if not self.feedback.poll_feedback(quiet=quiet):
            return False
        self._three_point.reset_with_measured_q(self.feedback.full_body_q_measured)
        return True

    def _retry_pending_feedback(self, stream_mode: StreamMode) -> None:
        if stream_mode != self._feedback_pending:
            return
        if stream_mode == StreamMode.PLANNER_FROZEN_UPPER_BODY:
            arrived, what = self.feedback.poll_feedback(quiet=True), "upper body frozen"
        else:
            arrived, what = self._try_recalibrate_for_vr3pt(quiet=True), "VR 3PT wrist calibration scheduled"
        if arrived:
            self._feedback_pending = None
            print(f"[PlannerStreamer] robot feedback arrived; {what}")

    def run_once(
        self, stream_mode: StreamMode, sample: BodySample | None, controllers: ControllerSample, timestamp_ns: int
    ) -> None:
        """Send one planner command (skipped while the headset stream is stalled)."""
        if timestamp_ns == self._last_timestamp_ns:
            self._sleep(IDLE_POLL_S)  # stalled headset stream: poll, don't spin the manager loop
            return
        self._last_timestamp_ns = timestamp_ns
        self._retry_pending_feedback(stream_mode)

        # A+B -> next mode, X+Y -> previous (rising edges), clamped to the
        # modes the reference exposes (IDLE .. INJURED_WALK).
        ab_now = controllers.a and controllers.b
        xy_now = controllers.x and controllers.y
        if ab_now and not self._prev_ab:
            self.mode = LocomotionMode(min(int(LocomotionMode.INJURED_WALK), int(self.mode) + 1))
            print(f"[PlannerStreamer] mode -> {int(self.mode)}: {self.mode.name}")
        if xy_now and not self._prev_xy:
            self.mode = LocomotionMode(max(int(LocomotionMode.IDLE), int(self.mode) - 1))
            print(f"[PlannerStreamer] mode -> {int(self.mode)}: {self.mode.name}")
        self._prev_ab = ab_now
        self._prev_xy = xy_now

        lx, ly = controllers.left_axis
        rx, _ = controllers.right_axis
        facing = self.yaw_accumulator.update(rx, self.dt)

        raw_mag = float(np.clip(np.hypot(lx, ly), 0.0, 1.0))
        if raw_mag < JOYSTICK_DEADZONE:
            mag, speed = 0.0, -1.0
            mode_to_send = LocomotionMode.IDLE
        else:
            mag = min((raw_mag - JOYSTICK_DEADZONE) / (1.0 - JOYSTICK_DEADZONE), 1.0)
            mode_to_send = self.mode
            if self.mode == LocomotionMode.SLOW_WALK:
                speed = 0.1 + 0.5 * mag  # 0.1 .. 0.6 m/s
            elif self.mode == LocomotionMode.WALK:
                speed = -1.0  # mode default
            elif self.mode == LocomotionMode.RUN:
                speed = 1.5 + 3.0 * mag  # 1.5 .. 4.5 m/s
            else:
                speed = mag

        scale = mag / (raw_mag if raw_mag > 0.0 else 1.0)
        movement_local = np.array([-lx, ly]) * scale
        rotation_facing = np.array([[-facing[1], facing[0]], [facing[0], facing[1]]])
        movement_global = rotation_facing @ movement_local
        movement = [float(movement_global[0]), float(movement_global[1]), 0.0]

        upper_body_position = None
        left_hand = None
        right_hand = None
        if stream_mode == StreamMode.PLANNER_FROZEN_UPPER_BODY:
            upper_body_position = self.feedback.upper_body_position_target
            left_hand = self.feedback.left_hand_position_target
            right_hand = self.feedback.right_hand_position_target

        vr_position = None
        vr_orientation = None
        if stream_mode == StreamMode.PLANNER_VR_3PT:
            if sample is not None and self._feedback_pending != StreamMode.PLANNER_VR_3PT:
                vr_pose = self._three_point.process(sample.body_poses)
                vr_position = vr_pose[:, :3].flatten()
                vr_orientation = vr_pose[:, 3:].flatten()
                if self._on_frame is not None:
                    self._on_frame(vr_pose, None)
            left_hand, right_hand = hand_targets_from_triggers(controllers.left_trigger, controllers.right_trigger)

        self._send(
            build_planner_message(
                mode=int(mode_to_send),
                movement=movement,
                facing=np.asarray(facing),
                speed=float(speed),
                height=-1.0,
                upper_body_position=upper_body_position,
                left_hand_position=left_hand,
                right_hand_position=right_hand,
                vr_3pt_position=vr_position,
                vr_3pt_orientation=vr_orientation,
            )
        )

        # ~20 Hz pacing.
        now = time.time()
        sleep_t = self.dt - (now - self._last_send)
        if sleep_t > 0:
            self._sleep(sleep_t)
        self._last_send = time.time()
