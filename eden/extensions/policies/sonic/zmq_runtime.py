"""Robot-side ZMQ input manager for SONIC teleoperation.

Port of the deploy binary's ``ZMQManager`` + ``ZMQEndpointInterface``: a
background thread receives the operator's ``command`` / ``planner`` / ``pose``
messages, and :meth:`SonicZmqServer.apply` — called once per control tick —
turns them into runtime state:

* ``command``: start/stop control (OR-accumulated between ticks) and the
  managed-mode select (PLANNER vs STREAMED_MOTION), with a safety reset on
  every switch.
* ``planner``: the movement command (squat height floor, static-mode speed
  override, normalized directions), the frozen-upper 17-joint override, hand
  targets, and the VR 3-point buffers whose presence toggles encoder mode
  1 <-> 0. A 1 s silence resets to IDLE and drops the upper-body override.
* ``pose``: streamed motion chunks folded through the
  :class:`~eden.extensions.policies.sonic.motion.StreamedMotionMerger`
  (protocol version latched on the first chunk; v1 -> encoder mode 0,
  v2/v3 -> mode 2), plus heading increments and hand targets.

:class:`Dex3HandController` reproduces the reference hand-command shaping:
close-ratio clipping against the Dex3 joint limits and a per-tick delta clamp
against the measured hand state.
"""

from __future__ import annotations

import threading
import time
from enum import Enum

import numpy as np

from eden.extensions.policies.sonic import wire
from eden.extensions.policies.sonic.motion import MotionFrames, MotionScheduler, StreamedMotionMerger
from eden.extensions.policies.sonic.planner import (
    LocomotionMode,
    MovementState,
    SonicPlanner,
    is_squat_mode,
    is_static_mode,
)


class ManagedMode(Enum):
    PLANNER = "planner"
    STREAMED_MOTION = "streamed_motion"


class Dex3HandController:
    """Shapes streamed Dex3 targets like the reference hand writer.

    Targets are clipped so no joint closes beyond ``max_close_ratio`` of its
    limit, then rate-limited to ``MAX_DELTA_Q`` per control tick against the
    measured joint positions. (The reference applies the delta clamp per
    500 Hz write; at the 50 Hz control rate the clamp is proportionally
    tighter but still ~12.5 rad/s, far above finger speeds.)
    """

    MAX_DELTA_Q = 0.25  # rad per write
    # Per-motor limits (thumb 0/1/2, middle 0/1, index 0/1), from dex3_hands.hpp.
    MAX_LIMITS_LEFT = np.array([1.05, 1.05, 1.75, 0.0, 0.0, 0.0, 0.0])
    MIN_LIMITS_LEFT = np.array([-1.05, -0.724, 0.0, -1.57, -1.75, -1.57, -1.75])
    MAX_LIMITS_RIGHT = np.array([1.05, 0.742, 0.0, 1.57, 1.75, 1.57, 1.75])
    MIN_LIMITS_RIGHT = np.array([-1.05, -1.05, -1.75, 0.0, 0.0, 0.0, 0.0])

    def __init__(self, max_close_ratio: float = 1.0) -> None:
        self._lock = threading.Lock()
        self._desired_left = np.zeros(7)  # open
        self._desired_right = np.zeros(7)
        self.set_max_close_ratio(max_close_ratio)

    def set_max_close_ratio(self, ratio: float) -> None:
        """Bound how far the fingers may close (1.0 = fully, floor 0.2)."""
        self._max_close_ratio = float(np.clip(ratio, 0.2, 1.0))

    @property
    def max_close_ratio(self) -> float:
        return self._max_close_ratio

    def reset(self) -> None:
        """Back to open hands; the per-tick delta clamp ramps there smoothly."""
        with self._lock:
            self._desired_left = np.zeros(7)
            self._desired_right = np.zeros(7)

    def set_targets(self, left: np.ndarray | None, right: np.ndarray | None) -> None:
        """Update the desired 7-DOF joint targets for each hand (None keeps the current one)."""
        with self._lock:
            if left is not None:
                self._desired_left = np.asarray(left, dtype=np.float64).reshape(7)
            if right is not None:
                self._desired_right = np.asarray(right, dtype=np.float64).reshape(7)

    def _shape(self, desired: np.ndarray, measured: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
        # clipToMaxOpen: q=0 is fully open; scale each limit by the close ratio.
        clipped = np.clip(desired, self._max_close_ratio * lo, self._max_close_ratio * hi)
        delta = np.clip(clipped - measured, -self.MAX_DELTA_Q, self.MAX_DELTA_Q)
        return measured + delta

    def compute(self, measured_left: np.ndarray, measured_right: np.ndarray) -> np.ndarray:
        """Return the 14 hand joint targets (left then right) for this tick."""
        with self._lock:
            left = self._shape(
                self._desired_left, np.asarray(measured_left), self.MIN_LIMITS_LEFT, self.MAX_LIMITS_LEFT
            )
            right = self._shape(
                self._desired_right, np.asarray(measured_right), self.MIN_LIMITS_RIGHT, self.MAX_LIMITS_RIGHT
            )
        return np.concatenate([left, right])


class SonicZmqServer:
    """Receives the operator's packed messages and drives the motion runtime.

    ``start()`` launches the receive thread; ``apply()`` must run once per
    control tick on the control thread (it is the only writer of scheduler /
    planner state). ``stop_requested`` latches once a stop or e-stop arrives.
    """

    PLANNER_TIMEOUT_S = 1.0  # silence on the planner topic -> IDLE
    PLANNER_SWITCH_FRESHNESS_S = 0.1  # planner message considered current at a mode switch
    PLANNER_INIT_TIMEOUT_S = 5.0  # waiting for the first planner motion
    RCVHWM = 100  # receive queue of the one SUB socket all three operator topics share (see _run)

    def __init__(
        self,
        scheduler: MotionScheduler,
        planner: SonicPlanner,
        hands: Dex3HandController,
        host: str = "localhost",
        port: int = wire.DEFAULT_PACKED_PORT,
    ) -> None:
        self._scheduler = scheduler
        self._planner = planner
        self._hands = hands
        self._merger = StreamedMotionMerger()
        self._endpoint = f"tcp://{host}:{port}"

        self._lock = threading.Lock()
        # command accumulation (reference: OR until consumed; planner flag overwrites)
        self._cmd_valid = False
        self._cmd_start = False
        self._cmd_stop = False
        self._cmd_planner = True
        # latest planner message + arrival time
        self._planner_msg: wire.PlannerMessage | None = None
        self._planner_msg_time = 0.0
        self._planner_msg_valid = False
        # latest pose chunk (conflate-latest, like has_new_data_)
        self._pose_chunk: wire.PoseChunk | None = None

        # control-thread state
        self.active_mode = ManagedMode.PLANNER
        self.use_zmq_stream = False
        self.operator_start = False
        self.stop_requested = False
        self.movement_state = MovementState(locomotion_mode=int(LocomotionMode.IDLE), movement_speed=-1.0, height=-1.0)
        self._safety_reset_pending = False
        self._zmq_toggle_pending = False
        self._planner_wait_deadline: float | None = None
        self._active_protocol_version = -1
        self._has_vr_3point_control = False
        self._last_has_vr_3point_control = False
        self._has_upper_body_control = False

        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    # ------------------------------------------------------------------
    # receive thread
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="sonic-zmq-server", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def _run(self) -> None:
        import zmq

        context = zmq.Context.instance()
        socket = context.socket(zmq.SUB)
        # One socket carries all three topics, so the HWM bounds them together: a shallow
        # queue (e.g. 3) lets a short pose burst while this thread waits on the GIL drop
        # whatever comes next — including a one-shot start/stop command. Dispatch only parses and
        # stores the latest planner message / pose chunk, so a deeper queue drains in
        # microseconds per message and adds no staleness.
        socket.setsockopt(zmq.RCVHWM, self.RCVHWM)
        socket.setsockopt(zmq.RCVTIMEO, 100)
        socket.setsockopt(zmq.LINGER, 0)
        for topic in (wire.COMMAND_TOPIC, wire.PLANNER_TOPIC, wire.POSE_TOPIC):
            socket.setsockopt_string(zmq.SUBSCRIBE, topic)
        socket.connect(self._endpoint)
        try:
            while not self._stop_event.is_set():
                try:
                    msg = socket.recv()
                except zmq.Again:
                    continue
                try:
                    self._dispatch(bytes(msg))
                except Exception as e:  # noqa: BLE001 — the receive thread must outlive any bad payload
                    print(f"[SonicZmqServer] dropped malformed message: {e}")
        finally:
            socket.close()

    def _dispatch(self, msg: bytes) -> None:
        if msg.startswith(b"planner"):  # the three topics share no prefix, so test order does not matter
            parsed = wire.parse_planner(wire.unpack_message(msg, wire.PLANNER_TOPIC))
            with self._lock:
                self._planner_msg = parsed
                self._planner_msg_time = time.monotonic()
                self._planner_msg_valid = True
        elif msg.startswith(b"pose"):
            chunk = wire.parse_pose(wire.unpack_message(msg, wire.POSE_TOPIC))
            with self._lock:
                self._pose_chunk = chunk
        elif msg.startswith(b"command"):
            cmd = wire.parse_command(wire.unpack_message(msg, wire.COMMAND_TOPIC))
            with self._lock:
                # start/stop accumulate with OR until consumed; the mode flag overwrites.
                self._cmd_start = self._cmd_start or cmd.start
                self._cmd_stop = self._cmd_stop or cmd.stop
                self._cmd_planner = cmd.planner
                self._cmd_valid = True

    # ------------------------------------------------------------------
    # control-thread processing (one call per control tick)
    # ------------------------------------------------------------------

    def emergency_stop(self) -> None:
        """Operator e-stop (the runtime's ``o`` key): stop control, drop the planner."""
        self.stop_requested = True
        self._planner.state.enabled = False
        self._planner.state.initialized = False
        with self._lock:
            self._planner_msg_valid = False
            self._planner_msg = None
        self._clear_upper_body()

    def reset_session(self) -> None:
        """Return to the waiting-for-operator launch state (fall auto-reset).

        Unlike :meth:`emergency_stop` this does not latch ``stop_requested`` —
        the session keeps running and the operator re-engages with a fresh
        start command (full calibration included, like the first engage).
        """
        self.operator_start = False
        self.active_mode = ManagedMode.PLANNER
        self.use_zmq_stream = False
        self.movement_state = MovementState(locomotion_mode=int(LocomotionMode.IDLE), movement_speed=-1.0, height=-1.0)
        self._safety_reset_pending = False
        self._zmq_toggle_pending = False
        self._planner_wait_deadline = None
        self._has_vr_3point_control = False
        self._last_has_vr_3point_control = False
        self._has_upper_body_control = False
        self._planner.state.enabled = False
        self._planner.state.initialized = False
        self._clear_upper_body()
        self._reset_stream()
        with self._lock:
            self._planner_msg = None
            self._planner_msg_valid = False
            self._cmd_valid = self._cmd_start = self._cmd_stop = False
            self._cmd_planner = True
        self._hands.reset()
        self._scheduler.set_motion(None, frame=0, play=False, reinitialize_heading=True)
        self._scheduler.vr_position = None
        self._scheduler.vr_orientation = None
        self._scheduler.left_hand_joints = None
        self._scheduler.right_hand_joints = None

    @property
    def midstream_catchups(self) -> int:
        """Merger catch-up resets that interrupted an already-playing stream."""
        return self._merger.midstream_catchup_count

    def apply(self) -> None:
        """Consume buffered messages and update the motion runtime for this tick."""
        start_control, stop_control = self._consume_command()

        if stop_control:
            self.stop_requested = True
            self._planner.state.enabled = False
            self._planner.state.initialized = False
            with self._lock:
                self._planner_msg_valid = False
                self._planner_msg = None
            self._clear_upper_body()
            return

        if self.active_mode == ManagedMode.PLANNER:
            self._apply_planner_mode(start_control)
        else:
            self._apply_streamed_mode(start_control)

    def _consume_command(self) -> tuple[bool, bool]:
        """Port of ``ZMQManager::update``: flags + mode switching."""
        with self._lock:
            if not self._cmd_valid:
                return False, False
            start_control, stop_control = self._cmd_start, self._cmd_stop
            new_mode = ManagedMode.PLANNER if self._cmd_planner else ManagedMode.STREAMED_MOTION
            self._cmd_valid = self._cmd_start = self._cmd_stop = False
            planner_msg_fresh = (
                self._planner_msg_valid and time.monotonic() - self._planner_msg_time < self.PLANNER_SWITCH_FRESHNESS_S
            )
            fresh_msg = self._planner_msg if planner_msg_fresh else None

        if new_mode != self.active_mode:
            self._safety_reset_pending = True
            if new_mode == ManagedMode.PLANNER:
                if fresh_msg is not None:
                    self._has_upper_body_control = fresh_msg.upper_body_position is not None
            else:
                self._zmq_toggle_pending = True
                with self._lock:
                    self._planner_msg_valid = False
                    self._planner_msg = None
            self.active_mode = new_mode
        return start_control, stop_control

    def _clear_upper_body(self) -> None:
        self._has_upper_body_control = False
        self._scheduler.frozen_upper_targets = None
        self._scheduler.frozen_upper_velocities = None

    # ---- PLANNER managed mode -----------------------------------------

    def _apply_planner_mode(self, start_control: bool) -> None:
        scheduler = self._scheduler
        planner = self._planner

        if self._safety_reset_pending:
            self._safety_reset_pending = False
            scheduler.play = False
            if self.operator_start:
                if planner.state.enabled and planner.state.initialized:
                    # Planner already running: keep it, drop a stale teleop mode.
                    motion = scheduler.motion
                    if motion is not None and motion.encode_mode == 1:
                        motion.encode_mode = 0
                    scheduler.play = True
                else:
                    self.movement_state = MovementState(
                        locomotion_mode=int(LocomotionMode.IDLE), movement_speed=-1.0, height=-1.0
                    )
                    self._begin_planner_wait()
            return

        if start_control and not self.operator_start:
            self.operator_start = True
            scheduler.play = False
            scheduler.request_heading_reinitialize()
            self._begin_planner_wait()
            return

        # Non-blocking port of the reference's 5 s planner-init wait: physics
        # keeps stepping; playback starts the tick the planner motion arrives.
        if self._planner_wait_deadline is not None:
            motion = scheduler.motion
            if motion is not None and motion.name == "planner_motion":
                self._planner_wait_deadline = None
                scheduler.play = True
            elif time.monotonic() > self._planner_wait_deadline:
                print("[SonicZmqServer] planner initialization timeout")
                self.stop_requested = True
                self._planner_wait_deadline = None
                return

        self._apply_planner_message()

    def _begin_planner_wait(self) -> None:
        if not self._planner.state.enabled:
            self._planner.state.enabled = True
        self._planner_wait_deadline = time.monotonic() + self.PLANNER_INIT_TIMEOUT_S
        # A fresh planner motion is coming (encode mode 0): judge VR presence against it
        # from scratch. A latch left on by an earlier VR session (VR_3PT -> POSE ->
        # PLANNER -> VR_3PT) would otherwise swallow the rising edge, and the new
        # motion would run the g1 encoder for the whole VR session.
        self._has_vr_3point_control = self._last_has_vr_3point_control = False

    def _apply_planner_message(self) -> None:
        planner = self._planner
        scheduler = self._scheduler
        if not (planner.state.enabled and planner.state.initialized):
            return

        with self._lock:
            msg = self._planner_msg if self._planner_msg_valid else None
            elapsed = time.monotonic() - self._planner_msg_time
            if msg is not None:
                self._planner_msg_valid = False  # consume; keep timestamp for the timeout check
            timed_out = self._planner_msg is not None and elapsed >= self.PLANNER_TIMEOUT_S

        if msg is not None:
            self._has_upper_body_control = msg.upper_body_position is not None
            if self._has_upper_body_control:
                scheduler.frozen_upper_targets = np.asarray(msg.upper_body_position, dtype=np.float64)
                scheduler.frozen_upper_velocities = (
                    np.asarray(msg.upper_body_velocity, dtype=np.float64)
                    if msg.upper_body_velocity is not None
                    else None
                )
            else:
                scheduler.frozen_upper_targets = None
                scheduler.frozen_upper_velocities = None

            if msg.left_hand_joints is not None or msg.right_hand_joints is not None:
                self._hands.set_targets(msg.left_hand_joints, msg.right_hand_joints)

            state = MovementState(
                locomotion_mode=msg.mode,
                movement_direction=msg.movement.copy(),
                facing_direction=msg.facing.copy(),
                movement_speed=msg.speed,
                height=msg.height,
            )
            if is_squat_mode(state.locomotion_mode) and state.height < 0.2:
                state.height = 0.2
            if is_static_mode(state.locomotion_mode):
                state.movement_speed = -1.0
            state.facing_direction = state.facing_direction / max(float(np.linalg.norm(state.facing_direction)), 1e-12)
            norm = float(np.linalg.norm(state.movement_direction))
            state.movement_direction = state.movement_direction / max(norm, 1e-12)
            self.movement_state = state

            # VR 3-point presence toggles the teleop encoder.
            self._has_vr_3point_control = msg.vr_position is not None
            if msg.vr_position is not None:
                scheduler.vr_position = np.asarray(msg.vr_position, dtype=np.float64)
                if msg.vr_orientation is not None:
                    scheduler.vr_orientation = np.asarray(msg.vr_orientation, dtype=np.float64)

        elif timed_out:
            # The timeout clears only the upper-body override; hand targets and
            # VR 3-point control persist (no encoder-mode edge fires).
            self._clear_upper_body()
            self.movement_state = MovementState(
                locomotion_mode=int(LocomotionMode.IDLE),
                facing_direction=self.movement_state.facing_direction.copy(),
                movement_speed=-1.0,
                height=-1.0,
            )
            with self._lock:
                self._planner_msg = None
                self._planner_msg_valid = False
            print("[SonicZmqServer] planner topic timeout — reset to IDLE")

        # Encoder-mode edges on VR presence (rising -> teleop, falling -> g1),
        # fired only against the planner's own motion. Until the first generation
        # is adopted (``SonicPlanner.advance``, at the end of the tick after the
        # planner flags itself initialized) ``scheduler.motion`` is None or the
        # previous mode's motion; latching the edge then would lose it — the
        # adopted planner motion starts in g1 mode and the edge never fires again,
        # so VR_3PT would run the g1 encoder. Holding the latch lets the edge land
        # on the first tick the planner motion is current. The reference has no
        # such window: its planner-init wait blocks until the motion is adopted.
        motion = scheduler.motion
        if motion is None or motion.name != "planner_motion":
            return
        if motion.encode_mode >= 0:
            if self._has_vr_3point_control and not self._last_has_vr_3point_control:
                motion.encode_mode = 1
            elif not self._has_vr_3point_control and self._last_has_vr_3point_control:
                motion.encode_mode = 0
        self._last_has_vr_3point_control = self._has_vr_3point_control

    # ---- STREAMED_MOTION managed mode ---------------------------------

    def _apply_streamed_mode(self, start_control: bool) -> None:
        scheduler = self._scheduler

        if self._safety_reset_pending:
            self._safety_reset_pending = False
            self.movement_state = MovementState(
                locomotion_mode=int(LocomotionMode.IDLE), movement_speed=-1.0, height=-1.0
            )
            # Freeze the current motion under a clip-like name so playback
            # clamps at its end instead of holding for more stream data.
            motion = scheduler.motion
            if motion is not None:
                frozen = MotionFrames(
                    name="temporary_motion",
                    joint_pos=motion.joint_pos,
                    joint_vel=motion.joint_vel,
                    body_quat=motion.body_quat,
                    body_pos=motion.body_pos,
                    smpl_joints=motion.smpl_joints,
                    smpl_pose=motion.smpl_pose,
                )
                frozen.encode_mode = motion.encode_mode
                scheduler.set_motion(frozen, frame=scheduler.frame, play=False, reinitialize_heading=True)
            else:
                scheduler.play = False
                scheduler.request_heading_reinitialize()
            self._planner.state.enabled = False
            self._planner.state.initialized = False
            self.use_zmq_stream = False
            self._reset_stream()

        if self._zmq_toggle_pending:
            self._zmq_toggle_pending = False
            self.use_zmq_stream = True
            scheduler.play = False
            scheduler.request_heading_reinitialize()
            self._reset_stream()

        if start_control:
            self.operator_start = True

        if not self.use_zmq_stream:
            return

        with self._lock:
            chunk = self._pose_chunk
            self._pose_chunk = None
        if chunk is None:
            return

        if chunk.version == 4:
            self._disable_streaming("token streaming (pose protocol v4) is not supported by the Eden runtime")
            return
        if self._active_protocol_version == -1:
            self._active_protocol_version = chunk.version
            print(f"[SonicZmqServer] pose protocol v{chunk.version} established")
        elif self._active_protocol_version != chunk.version:
            self._disable_streaming(
                f"pose protocol changed from v{self._active_protocol_version} to v{chunk.version} mid-session"
            )
            return

        try:
            result = self._merger.merge(
                frame_indices=chunk.frame_index,
                body_quat=np.asarray(chunk.body_quat, dtype=np.float64),
                joint_pos=None if chunk.joint_pos is None else np.asarray(chunk.joint_pos, dtype=np.float64),
                joint_vel=None if chunk.joint_vel is None else np.asarray(chunk.joint_vel, dtype=np.float64),
                smpl_joints=None if chunk.smpl_joints is None else np.asarray(chunk.smpl_joints, dtype=np.float64),
                smpl_pose=None if chunk.smpl_pose is None else np.asarray(chunk.smpl_pose, dtype=np.float64),
                protocol_version=chunk.version,
                current_playback_frame=scheduler.frame,
                catch_up_enabled=chunk.catch_up,
            )
        except ValueError as e:
            print(f"[SonicZmqServer] dropped inconsistent pose chunk: {e}")
            return
        if result.motion is not None:
            result.motion.encode_mode = 0 if chunk.version == 1 else 2
            scheduler.apply_merge(result)

        if chunk.left_hand_joints is not None or chunk.right_hand_joints is not None:
            self._hands.set_targets(chunk.left_hand_joints, chunk.right_hand_joints)
        if chunk.vr_position is not None:
            from eden.extensions.policies.sonic.teleop_policy import VR_3POINT_DEFAULT_ORIENTATION

            scheduler.vr_position = np.asarray(chunk.vr_position, dtype=np.float64)
            # A missing orientation resets to the defaults on the pose path
            # (the reference decodes defaults there, unlike the planner path).
            scheduler.vr_orientation = (
                np.asarray(chunk.vr_orientation, dtype=np.float64)
                if chunk.vr_orientation is not None
                else VR_3POINT_DEFAULT_ORIENTATION.copy()
            )
        if chunk.heading_increment:
            scheduler.add_heading_increment(chunk.heading_increment)

    def _reset_stream(self) -> None:
        self._merger.reset()
        self._active_protocol_version = -1
        with self._lock:
            self._pose_chunk = None

    def _disable_streaming(self, reason: str) -> None:
        print(f"[SonicZmqServer] {reason}; ZMQ streaming disabled for safety")
        self.use_zmq_stream = False
        self._scheduler.play = False
        self._scheduler.request_heading_reinitialize()
        self._reset_stream()
