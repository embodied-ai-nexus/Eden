"""Motion runtime for SONIC teleoperation: motion sequences, streaming merge, playback state.

Ports the reference deploy stack's motion machinery:

* :class:`MotionFrames` — the ``MotionSequence`` equivalent: a fixed-rate frame
  table (joints, root quaternion, optional SMPL data) with a mutable per-motion
  encoder mode.
* :class:`StreamedMotionMerger` — the sliding-window merger that folds
  overlapping streamed pose chunks (aligned by global frame index) into one
  playable motion, with catch-up reset when the stream falls too far behind.
* :class:`MotionScheduler` — the shared playback state (current motion, frame
  cursor, play flag, heading state, per-mode ZMQ extras). The ZMQ input thread
  and the planner thread mutate it under one lock; the control loop reads an
  immutable-per-tick :class:`MotionSnapshot`.

Encoder-mode ids follow ``joint_maps``: 0 = g1, 1 = teleop, 2 = smpl;
-2 = no encoder configured, -1 = external tokens expected.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

import numpy as np
from genesis.utils import geom as gu

from eden.utils.geom import heading_from_quat

_Z_AXIS = np.array([0.0, 0.0, 1.0])


def heading_quat(quat_wxyz: np.ndarray, inverse: bool = False) -> np.ndarray:
    """Yaw-only quaternion of ``quat_wxyz`` (its inverse when ``inverse``).

    Heading is the azimuth of the rotated x-axis, matching the reference
    deploy's ``calc_heading_quat`` / ``calc_heading_quat_inv``.
    """
    yaw = heading_from_quat(np.asarray(quat_wxyz, dtype=np.float64))
    return gu.axis_angle_to_quat(np.asarray(-yaw if inverse else yaw), _Z_AXIS)


@dataclass
class MotionFrames:
    """One motion sequence at a fixed frame rate (default 50 Hz).

    ``joint_pos``/``joint_vel`` are IsaacLab-order radians; ``body_quat`` is
    wxyz with body 0 = root. ``encode_mode`` is mutable runtime state exactly
    like the reference ``MotionSequence::encode_mode``.
    """

    name: str
    joint_pos: np.ndarray  # (T, J)
    joint_vel: np.ndarray  # (T, J)
    body_quat: np.ndarray  # (T, B, 4)
    body_pos: np.ndarray | None = None  # (T, B, 3)
    smpl_joints: np.ndarray | None = None  # (T, S, 3)
    smpl_pose: np.ndarray | None = None  # (T, P, 3)
    fps: float = 50.0
    encode_mode: int = -2
    body_part_indexes: tuple[int, ...] = (0,)

    @property
    def timesteps(self) -> int:
        return int(self.joint_pos.shape[0])


@dataclass
class MergeResult:
    """Outcome of one streamed-chunk merge (see ``StreamedMotionMerger.merge``)."""

    motion: MotionFrames | None
    window_start: int = 0  # global frame index of motion[0]
    frame_offset_adjustment: int = 0  # subtract from the playback cursor to track the window shift
    did_catchup_reset: bool = False  # True -> caller resets playback to frame 0
    frame_step: int = 1  # detected stride between consecutive frame indices
    protocol_version: int = 0


class StreamedMotionMerger:
    """Sliding-window merger for streamed motion chunks (port of ``streamed_motion_merger.hpp``).

    Chunks carry monotonically increasing global frame indices; the merger keeps
    ``HISTORY_FRAMES`` already-played frames for look-back, prepends the still-
    needed span of the previous window, and appends the incoming frames. When
    the playback position lags the stream by more than ``MAX_GAP_FRAMES`` it
    resets to the head of the incoming chunk and signals a catch-up. Not
    thread-safe; callers serialize access.
    """

    HISTORY_FRAMES = 5
    MAX_GAP_FRAMES = 200

    def __init__(self) -> None:
        # Catch-up resets that interrupted an already-playing stream (the
        # mandatory catch-up on a stream's first chunk is not counted). Each
        # one is a visible discontinuity, so the control loop reports it.
        self.midstream_catchup_count = 0
        self.reset()

    def reset(self) -> None:
        self._motion: MotionFrames | None = None
        self._window_start = 0

    @property
    def window_start(self) -> int:
        return self._window_start

    def merge(
        self,
        *,
        frame_indices: np.ndarray,
        body_quat: np.ndarray,
        joint_pos: np.ndarray | None = None,
        joint_vel: np.ndarray | None = None,
        smpl_joints: np.ndarray | None = None,
        smpl_pose: np.ndarray | None = None,
        protocol_version: int,
        current_playback_frame: int,
        catch_up_enabled: bool = True,
    ) -> MergeResult:
        """Merge one incoming chunk; returns the merged motion and playback adjustments.

        ``body_quat`` is ``(N, B, 4)`` (or ``(N, 4)`` for a root-only stream)
        and required for every version; v1 requires joint data, v2 SMPL data,
        v3 both — mirroring the reference validation.
        """
        frame_indices = np.asarray(frame_indices, dtype=np.int64).reshape(-1)
        body_quat = np.asarray(body_quat, dtype=np.float64)
        if body_quat.ndim == 2:
            body_quat = body_quat[:, None, :]
        num_frames = int(frame_indices.shape[0])
        if num_frames == 0 or body_quat.shape[0] != num_frames:
            raise ValueError("frame_indices and body_quat must carry the same non-zero frame count")

        has_joints = joint_pos is not None and joint_vel is not None
        has_smpl = smpl_joints is not None and smpl_pose is not None
        if protocol_version == 3 and not (has_joints and has_smpl):
            raise ValueError("pose protocol v3 requires joint and SMPL data")
        if protocol_version == 2 and not has_smpl:
            raise ValueError("pose protocol v2 requires SMPL data")
        if protocol_version == 1 and not has_joints:
            raise ValueError("pose protocol v1 requires joint data")
        if protocol_version not in (1, 2, 3):
            raise ValueError(f"unsupported pose protocol version: {protocol_version}")

        frame_step = 1
        if num_frames >= 2:
            step = abs(int(frame_indices[1] - frame_indices[0]))
            frame_step = step if step > 0 else 1
        incoming_start = int(frame_indices[0])
        incoming_end = int(frame_indices[-1])

        new_window_start, merge_dst_frame, did_catchup = self._sliding_window(
            incoming_start, incoming_end, frame_step, current_playback_frame, catch_up_enabled
        )

        # Zero joint width when the stream carries no joint data, so a g1-mode
        # gather fails and triggers the encoder fallback (reference num_joints=0).
        num_joints = int(joint_pos.shape[1]) if has_joints else 0
        total = merge_dst_frame + num_frames
        merged = MotionFrames(
            name="streamed",
            joint_pos=np.zeros((total, num_joints)),
            joint_vel=np.zeros((total, num_joints)),
            body_quat=np.zeros((total, body_quat.shape[1], 4)),
            smpl_joints=np.zeros((total,) + np.shape(smpl_joints)[1:]) if has_smpl else None,
            smpl_pose=np.zeros((total,) + np.shape(smpl_pose)[1:]) if has_smpl else None,
            body_part_indexes=(0,),
        )

        if merge_dst_frame > 0:
            self._copy_overlap(merged, new_window_start, incoming_start, frame_step, has_joints, has_smpl)

        if has_joints:
            merged.joint_pos[merge_dst_frame:] = np.asarray(joint_pos, dtype=np.float64)
            merged.joint_vel[merge_dst_frame:] = np.asarray(joint_vel, dtype=np.float64)
        merged.body_quat[merge_dst_frame:] = body_quat
        if has_smpl:
            merged.smpl_joints[merge_dst_frame:] = np.asarray(smpl_joints, dtype=np.float64)
            merged.smpl_pose[merge_dst_frame:] = np.asarray(smpl_pose, dtype=np.float64)

        window_shift = (new_window_start - self._window_start) // frame_step if frame_step > 0 else 0
        self._motion = merged
        self._window_start = new_window_start
        return MergeResult(
            motion=merged,
            window_start=new_window_start,
            frame_offset_adjustment=0 if did_catchup else window_shift,
            did_catchup_reset=did_catchup,
            frame_step=frame_step,
            protocol_version=protocol_version,
        )

    def _sliding_window(
        self,
        incoming_start: int,
        incoming_end: int,
        frame_step: int,
        current_playback_frame: int,
        catch_up_enabled: bool,
    ) -> tuple[int, int, bool]:
        """Return (new_window_start, merge_dst_frame, did_catchup)."""
        if self._motion is None or self._motion.timesteps <= 0:
            return incoming_start, 0, True

        max_gap_frames = self.MAX_GAP_FRAMES + self.HISTORY_FRAMES if catch_up_enabled else np.iinfo(np.int64).max
        window_end = self._window_start + frame_step * (self._motion.timesteps - 1)
        global_playback = self._window_start + frame_step * max(0, current_playback_frame - self.HISTORY_FRAMES)

        # Stale or already-covered data forces a catch-up reset (reference behavior).
        if incoming_start <= self._window_start or incoming_end <= window_end:
            self.midstream_catchup_count += 1
            return incoming_start, 0, True

        tentative_start = min(global_playback, incoming_start)
        tentative_dst = (incoming_start - tentative_start) // frame_step if frame_step > 0 else 0
        large_gap_from_old = incoming_start > window_end + frame_step
        if tentative_dst > max_gap_frames or large_gap_from_old:
            self.midstream_catchup_count += 1
            return incoming_start, 0, True
        return tentative_start, tentative_dst, False

    def _copy_overlap(
        self,
        merged: MotionFrames,
        new_window_start: int,
        incoming_start: int,
        frame_step: int,
        has_joints: bool,
        has_smpl: bool,
    ) -> None:
        """Fill ``merged[:merge_dst]`` with the still-valid span of the previous window."""
        old = self._motion
        if old is None or old.timesteps <= 0:
            return
        old_window_end = self._window_start + frame_step * old.timesteps
        overlap_start = max(new_window_start, self._window_start)
        overlap_end = min(incoming_start, old_window_end)
        if overlap_start >= overlap_end:
            return
        src = (overlap_start - self._window_start) // frame_step
        dst = (overlap_start - new_window_start) // frame_step
        count = (overlap_end - overlap_start) // frame_step

        if has_joints and old.joint_pos.shape[1] > 0:
            joints = min(merged.joint_pos.shape[1], old.joint_pos.shape[1])
            merged.joint_pos[dst : dst + count, :joints] = old.joint_pos[src : src + count, :joints]
            merged.joint_vel[dst : dst + count, :joints] = old.joint_vel[src : src + count, :joints]
        bodies = min(merged.body_quat.shape[1], old.body_quat.shape[1])
        merged.body_quat[dst : dst + count, :bodies] = old.body_quat[src : src + count, :bodies]
        if has_smpl and old.smpl_joints is not None:
            merged.smpl_joints[dst : dst + count] = old.smpl_joints[src : src + count]
            merged.smpl_pose[dst : dst + count] = old.smpl_pose[src : src + count]


@dataclass
class MotionSnapshot:
    """Frozen view of the playback state for one control tick.

    ``motion`` is treated as immutable by every consumer (writers always swap
    in a fresh :class:`MotionFrames`), so holding it outside the lock is safe.
    """

    motion: MotionFrames | None
    frame: int
    play: bool
    encode_mode: int
    apply_delta_heading: np.ndarray  # (4,) wxyz
    frozen_upper_targets: np.ndarray | None  # (17,) IsaacLab upper-body order
    frozen_upper_velocities: np.ndarray | None  # (17,) zeros unless streamed
    vr_position: np.ndarray | None  # (9,)
    vr_orientation: np.ndarray | None  # (12,) wxyz triple
    left_hand_joints: np.ndarray | None  # (7,)
    right_hand_joints: np.ndarray | None  # (7,)


@dataclass
class _HeadingState:
    """Captured robot heading vs. reference-motion heading (see ``UpdateHeadingState``)."""

    init_base_quat: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))
    init_ref_root_quat: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))
    delta_heading: float = 0.0


class MotionScheduler:
    """Thread-safe owner of the playback state shared by control, ZMQ, and planner threads.

    Mirrors the deploy binary's shared fields: ``current_motion_`` /
    ``current_frame_`` / ``operator_state.play`` / the heading state / the
    observation-window hold for streamed motions, plus the per-mode extras the
    ZMQ manager buffers (frozen-upper targets, VR 3-point values, hand
    targets).
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._motion: MotionFrames | None = None
        self._frame = 0
        self._play = False
        self._reinitialize_heading = True
        self._heading = _HeadingState()
        self._saved_frames_for_observation_window = 0
        self.frozen_upper_targets: np.ndarray | None = None
        self.frozen_upper_velocities: np.ndarray | None = None
        self.vr_position: np.ndarray | None = None
        self.vr_orientation: np.ndarray | None = None
        self.left_hand_joints: np.ndarray | None = None
        self.right_hand_joints: np.ndarray | None = None

    # -- motion swapping ---------------------------------------------------

    def set_motion(
        self, motion: MotionFrames | None, frame: int = 0, play: bool | None = None, reinitialize_heading: bool = True
    ) -> None:
        """Swap the current motion, optionally forcing the play flag."""
        with self._lock:
            self._motion = motion
            self._frame = frame
            if play is not None:
                self._play = play
            if reinitialize_heading:
                self._reinitialize_heading = True

    def apply_merge(self, result: MergeResult) -> None:
        """Adopt a merged streamed motion, keeping the global playback position.

        Port of the reference merge-result handling: a catch-up reset restarts
        at frame 0 and re-captures heading; otherwise the cursor shifts with
        the window (clamped into the new motion). Streaming always auto-plays.
        """
        if result.motion is None:
            return
        with self._lock:
            if result.did_catchup_reset:
                self._motion = result.motion
                self._frame = 0
                self._play = True
                self._reinitialize_heading = True
            else:
                adjusted = self._frame - result.frame_offset_adjustment
                adjusted = max(0, min(adjusted, result.motion.timesteps - 1 if result.motion.timesteps else 0))
                self._motion = result.motion
                self._frame = adjusted
                self._play = True

    # -- playback ----------------------------------------------------------

    @property
    def play(self) -> bool:
        with self._lock:
            return self._play

    @play.setter
    def play(self, value: bool) -> None:
        with self._lock:
            self._play = value

    @property
    def frame(self) -> int:
        with self._lock:
            return self._frame

    def set_frame(self, frame: int) -> None:
        with self._lock:
            self._frame = frame

    @property
    def motion(self) -> MotionFrames | None:
        with self._lock:
            return self._motion

    def motion_and_frame(self) -> tuple[MotionFrames | None, int]:
        """The current motion and cursor as one atomic pair.

        Reading ``motion`` then ``frame`` in two separate acquisitions lets the
        planner/ZMQ thread swap the motion between them, so the frame would
        index a different motion; callers that need both consistent (e.g. the
        fall-reference lookup) use this single critical section, mirroring the
        reference's one-lock ``current_motion_``/``current_frame_`` snapshot.
        """
        with self._lock:
            return self._motion, self._frame

    def stream_lag_frames(self) -> int | None:
        """Frames between the playback cursor and the newest streamed frame (None when not streaming)."""
        with self._lock:
            if self._motion is None or self._motion.name != "streamed" or self._motion.timesteps == 0:
                return None
            return self._motion.timesteps - 1 - self._frame

    def advance_frame(self, is_planner_motion: bool = False) -> None:
        """Advance playback by one control tick (reference ``CurrentFrameAdvancement``).

        A planner motion clamps at its last frame (the planner thread keeps
        extending it); a streamed motion holds far enough from the end to keep
        the future observation window inside the buffer; a clip resets to
        frame 0 and stops at the end, re-capturing heading.
        """
        with self._lock:
            if self._motion is None or self._motion.timesteps <= 0 or not self._play:
                return
            if is_planner_motion:
                self._frame = min(self._frame + 1, self._motion.timesteps - 1)
                return
            self._frame += 1
            if self._motion.name != "streamed":
                if self._frame >= self._motion.timesteps:
                    self._play = False
                    self._frame = 0
                    self._reinitialize_heading = True
            elif self._frame >= self._motion.timesteps - self._saved_frames_for_observation_window:
                self._frame -= 1

    # -- observation-window bookkeeping ------------------------------------

    def begin_control_tick(self, robot_base_quat: np.ndarray) -> None:
        """Reset the observation-window hold and refresh heading state for this tick.

        Must run before the tick's slot gathers, mirroring the reference
        control loop (``saved_frame_for_observation_window_ = 0`` then
        ``UpdateHeadingState``).
        """
        with self._lock:
            self._saved_frames_for_observation_window = 0
            self._update_heading_state(np.asarray(robot_base_quat, dtype=np.float64))

    def note_observation_window(self, num_frames: int, step_size: int) -> None:
        """Record a slot gather's future reach so streamed playback can hold before the end."""
        with self._lock:
            self._saved_frames_for_observation_window = max(
                self._saved_frames_for_observation_window, (num_frames - 1) * step_size + 1
            )

    # -- heading -----------------------------------------------------------

    def _update_heading_state(self, base_quat: np.ndarray) -> None:
        if self._motion is None or self._motion.timesteps == 0 or self._motion.body_quat.shape[1] == 0:
            return
        if self._reinitialize_heading:
            self._heading = _HeadingState(init_base_quat=base_quat.copy())
            self._heading.init_ref_root_quat = self._motion.body_quat[self._frame, 0].copy()
            self._reinitialize_heading = False
        if self._frame == 0:
            self._heading.init_ref_root_quat = self._motion.body_quat[0, 0].copy()

    def request_heading_reinitialize(self) -> None:
        with self._lock:
            self._reinitialize_heading = True

    def add_heading_increment(self, dyaw: float) -> None:
        with self._lock:
            self._heading.delta_heading += float(dyaw)

    @property
    def delta_heading(self) -> float:
        with self._lock:
            return self._heading.delta_heading

    def _apply_delta_heading(self) -> np.ndarray:
        """``euler_z(delta) ⊗ heading(init_base) ⊗ heading_inv(init_ref_root)``."""
        # transform_quat_by_quat(v, u) = u ⊗ v (left-multiply u onto v).
        apply = gu.transform_quat_by_quat(
            heading_quat(self._heading.init_ref_root_quat, inverse=True),
            heading_quat(self._heading.init_base_quat),
        )
        if self._heading.delta_heading != 0.0:
            apply = gu.transform_quat_by_quat(
                apply, gu.axis_angle_to_quat(np.asarray(self._heading.delta_heading), _Z_AXIS)
            )
        return apply

    # -- snapshot ----------------------------------------------------------

    def snapshot(self) -> MotionSnapshot:
        """Take the per-tick frozen view the policy step reads."""
        with self._lock:
            motion = self._motion
            return MotionSnapshot(
                motion=motion,
                frame=self._frame,
                play=self._play,
                encode_mode=motion.encode_mode if motion is not None else -2,
                apply_delta_heading=self._apply_delta_heading(),
                frozen_upper_targets=None if self.frozen_upper_targets is None else self.frozen_upper_targets.copy(),
                frozen_upper_velocities=(
                    None if self.frozen_upper_velocities is None else self.frozen_upper_velocities.copy()
                ),
                vr_position=None if self.vr_position is None else self.vr_position.copy(),
                vr_orientation=None if self.vr_orientation is None else self.vr_orientation.copy(),
                left_hand_joints=None if self.left_hand_joints is None else self.left_hand_joints.copy(),
                right_hand_joints=None if self.right_hand_joints is None else self.right_hand_joints.copy(),
            )
