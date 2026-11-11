"""POSE-mode streamer: PICO body frames -> 50 Hz windowed SMPL pose messages.

Port of the reference ``PoseStreamer``. Device frames (~90 Hz) are resampled
onto a fixed 50 Hz virtual clock by interpolating between the two frames that
bracket each virtual tick (nlerp for rotations), then the last five resampled
frames are re-sent as an overlapping window on every tick — the robot's
motion merger aligns them by frame index and rides out packet loss. Publishing
is gated until the window is full after a mode entry.

Each message also carries: the wrist joint targets derived from the elbow
twist/swing mapping, the calibrated VR 3-point pose of the *current* frame,
Dex3 hand targets from the triggers, data-collection toggle edges, and the
right-stick heading increment.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from collections.abc import Callable

import numpy as np
from scipy.spatial.transform import Rotation as sRot

from eden.extensions.policies.sonic.wire import pack_pose_message
from eden.extensions.teleop.base import TeleopStreamer
from eden.extensions.teleop.sonic.hands import hand_targets_from_triggers
from eden.extensions.teleop.sonic.inputs import BodySample, ControllerSample
from eden.extensions.teleop.sonic.smpl import SmplFk, process_body_poses
from eden.extensions.teleop.sonic.three_point import ThreePointPose
from eden.extensions.teleop.sonic.wrist import wrist_joints_from_body_pose
from eden.utils.geom import quat_nlerp

JOYSTICK_DEADZONE = 0.15


# Wait between polls while the input has no new frame (well under the ~11 ms PICO frame period).
IDLE_POLL_S = 0.002


class YawAccumulator:
    """Accumulates a heading angle from the right stick (yaw gain 1.5 rad/s)."""

    def __init__(self, yaw_gain: float = 1.5, deadzone: float = JOYSTICK_DEADZONE) -> None:
        self.yaw_gain = yaw_gain
        self.deadzone = deadzone
        self.reset()

    def reset(self) -> None:
        self.heading = [1.0, 0.0, 0.0]
        self.yaw_angle_rad = 0.0
        self.dyaw = 0.0

    def update(self, rx: float, dt: float) -> list[float]:
        """Integrate the stick's x-axis; returns the facing unit vector [x, y, 0]."""
        self.dyaw = self.yaw_gain * (-rx) * dt
        if abs(rx) >= self.deadzone:
            self.yaw_angle_rad += self.dyaw
            self.heading = [np.cos(self.yaw_angle_rad), np.sin(self.yaw_angle_rad), 0.0]
        return self.heading


class PoseStreamer(TeleopStreamer):
    """One POSE-mode streaming iteration per manager tick."""

    def __init__(
        self,
        send: Callable[[bytes], None],
        three_point: ThreePointPose,
        fk: SmplFk | None = None,
        num_frames_to_send: int = 5,
        target_fps: int = 50,
        record_dir: str = "",
        on_frame: Callable[[np.ndarray, np.ndarray], None] | None = None,
    ) -> None:
        """``send`` publishes one packed message; ``on_frame(vr_3pt_pose,
        smpl_joints_local)`` is the visualization hook, called once per
        processed device frame.
        """
        self._send = send
        self._three_point = three_point
        self._fk = fk or SmplFk()
        self._on_frame = on_frame
        self.num_frames_to_send = num_frames_to_send
        self.target_fps = target_fps
        self.record_dir = record_dir
        self._record_idx = 0
        if record_dir:
            import os

            os.makedirs(record_dir, exist_ok=True)

        self.yaw_accumulator = YawAccumulator()
        # 95% of the frame period: the remaining 5% absorbs processing overhead.
        self.frame_time = 0.95 / max(1, target_fps)
        self._sleep = time.sleep

        self.step = 0
        self.frame_buffer: dict[str, deque] = defaultdict(lambda: deque(maxlen=num_frames_to_send))
        self.prev_stamp_ns: int | None = None
        self._last_seen_stamp_ns: int | None = None  # newest device frame run_once has processed
        self.prev_smpl_pose: np.ndarray | None = None
        self.prev_smpl_joints: np.ndarray | None = None
        self.prev_body_quat: np.ndarray | None = None
        self.next_target_ns: int | None = None
        self.frame_start = time.time()
        self.buffer_cleared = True  # publish only once the window refills
        self._toggle_dc_last = False
        self._toggle_da_last = False
        self._emit_count = 0
        self._rate_log_time = time.time()

    def reset_yaw(self) -> None:
        self.yaw_accumulator.reset()

    def on_mode_exit(self) -> None:
        """Clear the stream window; re-entry publishes only after it refills."""
        self.frame_buffer.clear()
        self.prev_stamp_ns = None
        self._last_seen_stamp_ns = None
        self.prev_smpl_pose = None
        self.prev_smpl_joints = None
        self.prev_body_quat = None
        self.next_target_ns = None
        self.buffer_cleared = True
        self.step = 0

    def run_once(self, sample: BodySample | None, controllers: ControllerSample) -> None:
        """Process the latest device frame; publish when a virtual tick completes."""
        if sample is None:
            self._sleep(0.005)
            return
        if sample.timestamp_ns == self._last_seen_stamp_ns:
            # Same device frame as last tick: every path below would reach the same
            # result again, after a full SMPL-X FK. Without this wait the manager
            # loop would spin between device frames, re-running the FK and publishing
            # manager_state at kHz rates while competing with the reader thread.
            self._sleep(IDLE_POLL_S)
            return
        self._last_seen_stamp_ns = sample.timestamp_ns

        frame = process_body_poses(sample.body_poses, self._fk)
        smpl_pose = frame.body_pose[:63].reshape(21, 3).astype(np.float32)
        smpl_joints = frame.smpl_joints_local.astype(np.float32)
        body_quat = frame.global_orient_quat.astype(np.float32)

        toggle_dc_now = controllers.a and controllers.left_grip > 0.5
        toggle_da_now = controllers.b and controllers.left_grip > 0.5
        toggle_dc = toggle_dc_now and not self._toggle_dc_last
        toggle_da = toggle_da_now and not self._toggle_da_last
        self._toggle_dc_last = toggle_dc_now
        self._toggle_da_last = toggle_da_now

        left_hand, right_hand = hand_targets_from_triggers(controllers.left_trigger, controllers.right_trigger)

        # --- resample onto the 50 Hz virtual clock ---
        curr_stamp_ns = sample.timestamp_ns
        step_ns = int(1e9 / max(1, self.target_fps))
        if self.prev_stamp_ns is None:
            self.prev_stamp_ns = curr_stamp_ns
            self.prev_smpl_pose = smpl_pose
            self.prev_smpl_joints = smpl_joints
            self.prev_body_quat = body_quat
            self.next_target_ns = curr_stamp_ns
            return
        if curr_stamp_ns <= self.prev_stamp_ns:
            return  # duplicate device frame
        if self.next_target_ns is None:
            self.next_target_ns = self.prev_stamp_ns + step_ns
        self.next_target_ns = max(self.next_target_ns, self.prev_stamp_ns)
        if self.next_target_ns > curr_stamp_ns:
            return  # virtual tick not yet bracketed
        denom = float(curr_stamp_ns - self.prev_stamp_ns)
        alpha = float(np.clip((self.next_target_ns - self.prev_stamp_ns) / denom if denom > 0 else 1.0, 0.0, 1.0))

        use_joints = (1.0 - alpha) * self.prev_smpl_joints + alpha * smpl_joints
        # Per-joint NLERP of the (21, 3) axis-angle pose, via quaternions.
        prev_q = sRot.from_rotvec(self.prev_smpl_pose.reshape(-1, 3)).as_quat()
        curr_q = sRot.from_rotvec(smpl_pose.reshape(-1, 3)).as_quat()
        use_pose = (
            sRot.from_quat(quat_nlerp(prev_q, curr_q, alpha)).as_rotvec().reshape(smpl_pose.shape).astype(np.float32)
        )
        use_body_quat = quat_nlerp(self.prev_body_quat, body_quat, alpha).astype(np.float32)

        joint_pos = wrist_joints_from_body_pose(use_pose)

        # Calibrated VR 3-point pose of the current (raw) frame.
        vr_3pt_pose = self._three_point.process(sample.body_poses)
        if self._on_frame is not None:
            self._on_frame(vr_3pt_pose, frame.smpl_joints_local)

        self.frame_buffer["smpl_pose"].append(use_pose)
        self.frame_buffer["smpl_joints"].append(use_joints)
        self.frame_buffer["body_quat_w"].append(use_body_quat)
        self.frame_buffer["frame_index"].append(int(self.step))
        self.frame_buffer["joint_pos"].append(joint_pos)

        buffer_is_full = len(self.frame_buffer["frame_index"]) >= self.num_frames_to_send
        if buffer_is_full and self.buffer_cleared:
            self.buffer_cleared = False

        self.yaw_accumulator.update(controllers.right_axis[0], self.frame_time)

        if buffer_is_full and not self.buffer_cleared:
            n = len(self.frame_buffer["frame_index"])
            message_data = {
                "smpl_pose": np.stack(self.frame_buffer["smpl_pose"]),
                "smpl_joints": np.stack(self.frame_buffer["smpl_joints"]),
                "body_quat_w": np.stack(self.frame_buffer["body_quat_w"]),
                "joint_pos": np.stack(self.frame_buffer["joint_pos"]),
                "joint_vel": np.zeros((n, 29)),
                "vr_position": vr_3pt_pose[:, :3].flatten().astype(np.float32),
                "vr_orientation": vr_3pt_pose[:, 3:].flatten().astype(np.float32),
                "frame_index": np.array(self.frame_buffer["frame_index"], dtype=np.int64),
                "left_trigger": np.array([controllers.left_trigger], dtype=np.float32),
                "right_trigger": np.array([controllers.right_trigger], dtype=np.float32),
                "left_grip": np.array([controllers.left_grip], dtype=np.float32),
                "right_grip": np.array([controllers.right_grip], dtype=np.float32),
                "pico_dt": np.array([sample.dt], dtype=np.float32),
                "pico_fps": np.array([sample.fps], dtype=np.float32),
                "timestamp_realtime": np.array([sample.timestamp_realtime], dtype=np.float64),
                "timestamp_monotonic": np.array([sample.timestamp_monotonic], dtype=np.float64),
                "left_hand_joints": left_hand.astype(np.float32),
                "right_hand_joints": right_hand.astype(np.float32),
                "toggle_data_collection": np.array([toggle_dc], dtype=bool),
                "toggle_data_abort": np.array([toggle_da], dtype=bool),
                "heading_increment": np.array([self.yaw_accumulator.dyaw], dtype=np.float32),
            }
            self._send(pack_pose_message(message_data, topic="pose"))
            if self.record_dir:
                import os

                np.savez_compressed(os.path.join(self.record_dir, f"pose_{self._record_idx:06d}.npz"), **message_data)
                self._record_idx += 1

        self.step += 1
        self.next_target_ns += step_ns
        self.prev_stamp_ns = curr_stamp_ns
        self.prev_smpl_pose = smpl_pose
        self.prev_smpl_joints = smpl_joints
        self.prev_body_quat = body_quat

        # Reference parity ([PoseLoop] FPS/Step every 5 s). The resampler emits
        # at most one virtual frame per distinct device sample, so a sustained
        # rate below target means the input path is starving the robot's stream.
        self._emit_count += 1
        now = time.time()
        window = now - self._rate_log_time
        if window >= 5.0:
            rate = self._emit_count / window
            print(f"[PoseLoop] FPS: {rate:.1f} Step: {self.step}")
            if rate < 0.9 * self.target_fps:
                print(
                    f"[PoseLoop] WARNING: emitting {rate:.1f} Hz < target {self.target_fps} — "
                    "device/bridge rate too low, the robot-side stream will stall"
                )
            self._emit_count = 0
            self._rate_log_time = now

        elapsed = time.time() - self.frame_start
        if elapsed < self.frame_time:
            self._sleep(self.frame_time - elapsed)
        self.frame_start = time.time()
