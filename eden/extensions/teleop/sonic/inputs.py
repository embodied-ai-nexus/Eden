"""PICO body-tracking readers for the SONIC operator station.

Both readers hand the pipelines the **raw** XRoboToolkit service data — the
``(24, 7)`` full-body joint poses in the service's own frame (positions +
scalar-last quaternions) plus controller buttons/sticks/triggers. The SONIC
pipelines apply their own frame conversions (see ``smpl.py`` /
``three_point.py``), so neither reader goes through
:class:`eden.utils.devices.pico.xr_client.PicoXrClient`'s robot-frame poses,
whose conversion differs from the reference teleop math.

:class:`BridgeReader` (the default) subscribes to ``eden pico bridge``'s pose
channel and consumes its lossless ``body_raw`` passthrough, so the SDK and its
service-library linkage live in the bridge process. :class:`PicoReader` is the
in-process port of the reference ``input_readers.PicoReader``: a daemon thread
spins on ``is_body_data_available``, dedupes frames by device timestamp, keeps
exactly one latest sample, and flags a disconnect after 5 s without fresh data.
The bridge reader keeps the same sample semantics (dedupe on the same
timestamp, identical dt/fps bookkeeping).
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass

import numpy as np

from eden.extensions.teleop.base import TeleopInputSource
from eden.utils.devices.pico.xr_client import N_BODY_JOINTS, load_sdk

# Off the metaquest/pico bridge defaults (5556/5557), which the SONIC wire
# protocol already occupies on the robot side.
SONIC_BRIDGE_POSE_PORT = 5559


@dataclass
class BodySample:
    """One deduplicated full-body frame from the PICO service."""

    body_poses: np.ndarray  # (24, 7): x, y, z, qx, qy, qz, qw — service frame
    timestamp_realtime: float
    timestamp_monotonic: float
    timestamp_ns: int
    dt: float  # device frame spacing, seconds (0 for the first frame)
    fps: float  # EMA of the device rate


@dataclass
class ControllerSample:
    """Controller buttons/sticks/triggers, read at the manager rate."""

    a: bool  # right primary
    b: bool  # right secondary
    x: bool  # left primary
    y: bool  # left secondary
    left_axis_click: bool
    right_axis_click: bool
    left_menu: bool
    right_menu: bool
    left_trigger: float
    right_trigger: float
    left_grip: float
    right_grip: float
    left_axis: tuple[float, float]
    right_axis: tuple[float, float]


_NEUTRAL_CONTROLLERS = ControllerSample(
    a=False,
    b=False,
    x=False,
    y=False,
    left_axis_click=False,
    right_axis_click=False,
    left_menu=False,
    right_menu=False,
    left_trigger=0.0,
    right_trigger=0.0,
    left_grip=0.0,
    right_grip=0.0,
    left_axis=(0.0, 0.0),
    right_axis=(0.0, 0.0),
)


# A body quaternion shorter than this cannot be normalized into a rotation.
_MIN_QUAT_NORM = 1e-6


def _advance_rate(last_stamp_ns: int | None, fps_ema: float, stamp_ns: int) -> tuple[float, float]:
    """Device-frame spacing and fps-EMA update for one deduplicated frame."""
    device_dt = (stamp_ns - last_stamp_ns) * 1e-9 if last_stamp_ns is not None else 0.0
    if device_dt > 0.0:
        inst = 1.0 / device_dt
        fps_ema = inst if fps_ema == 0.0 else 0.9 * fps_ema + 0.1 * inst
    return device_dt, fps_ema


class _BodyStreamReader(TeleopInputSource):
    """Latest-sample, rate, and staleness bookkeeping shared by both readers.

    Implements the ``get_latest`` / ``disconnected`` half of
    :class:`~eden.extensions.teleop.base.TeleopInputSource`; ``start`` / ``stop``
    / ``get_timestamp_ns`` / ``controllers`` stay abstract for the concrete
    :class:`PicoReader` / :class:`BridgeReader` to supply.
    """

    STALE_TIMEOUT = 5.0
    RATE_LOG_INTERVAL = 5.0

    def __init__(self) -> None:
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._latest: BodySample | None = None
        self._fps_ema = 0.0
        self._last_stamp_ns: int | None = None
        self._last_new_data_time = time.monotonic()
        self._last_rate_log_time = time.monotonic()
        self._disconnected = threading.Event()
        self._dropped = 0
        self._last_drop_log_time: float | None = None

    def get_latest(self) -> BodySample | None:
        with self._lock:
            return self._latest

    @property
    def disconnected(self) -> bool:
        return self._disconnected.is_set()

    def _mark_fresh(self) -> None:
        self._last_new_data_time = time.monotonic()
        if self._disconnected.is_set():
            print(f"[{type(self).__name__}] fresh data received, connection restored")
            self._disconnected.clear()

    def _check_stale(self) -> None:
        if time.monotonic() - self._last_new_data_time > self.STALE_TIMEOUT and not self._disconnected.is_set():
            print(f"[{type(self).__name__}] no new body data for {self.STALE_TIMEOUT:.0f}s, flagging disconnect")
            self._disconnected.set()

    def _drop(self, reason: str) -> None:
        """Drop one input frame/message, loudly but rate-limited (first one, then every 5 s with a count).

        A silent drop would hide a layout mismatch: setup would then wait for body data forever.
        A dropped frame is not fresh data, so a stream of nothing but bad frames still
        trips the stale check (and the operator's input-loss hand-off).
        """
        self._check_stale()
        self._dropped += 1
        now = time.monotonic()
        if self._last_drop_log_time is None or now - self._last_drop_log_time > self.RATE_LOG_INTERVAL:
            print(f"[{type(self).__name__}] dropped {self._dropped} input frame(s); latest: {reason}")
            self._last_drop_log_time = now
            self._dropped = 0

    def _publish_frame(self, stamp_ns: int, body_poses: np.ndarray) -> None:
        """Record one deduplicated device frame as the latest sample (or drop it if unusable)."""
        # Recorded first, bad frame or not, so the dedupe never re-reads the same bad frame.
        prev_stamp_ns, self._last_stamp_ns = self._last_stamp_ns, stamp_ns
        if body_poses.shape != (N_BODY_JOINTS, 7):
            self._drop(f"body pose array has shape {body_poses.shape}, expected ({N_BODY_JOINTS}, 7)")
            return
        # One such frame would raise in the pipelines' sRot.from_quat and end the operator mid-teleop.
        bad_rows = ~np.isfinite(body_poses).all(axis=1) | (np.linalg.norm(body_poses[:, 3:], axis=1) < _MIN_QUAT_NORM)
        if bad_rows.any():
            self._drop(f"non-finite or zero-norm quaternion on body joint(s) {np.flatnonzero(bad_rows).tolist()}")
            return
        self._mark_fresh()
        device_dt, self._fps_ema = _advance_rate(prev_stamp_ns, self._fps_ema, stamp_ns)
        # Reference parity: the PicoReader reports its device rate every 5 s —
        # the number that reveals a starved 50 Hz pose resampler.
        now = time.monotonic()
        if now - self._last_rate_log_time > self.RATE_LOG_INTERVAL:
            self._last_rate_log_time = now
            print(f"[{type(self).__name__}] device dt {device_dt * 1e3:.1f} ms, fps {self._fps_ema:.1f}")
        sample = BodySample(
            body_poses=body_poses,
            timestamp_realtime=time.time(),
            timestamp_monotonic=time.monotonic(),
            timestamp_ns=stamp_ns,
            dt=device_dt,
            fps=self._fps_ema,
        )
        with self._lock:
            self._latest = sample


class PicoReader(_BodyStreamReader):
    """In-process reader for the raw PICO body stream + on-demand controller reads."""

    def __init__(self) -> None:
        super().__init__()
        self._sdk = load_sdk()
        self._sdk.init()
        self._thread = threading.Thread(target=self._run, name="sonic-pico-reader", daemon=True)

    def start(self) -> None:
        if not self._thread.is_alive():
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)

    def get_timestamp_ns(self) -> int:
        """The service's latest device timestamp (used as a liveness check)."""
        return int(self._sdk.get_time_stamp_ns())

    def controllers(self) -> ControllerSample:
        """Read the current controller state (level values; edges are the caller's job)."""
        sdk = self._sdk
        lx, ly = sdk.get_left_axis()
        rx, ry = sdk.get_right_axis()
        return ControllerSample(
            a=bool(sdk.get_A_button()),
            b=bool(sdk.get_B_button()),
            x=bool(sdk.get_X_button()),
            y=bool(sdk.get_Y_button()),
            left_axis_click=bool(sdk.get_left_axis_click()),
            right_axis_click=bool(sdk.get_right_axis_click()),
            left_menu=bool(sdk.get_left_menu_button()),
            right_menu=bool(sdk.get_right_menu_button()),
            left_trigger=float(sdk.get_left_trigger()),
            right_trigger=float(sdk.get_right_trigger()),
            left_grip=float(sdk.get_left_grip()),
            right_grip=float(sdk.get_right_grip()),
            left_axis=(float(lx), float(ly)),
            right_axis=(float(rx), float(ry)),
        )

    def _run(self) -> None:
        while not self._stop.is_set():
            if not self._sdk.is_body_data_available():
                self._check_stale()
                time.sleep(0.001)
                continue

            stamp_ns = int(self._sdk.get_time_stamp_ns())
            if self._last_stamp_ns is not None and stamp_ns == self._last_stamp_ns:
                self._check_stale()
                time.sleep(1e-6)
                continue

            self._publish_frame(stamp_ns, np.array(self._sdk.get_body_joints_pose(), dtype=np.float64))


class BridgeReader(_BodyStreamReader):
    """PicoReader-compatible reader fed by ``eden pico bridge`` over ZMQ.

    Same interface and sample semantics; the SDK (and its service-library
    linkage) lives in the bridge process instead. Consumes the bridge state's
    ``body_raw`` / ``body_raw_timestamp_ns`` passthrough, so the pipelines see
    exactly the bytes the SDK reported; buttons map from the bridge's
    metaquest-compatible keys (left primary = X, right primary = A).
    """

    def __init__(self, host: str = "127.0.0.1", port: int = SONIC_BRIDGE_POSE_PORT) -> None:
        super().__init__()
        # Lazy import: the sonic tests import this module before gating on zmq.
        import zmq

        self._zmq = zmq
        self._sock = zmq.Context.instance().socket(zmq.SUB)
        self._sock.setsockopt(zmq.SUBSCRIBE, b"")
        self._sock.setsockopt(zmq.RCVTIMEO, 100)
        self._sock.setsockopt(zmq.LINGER, 0)
        self._sock.connect(f"tcp://{host}:{port}")
        self._state: dict = {}
        self._thread = threading.Thread(target=self._run, name="sonic-bridge-reader", daemon=True)

    def start(self) -> None:
        if not self._thread.is_alive():
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._sock.close()

    def get_timestamp_ns(self) -> int:
        """The bridge's latest service timestamp (advances at the publish rate)."""
        with self._lock:
            return int(self._state.get("timestamp_ns", 0))

    def controllers(self) -> ControllerSample:
        """The latest published controller state (neutral before the first message)."""
        with self._lock:
            state = self._state
        if not state:
            return _NEUTRAL_CONTROLLERS
        lx, ly = state.get("left_thumbstick_value", (0.0, 0.0))
        rx, ry = state.get("right_thumbstick_value", (0.0, 0.0))
        return ControllerSample(
            a=bool(state.get("right_a_button", False)),
            b=bool(state.get("right_b_button", False)),
            x=bool(state.get("left_a_button", False)),
            y=bool(state.get("left_b_button", False)),
            left_axis_click=bool(state.get("left_thumbstick", False)),
            right_axis_click=bool(state.get("right_thumbstick", False)),
            left_menu=bool(state.get("left_menu_button", False)),
            right_menu=bool(state.get("right_menu_button", False)),
            left_trigger=float(state.get("left_trigger_value", 0.0)),
            right_trigger=float(state.get("right_trigger_value", 0.0)),
            left_grip=float(state.get("left_squeeze_value", 0.0)),
            right_grip=float(state.get("right_squeeze_value", 0.0)),
            left_axis=(float(lx), float(ly)),
            right_axis=(float(rx), float(ry)),
        )

    def _run(self) -> None:
        zmq = self._zmq
        while not self._stop.is_set():
            try:
                text = self._sock.recv_string()
            except zmq.Again:
                self._check_stale()
                continue
            except zmq.ZMQError:
                break  # socket closed during stop()
            # Drain to the newest message (catch-up after a stall).
            while True:
                try:
                    text = self._sock.recv_string(flags=zmq.NOBLOCK)
                except zmq.Again:
                    break
            self._handle_message(text)

    def _handle_message(self, text: str) -> None:
        try:
            state = json.loads(text)
        except json.JSONDecodeError as e:
            self._drop(f"bridge message is not JSON ({e})")
            return
        if not isinstance(state, dict):
            self._drop(f"bridge message is a JSON {type(state).__name__}, expected an object")
            return
        self._ingest_body(state)
        # Stored last, so a caller that sees the new timestamp_ns also sees the
        # body sample the same message carried.
        with self._lock:
            self._state = state

    def _ingest_body(self, state: dict) -> None:
        stamp_ns = state.get("body_raw_timestamp_ns")
        raw = state.get("body_raw")
        if stamp_ns is None or raw is None or int(stamp_ns) == self._last_stamp_ns:
            # No body tracking yet, or a publish tick between device frames —
            # the same dedupe PicoReader applies to repeated SDK reads.
            self._check_stale()
            return
        try:
            body_poses = np.array(raw, dtype=np.float64)
        except (TypeError, ValueError) as e:
            self._last_stamp_ns = int(stamp_ns)
            self._drop(f"bridge 'body_raw' is not a numeric array ({e})")
            return
        self._publish_frame(int(stamp_ns), body_poses)
