"""Where a tracked object's pose comes from: cameras, estimators, and the sources that run them.

Three layers, each replaceable on its own:

- :class:`FrameSource`: a camera. :meth:`~FrameSource.grab` blocks for the next frame and returns a
  :class:`CameraFrame` (RGB, optional metric depth, intrinsics, capture time).
- :class:`PoseEstimator`: one frame -> the object's pose in the estimator's *report frame* (the
  camera, or an anchor the estimator itself detects), or ``None`` when the object is not found.
  AprilTag ships (:mod:`.apriltag`).
- :class:`PoseSource`: what the deployment reads. :meth:`~PoseSource.latest` returns the newest
  :class:`PoseSample` without blocking. :class:`EstimatorPoseSource` runs a camera and an estimator
  on a thread of this process;
  :class:`~eden.extensions.deployment.object_pose.zmq_stream.ZmqPoseSource` subscribes to one
  running in another process, which is where a GPU estimator belongs: it then shares neither the
  control loop's GIL nor its CUDA stream.

Every quaternion here is wxyz. Every time is ``time.monotonic()``: a sample's :attr:`PoseSample.stamp`
is when its *frame was captured* where the source knows it, so :attr:`PoseSample.age` counts the
estimator's latency too, not only the time since the result arrived.
"""

from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

import eden as en


@dataclass(frozen=True)
class PoseSample:
    """One pose estimate of an object, in its source's report frame.

    Attributes
    ----------
    pos : np.ndarray
        ``(3,)`` position in metres.
    quat : np.ndarray
        ``(4,)`` unit quaternion, wxyz.
    stamp : float
        ``time.monotonic()`` at the capture of the frame it was estimated from (at receipt, for a
        source that cannot know the capture time).
    score : float
        The estimator's confidence, in its own units (NaN when it reports none). For logging and
        gating only; nothing here compares scores across estimators.
    """

    pos: np.ndarray
    quat: np.ndarray
    stamp: float
    score: float = float("nan")

    @property
    def age(self) -> float:
        """Seconds since the frame this pose came from was captured."""
        return time.monotonic() - self.stamp


def checked_pose(pos, quat) -> tuple[np.ndarray, np.ndarray] | None:
    """``(pos, unit quat)`` as float64 arrays, or ``None`` for a non-finite or zero-norm pose."""
    pos = np.asarray(pos, dtype=np.float64).reshape(-1)
    quat = np.asarray(quat, dtype=np.float64).reshape(-1)
    if pos.shape != (3,) or quat.shape != (4,):
        return None
    norm = np.linalg.norm(quat)
    if not (np.isfinite(pos).all() and np.isfinite(quat).all() and norm > 0):
        return None
    return pos, quat / norm


class PoseSource(ABC):
    """The latest pose of one object, read by the deployment once per control tick.

    :meth:`latest` must not block: the control loop calls it every tick. A source that estimates
    does so on its own thread (or process) and :meth:`latest` hands back what it has.
    """

    #: What the source is, for log and error messages (``"the AprilTag camera"``).
    name: str = "the pose source"

    def start(self) -> None:
        """Begin producing poses (open the camera, start the thread). Called once, from the deployment's ``connect``."""

    @abstractmethod
    def latest(self) -> PoseSample | None:
        """The newest pose, or ``None`` before the first."""

    def message_counts(self) -> dict[str, int]:
        """Frames or messages handled so far by outcome (``accepted`` / ``untracked`` / ``rejected`` / ``failed``)."""
        return {}

    def request_reset(self) -> None:
        """Ask the estimator to drop its tracking state (a model-based tracker registers again). No-op by default.

        Nothing calls it automatically: a re-registration takes long enough to trip a tight
        ``max_age_s``. Call it from an operator hook when a tracker has drifted.
        """

    def close(self) -> None:
        """Release the camera, the thread and the socket. Idempotent."""


@dataclass
class CameraFrame:
    """One camera frame.

    Attributes
    ----------
    rgb : np.ndarray
        ``(H, W, 3)`` uint8, RGB channel order.
    K : np.ndarray
        ``(3, 3)`` pinhole intrinsics of ``rgb``.
    stamp : float
        ``time.monotonic()`` at capture (at the return of the grab when the driver gives no better).
    depth : np.ndarray | None
        ``(H, W)`` float32 depth in metres, registered to ``rgb``; ``None`` from a colour-only camera.
    dist_coeffs : np.ndarray | None
        OpenCV distortion coefficients of ``rgb``; ``None`` for an undistorted (or rectified) image.
    """

    rgb: np.ndarray
    K: np.ndarray
    stamp: float
    depth: np.ndarray | None = None
    dist_coeffs: np.ndarray | None = None


class FrameSource(ABC):
    """A camera. :meth:`grab` blocks until the next frame."""

    name: str = "the camera"

    @abstractmethod
    def grab(self) -> CameraFrame:
        raise NotImplementedError

    def close(self) -> None:
        """Release the device. Idempotent."""


class PoseEstimator(ABC):
    """One frame -> the object's pose in the estimator's report frame.

    The report frame is the estimator's to choose and to document: the camera's optical frame
    (x right, y down, z forward) unless it anchors on something it detects itself, as
    :class:`~eden.extensions.deployment.object_pose.apriltag.AprilTagEstimator` does with an anchor
    tag. The deployment places that frame in the scene through
    :attr:`~eden.extensions.deployment.object_pose.tracked_object.TrackedObjectSpec.parent`.

    Called from one thread only (the source's), so an estimator may keep tracking state unguarded.
    """

    #: Whether :meth:`estimate` needs ``frame.depth``. A source refuses a colour-only camera up front.
    needs_depth: bool = False

    @abstractmethod
    def estimate(self, frame: CameraFrame) -> tuple[np.ndarray, np.ndarray, float] | None:
        """``(pos, quat_wxyz, score)`` of the object in the report frame, or ``None`` if it is not found."""
        raise NotImplementedError

    def reset(self) -> None:
        """Drop any tracking state, so the next :meth:`estimate` starts from scratch."""

    def close(self) -> None:
        """Release the model. Idempotent."""


class EstimatorPoseSource(PoseSource):
    """Runs a camera and an estimator on a thread of this process; :meth:`latest` reads the newest result.

    Suited to a cheap estimator (AprilTag). A GPU model in the control process competes with the
    policy for the GIL and the GPU: run it under :func:`serve_pose_stream` and read it through
    :class:`~eden.extensions.deployment.object_pose.zmq_stream.ZmqPoseSource` instead.

    Parameters
    ----------
    camera : FrameSource
        The frames. Owned: :meth:`close` closes it.
    estimator : PoseEstimator
        Frame -> pose. Owned: :meth:`close` closes it.
    name : str | None
        For messages; defaults to the estimator's class name.
    """

    def __init__(self, camera: FrameSource, estimator: PoseEstimator, name: str | None = None) -> None:
        self.camera, self.estimator = camera, estimator
        self.name = name or f"{type(estimator).__name__} on {camera.name}"
        self._lock = threading.Lock()
        self._sample: PoseSample | None = None
        self._counts = {"accepted": 0, "untracked": 0, "rejected": 0, "failed": 0}
        self._stop = threading.Event()
        self._reset_requested = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._loop, daemon=True, name=f"pose-{type(self.estimator).__name__}")
        self._thread.start()

    def _loop(self) -> None:
        logged_failure = False
        while not self._stop.is_set():
            if self._reset_requested.is_set():
                self._reset_requested.clear()
                self.estimator.reset()
            try:
                frame = self.camera.grab()
                if self.estimator.needs_depth and frame.depth is None:
                    raise RuntimeError(
                        f"{type(self.estimator).__name__} needs depth, and {self.camera.name} returned a frame without "
                        "it. Fix: open the camera with depth enabled (aligned to colour)."
                    )
                got = self.estimator.estimate(frame)
            except Exception as exc:  # noqa: BLE001 -- counted and logged; latest() then goes stale and the deployment acts on that
                with self._lock:
                    self._counts["failed"] += 1
                if not logged_failure:
                    en.logger.exception(
                        f"[object_pose] {self.name}: estimation failed ({exc}); no pose arrives while it does"
                    )
                    logged_failure = True
                time.sleep(0.05)  # a dead camera fails every grab: do not spin
                continue
            with self._lock:
                if got is None:
                    self._counts["untracked"] += 1
                    continue
                pose = checked_pose(got[0], got[1])
                if pose is None:
                    self._counts["rejected"] += 1
                    continue
                self._counts["accepted"] += 1
                self._sample = PoseSample(pose[0], pose[1], frame.stamp, float(got[2]))

    def latest(self) -> PoseSample | None:
        with self._lock:
            return self._sample

    def message_counts(self) -> dict[str, int]:
        with self._lock:
            return dict(self._counts)

    def request_reset(self) -> None:
        self._reset_requested.set()  # served on the estimator's own thread

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        try:
            self.camera.close()
        finally:
            self.estimator.close()


def wait_for_pose(source: PoseSource, timeout_s: float, max_age_s: float, what: str) -> None:
    """Block until ``source`` has a pose younger than ``max_age_s``; raise, saying which way it failed, after ``timeout_s``."""
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout_s:
        got = source.latest()
        if got is not None and got.age <= max_age_s:
            return
        time.sleep(0.02)
    got, counts = source.latest(), source.message_counts()
    if got is not None:
        raise RuntimeError(
            f"{source.name} last saw {what} {got.age:.2f} s ago (limit {max_age_s:g} s) and {timeout_s:g} s of waiting "
            f"brought no newer pose: it tracked {what} and lost it. Fix: put {what} back in the camera's view."
        )
    if not any(counts.values()):
        raise RuntimeError(
            f"nothing from {source.name} within {timeout_s:g} s. Fix: check that the camera is connected (and, for a "
            "ZmqPoseSource, that serve_pose_stream is running and host/port point at it)."
        )
    detail = ", ".join(f"{n} {k}" for k, n in counts.items() if n)
    raise RuntimeError(
        f"{source.name} handled {sum(counts.values())} frames within {timeout_s:g} s, none with a pose of {what} "
        f"({detail}). Fix: put {what} in the camera's view; 'failed' counts are estimator exceptions, logged above."
    )
