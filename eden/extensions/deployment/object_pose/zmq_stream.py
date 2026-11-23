"""Pose estimates over ZMQ, for an estimator that runs in another process or on another machine.

:func:`serve_pose_stream` runs a camera and an estimator and publishes every frame's estimate on a
PUB socket; :class:`ZmqPoseSource` subscribes to it and is read like any other
:class:`~eden.extensions.deployment.object_pose.pose_source.PoseSource`. A GPU estimator belongs out
of process: it then shares neither the control loop's GIL nor its CUDA stream.

Needs ``pyzmq``, imported when a socket opens, so this module imports without it.
"""

from __future__ import annotations

import json
import threading
import time

import numpy as np

import eden as en
from eden.extensions.deployment.object_pose.pose_source import (
    FrameSource,
    PoseEstimator,
    PoseSample,
    PoseSource,
    checked_pose,
)

#: The wire format of :func:`serve_pose_stream` / :class:`ZmqPoseSource`: one JSON object per frame,
#: ``{"key": str, "pos": [x, y, z], "quat_wxyz": [w, x, y, z], "latency_s": float, "score": float}``,
#: or ``{"key": str, "pos": null}`` for a frame in which the object was not found. ``latency_s`` is
#: capture-to-publish time, which the subscriber subtracts from its receipt time: the two processes'
#: monotonic clocks need not agree (another machine), but a duration does.
POSE_MESSAGE_FIELDS = ("key", "pos", "quat_wxyz", "latency_s", "score")


def encode_pose_message(key: str, got: tuple[np.ndarray, np.ndarray, float] | None, stamp: float) -> str:
    """The message :func:`serve_pose_stream` publishes for one frame's estimate (``None``: not found)."""
    if got is None:
        return json.dumps({"key": key, "pos": None})
    pos, quat, score = got
    return json.dumps(
        {
            "key": key,
            "pos": [float(v) for v in pos],
            "quat_wxyz": [float(v) for v in quat],
            "latency_s": max(0.0, time.monotonic() - stamp),
            "score": float(score),
        }
    )


def parse_pose_message(text: str, key: str) -> tuple[np.ndarray, np.ndarray, float, float] | str:
    """``(pos, quat_wxyz, latency_s, score)`` from one message, or why it carries none for ``key``.

    ``"other"`` for another object's message, ``"untracked"`` for one sent without a pose and
    ``"rejected"`` for a malformed or non-finite one.
    """
    try:
        msg = json.loads(text)
    except ValueError:
        return "rejected"
    if not isinstance(msg, dict):
        return "rejected"
    if msg.get("key") != key:
        return "other"
    if msg.get("pos") is None:
        return "untracked"
    try:
        pose = checked_pose(msg["pos"], msg["quat_wxyz"])
        latency = float(msg.get("latency_s", 0.0))
        score = float(msg.get("score", float("nan")))
    except (KeyError, TypeError, ValueError):
        return "rejected"
    if pose is None or not np.isfinite(latency) or latency < 0:
        return "rejected"
    return pose[0], pose[1], latency, score


def serve_pose_stream(
    camera: FrameSource, estimator: PoseEstimator, key: str, port: int = 5556, host: str = "*"
) -> None:
    """Run ``estimator`` over ``camera`` and publish every frame's estimate on ``tcp://host:port``, until interrupted.

    The process to run a GPU estimator in, for :class:`ZmqPoseSource` to read. Publishes a message
    for every frame, found or not, so the subscriber tells a lost object from a dead publisher.
    """
    try:
        import zmq
    except ImportError as e:
        raise ImportError("serve_pose_stream needs pyzmq: pip install pyzmq") from e
    sock = zmq.Context.instance().socket(zmq.PUB)
    sock.bind(f"tcp://{host}:{port}")
    en.logger.info(f"[object_pose] publishing {key!r} poses from {type(estimator).__name__} on tcp://{host}:{port}")
    try:
        while True:
            frame = camera.grab()
            sock.send_string(encode_pose_message(key, estimator.estimate(frame), frame.stamp))
    finally:
        sock.close(linger=0)
        camera.close()
        estimator.close()


class ZmqPoseSource(PoseSource):
    """Subscribes to a pose publisher (:func:`serve_pose_stream`) for one object.

    Parameters
    ----------
    key : str
        The object's key in the messages (the ``key`` the publisher was started with).
    host, port : str, int
        The publisher's PUB socket.
    """

    def __init__(self, key: str, host: str = "localhost", port: int = 5556) -> None:
        self.key, self.host, self.port = key, host, port
        self.name = f"the {key!r} pose publisher at tcp://{host}:{port}"
        self._lock = threading.Lock()
        self._sample: PoseSample | None = None
        # "other": another object's messages on the same socket (a publisher serving several keys).
        self._counts = {"accepted": 0, "untracked": 0, "rejected": 0, "other": 0}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._sock = None

    def start(self) -> None:
        if self._thread is not None:
            return
        try:
            import zmq
        except ImportError as e:
            raise ImportError("ZmqPoseSource needs pyzmq: pip install pyzmq") from e
        self._sock = zmq.Context.instance().socket(zmq.SUB)
        self._sock.setsockopt(zmq.RCVTIMEO, 200)
        self._sock.setsockopt_string(zmq.SUBSCRIBE, "")
        self._sock.connect(f"tcp://{self.host}:{self.port}")
        self._thread = threading.Thread(target=self._loop, daemon=True, name=f"pose-zmq-{self.key}")
        self._thread.start()

    def _loop(self) -> None:
        import zmq

        logged_failure = False
        while not self._stop.is_set():
            try:
                text = self._sock.recv_string()
            except zmq.Again:
                continue
            except zmq.ZMQError as e:
                if self._stop.is_set() or e.errno in (zmq.ETERM, zmq.ENOTSOCK):
                    break  # the context or socket is gone: retrying would spin
                if not logged_failure:
                    en.logger.error(f"[object_pose] {self.name}: receive failed ({e}); no pose arrives while it does")
                    logged_failure = True
                continue
            received = time.monotonic()
            got = parse_pose_message(text, self.key)
            with self._lock:
                if isinstance(got, str):
                    self._counts[got] += 1
                    continue
                pos, quat, latency, score = got
                self._counts["accepted"] += 1
                self._sample = PoseSample(pos, quat, received - latency, score)

    def latest(self) -> PoseSample | None:
        with self._lock:
            return self._sample

    def message_counts(self) -> dict[str, int]:
        with self._lock:
            return dict(self._counts)

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._sock is not None:
            self._sock.close(linger=0)
            self._sock = None
