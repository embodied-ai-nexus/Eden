"""Wire protocol for the Eden <-> XR (televuer) ZMQ bridge.

Two localhost channels, both single-part so ZMQ_CONFLATE (keep-latest) works:

  * frames : Eden  PUSH -> bridge PULL  (FRAME_PORT)  stereo RGB uint8 (H, W, 3)
  * poses  : bridge PUB  -> Eden  SUB   (POSE_PORT)   head/wrist/hand JSON state

This module is imported by *both* processes (the XR bridge and Eden), so it must
depend only on numpy + pyzmq -- never on televuer or eden.

Pose matrices published here are already expressed in the Genesis/robot world
frame: z up, x forward, y left, metres, origin at the XR tracking origin.
"""

import json

import numpy as np
import zmq

FRAME_PORT = 5556
POSE_PORT = 5557

# ----------------------------------------------------------------------------
# frames (single-part: 8-byte header [h, w] uint32, then raw uint8 RGB bytes)
# ----------------------------------------------------------------------------


def send_frame(sock, rgb):
    """Send one stereo RGB frame. ``rgb`` is (H, W, 3) uint8 (left|right halves)."""
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    h, w = rgb.shape[:2]
    header = np.array([h, w], dtype=np.uint32).tobytes()
    sock.send(header + rgb.tobytes(), copy=False)


def recv_frame(sock, flags=0):
    """Receive one frame, or return None if ``flags=zmq.NOBLOCK`` and none waiting."""
    try:
        buf = sock.recv(flags=flags)
    except zmq.Again:
        return None
    h, w = np.frombuffer(buf[:8], dtype=np.uint32)
    return np.frombuffer(buf[8:], dtype=np.uint8).reshape(int(h), int(w), 3)


# ----------------------------------------------------------------------------
# poses (single-part JSON string)
# ----------------------------------------------------------------------------


def send_pose(sock, data: dict):
    """Publish one pose state dict as JSON."""
    sock.send_string(json.dumps(data))


def recv_pose(sock, flags=0):
    """Receive the latest pose dict, or None if ``flags=zmq.NOBLOCK`` and none waiting."""
    try:
        return json.loads(sock.recv_string(flags=flags))
    except zmq.Again:
        return None


# ----------------------------------------------------------------------------
# socket factories
# ----------------------------------------------------------------------------


def frame_pull(ctx, port=FRAME_PORT):
    """Bridge side: PULL that keeps only the newest frame."""
    s = ctx.socket(zmq.PULL)
    s.setsockopt(zmq.CONFLATE, 1)
    s.setsockopt(zmq.RCVHWM, 1)
    s.bind(f"tcp://*:{port}")
    return s


def frame_push(ctx, host="127.0.0.1", port=FRAME_PORT):
    """Eden side: PUSH that drops frames rather than blocking when the bridge lags."""
    s = ctx.socket(zmq.PUSH)
    s.setsockopt(zmq.SNDHWM, 1)
    s.setsockopt(zmq.CONFLATE, 1)
    s.connect(f"tcp://{host}:{port}")
    return s


def pose_pub(ctx, port=POSE_PORT):
    """Bridge side: PUB of pose state."""
    s = ctx.socket(zmq.PUB)
    s.bind(f"tcp://*:{port}")
    return s


def pose_sub(ctx, host="127.0.0.1", port=POSE_PORT):
    """Eden side: SUB that keeps only the newest pose."""
    s = ctx.socket(zmq.SUB)
    s.setsockopt(zmq.CONFLATE, 1)
    s.setsockopt_string(zmq.SUBSCRIBE, "")
    s.connect(f"tcp://{host}:{port}")
    return s
