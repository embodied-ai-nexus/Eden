"""ZMQ client for the GELLO teleoperation server."""

from __future__ import annotations

import pickle
from typing import Sequence

import numpy as np

try:
    import zmq

    ZMQ_AVAILABLE = True
except ImportError:
    ZMQ_AVAILABLE = False


class GelloClient:
    """Minimal ZMQ REQ client for :class:`GelloZMQServer`.

    Exposes the four RPC methods served by the Gello follower server:
    ``num_dofs``, ``get_joint_positions``, ``get_observations``, and ``command_joint_state``.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 6001, timeout_ms: int = 2000):
        if not ZMQ_AVAILABLE:
            raise ImportError("pyzmq is not installed. Install it with `pip install pyzmq`.")
        if timeout_ms <= 0:
            raise ValueError("'timeout_ms' must be greater than 0.")

        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REQ)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.setsockopt(zmq.RCVTIMEO, int(timeout_ms))
        self._socket.setsockopt(zmq.SNDTIMEO, int(timeout_ms))
        self._socket.connect(f"tcp://{host}:{port}")

    def num_dofs(self) -> int:
        return int(self._request({"method": "num_dofs"}))

    def get_joint_positions(self) -> np.ndarray:
        result = self._request({"method": "get_joint_positions"})
        return np.asarray(result, dtype=np.float64).reshape(-1)

    def get_observations(self) -> dict[str, np.ndarray]:
        return self._request({"method": "get_observations"})

    def command_joint_state(self, joint_state: Sequence[float] | np.ndarray) -> None:
        arr = np.asarray(joint_state, dtype=np.float32).reshape(-1)
        self._request({"method": "command_joint_state", "args": {"joint_state": arr}})

    def close(self):
        socket = getattr(self, "_socket", None)
        if socket is not None:
            socket.close(0)
            self._socket = None
        context = getattr(self, "_context", None)
        if context is not None:
            context.term()
            self._context = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _request(self, payload: dict):
        try:
            self._socket.send(pickle.dumps(payload))
            result = pickle.loads(self._socket.recv())
        except zmq.Again as exc:
            raise RuntimeError("ZMQ timeout while communicating with follower server.") from exc
        except zmq.ZMQError as exc:
            raise RuntimeError(f"ZMQ communication failure: {exc}") from exc

        if isinstance(result, dict) and "error" in result:
            raise RuntimeError(str(result["error"]))
        return result
