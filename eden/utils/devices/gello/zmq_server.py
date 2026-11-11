"""ZMQ server bridging the GELLO leader arm to the sim follower."""

from __future__ import annotations

from typing import TYPE_CHECKING
import pickle
import threading

import numpy as np
import torch

from strenum import StrEnum

import eden as en
from eden.utils.devices.base import DeviceBase
from eden.utils.distribution import install_hint

if TYPE_CHECKING:
    from eden.envs.base import EnvBase
    from eden.entities.rigid import RigidEntity
    from genesis.engine.entities.rigid_entity.rigid_link import RigidLink

try:
    import zmq

    ZMQ_AVAILABLE = True
except ImportError:
    ZMQ_AVAILABLE = False


class _ServerState(StrEnum):
    IDLE = "idle"
    RUNNING = "running"
    STOPPED = "stopped"
    STARTING = "starting"


class GelloZMQServer(DeviceBase):
    """
    ZMQ RPC server exposing a sim arm (and optional two-DOF gripper) as a GELLO follower.

    A :class:`~eden.utils.devices.gello.zmq_client.GelloClient` (e.g. ``eden gello leader``) sends
    ``command_joint_state`` requests, which :meth:`get_command` returns; the observation methods read
    the follower's state from ``env``.

    Parameters
    ----------
    env: EnvBase
        The environment to use.
    arm_entity_name: str
        The name of the arm entity.
    arm_dofs_name: list[str]
        The names of the arm DOFs.
    ee_link_name: str
        The name of the end-effector link.
    gripper_entity_name: str | None = None
        (Optional) The name of the gripper entity.
    gripper_dofs_name: list[str] | None = None
        (Optional) The names of the gripper DOFs.
    host: str = "localhost"
        The host to use.
    port: int = 5001
        The port to use.
    verbose: bool = False
        Whether to log verbose output.
    """

    def __init__(
        self,
        env: EnvBase,
        arm_entity_name: str,
        arm_dofs_name: list[str],
        ee_link_name: str,
        *,
        gripper_entity_name: str | None = None,
        gripper_dofs_name: list[str] | None = None,
        host: str = "localhost",
        port: int = 5001,
        verbose: bool = False,
    ):
        if not ZMQ_AVAILABLE:
            raise ImportError(
                "pyzmq (import name `zmq`) is not installed. "
                f"Install it with Eden's 'gello' extra, {install_hint('gello')}; or alone with `pip install pyzmq`."
            )

        self._host = host
        self._port = port
        self._verbose = verbose

        self._env = env
        self._arm_entity: RigidEntity = env.entities[arm_entity_name]
        _, self._arm_dofs_idx = self._arm_entity.find_named_dofs_idx_local(arm_dofs_name, preserve_order=True)
        self._ee_link: RigidLink = self._arm_entity.get_link(ee_link_name)
        if gripper_entity_name is not None:
            self._gripper_entity: RigidEntity = env.entities[gripper_entity_name]
            gripper_dofs_name, self._gripper_dofs_idx = self._gripper_entity.find_named_dofs_idx_local(
                gripper_dofs_name, preserve_order=True
            )
            assert len(self._gripper_dofs_idx) == 2, "Gripper must have exactly 2 DOFs."

            gripper_open = torch.zeros(2)
            gripper_close = torch.zeros(2)
            open_dofs_pos = getattr(self._gripper_entity._options, "open_dofs_pos", None)
            close_dofs_pos = getattr(self._gripper_entity._options, "close_dofs_pos", None)

            if isinstance(open_dofs_pos, float):
                gripper_open[:] = open_dofs_pos
            elif isinstance(open_dofs_pos, dict):
                for dofs_name, pos in open_dofs_pos.items():
                    # Robot-level configs may include both left/right gripper keys. Skip entries that do not
                    # match this server's gripper scope, as decided by the lookup itself (an exact joint name is
                    # literal against every joint, which a separate probe over the scope alone would not see).
                    try:
                        _dofs_name, dofs_idx = self._gripper_entity.find_named_dofs_idx_local(
                            dofs_name, name_scope=gripper_dofs_name
                        )
                    except ValueError:
                        continue
                    dofs_idx = [gripper_dofs_name.index(name) for name in _dofs_name]
                    gripper_open[dofs_idx] = pos
            else:
                raise ValueError(f"Unsupported open_dofs_pos type: {type(open_dofs_pos)}.")

            if isinstance(close_dofs_pos, float):
                gripper_close[:] = close_dofs_pos
            elif isinstance(close_dofs_pos, dict):
                for dofs_name, pos in close_dofs_pos.items():
                    # Robot-level configs may include both left/right gripper keys. Skip entries that do not
                    # match this server's gripper scope, as decided by the lookup itself (an exact joint name is
                    # literal against every joint, which a separate probe over the scope alone would not see).
                    try:
                        _dofs_name, dofs_idx = self._gripper_entity.find_named_dofs_idx_local(
                            dofs_name, name_scope=gripper_dofs_name
                        )
                    except ValueError:
                        continue
                    dofs_idx = [gripper_dofs_name.index(name) for name in _dofs_name]
                    gripper_close[dofs_idx] = pos
            else:
                raise ValueError(f"Unsupported close_dofs_pos type: {type(close_dofs_pos)}.")

            self._gripper_open = gripper_open
            self.max_opening = (gripper_close - gripper_open).abs().sum(dim=-1)
        else:
            self._gripper_entity = None

        self._num_dofs = len(self._arm_dofs_idx) + int(self._gripper_entity is not None)

        self._lifecycle_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._server_ready = threading.Event()
        self._server_startup_done = threading.Event()
        self._server_thread: threading.Thread | None = None
        self._server_startup_error: Exception | None = None
        self._server_state: _ServerState = _ServerState.IDLE
        self._command_lock = threading.Lock()
        self._command = (
            np.asarray(self._get_observations_for_rpc(dofs_pos_only=True), dtype=np.float32).reshape(-1).copy()
        )

    @property
    def host(self) -> str:
        return self._host

    @property
    def port(self) -> int:
        return self._port

    @property
    def endpoint(self) -> str:
        return f"tcp://{self._host}:{self._port}"

    def __del__(self):
        try:
            self.stop()
        except Exception:
            pass

    def start(self):
        startup_timeout_seconds = 10.0

        with self._lifecycle_lock:
            if self._server_state == _ServerState.STOPPED:
                raise RuntimeError("Gello RPC server has already been stopped and cannot be restarted.")

            if self._server_state == _ServerState.RUNNING:
                thread = self._server_thread
                if thread is not None and thread.is_alive() and self._server_ready.is_set():
                    return
                self._server_state = _ServerState.IDLE
                self._server_thread = None

            if self._server_state == _ServerState.IDLE:
                self._server_startup_error = None
                self._server_startup_done.clear()
                self._server_ready.clear()
                self._stop_event.clear()
                self._server_state = _ServerState.STARTING

                thread = threading.Thread(target=self._serve, daemon=True)
                try:
                    thread.start()
                except Exception:
                    self._server_state = _ServerState.IDLE
                    raise
                self._server_thread = thread

        if not self._server_startup_done.wait(timeout=startup_timeout_seconds):
            raise RuntimeError(
                f"Gello RPC server startup timed out after {startup_timeout_seconds} seconds "
                f"for endpoint {self.endpoint!r}."
            )

        with self._lifecycle_lock:
            if self._server_state == _ServerState.STOPPED:
                raise RuntimeError("Gello RPC server was stopped during startup.")
            if self._server_startup_error is not None:
                self._server_state = _ServerState.IDLE
                self._server_thread = None
                raise RuntimeError(
                    f"Failed to start Gello RPC server on {self.endpoint}."
                ) from self._server_startup_error
            if not self._server_ready.is_set():
                self._server_state = _ServerState.IDLE
                self._server_thread = None
                raise RuntimeError("Gello RPC server thread exited before becoming ready.")
            self._server_state = _ServerState.RUNNING

    def stop(self):
        with self._lifecycle_lock:
            if self._server_state == _ServerState.STOPPED:
                return
            self._server_state = _ServerState.STOPPED
            self._stop_event.set()
            thread = self._server_thread

        self._wake_server_for_shutdown()

        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)

        self._server_ready.clear()
        self._server_startup_done.set()

    def _wake_server_for_shutdown(self):
        thread = self._server_thread
        if thread is None or not thread.is_alive() or not self._server_ready.is_set():
            return

        wake_ctx = None
        wake_socket = None
        try:
            wake_ctx = zmq.Context()
            wake_socket = wake_ctx.socket(zmq.REQ)
            wake_socket.setsockopt(zmq.LINGER, 0)
            wake_socket.setsockopt(zmq.RCVTIMEO, 100)
            wake_socket.setsockopt(zmq.SNDTIMEO, 100)
            wake_socket.connect(self.endpoint)
            wake_socket.send(pickle.dumps({"method": "num_dofs"}))
            try:
                wake_socket.recv()
            except Exception:
                pass
        except Exception:
            pass
        finally:
            if wake_socket is not None:
                wake_socket.close(0)
            if wake_ctx is not None:
                wake_ctx.term()

    def _create_server_socket(self):
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        try:
            socket.setsockopt(zmq.LINGER, 0)
            socket.setsockopt(zmq.RCVTIMEO, 100)
            socket.bind(self.endpoint)
        except Exception:
            socket.close(0)
            context.term()
            raise
        return context, socket

    def _serve(self):
        try:
            context, socket = self._create_server_socket()
        except Exception as exc:
            with self._lifecycle_lock:
                self._server_startup_error = exc
            self._server_startup_done.set()
            return

        self._server_ready.set()
        self._server_startup_done.set()

        try:
            while not self._stop_event.is_set():
                try:
                    raw = socket.recv()
                except zmq.Again:
                    continue
                except zmq.ZMQError:
                    break

                try:
                    response = self._handle_raw_request(raw)
                except Exception as exc:
                    response = {"error": str(exc)}
                    if self._verbose:
                        en.logger.info(f"Gello RPC processing error: {exc}")
                try:
                    socket.send(pickle.dumps(response))
                except zmq.ZMQError:
                    break
        finally:
            self._server_ready.clear()
            socket.close(0)
            context.term()

    def _handle_raw_request(self, raw: bytes):
        """Decode one pickled RPC request, dispatch it by ``method`` and return the response."""
        request = pickle.loads(raw)
        if not isinstance(request, dict):
            raise TypeError(f"RPC request must be a dict, got {type(request).__name__}.")

        method = str(request.get("method", "")).strip().lower()
        if method == "num_dofs":
            return self._num_dofs
        elif method == "command_joint_state":
            self._set_command_from_request(request)
            return None
        elif method == "get_joint_positions":
            return self._get_observations_for_rpc(dofs_pos_only=True)
        elif method == "get_observations":
            # NOTE: used for gello software data collection
            return self._get_observations_for_rpc()
        else:
            raise ValueError(f"Invalid method: {method}")

    def _set_command_from_request(self, request: dict):
        args = request.get("args")
        if not isinstance(args, dict):
            raise ValueError("'args' field is required for command_joint_state method.")
        if "joint_state" not in args:
            raise ValueError("'joint_state' argument is required for command_joint_state method.")

        dofs_pos = np.asarray(args["joint_state"], dtype=np.float32).reshape(-1)
        if dofs_pos.size != self._num_dofs:
            raise ValueError(f"'joint_state' must have length {self._num_dofs}.")
        if not np.isfinite(dofs_pos).all():
            raise ValueError("'joint_state' must contain only finite values.")
        with self._command_lock:
            self._command = dofs_pos.copy()

    def _get_observations_for_rpc(self, dofs_pos_only: bool = False) -> dict[str, np.ndarray]:
        """Return the simulation state of the follower for gello_software data collection.

        With ``dofs_pos_only`` only the joint positions array (arm DOFs, then the gripper opening in
        ``[0, 1]`` if a gripper is configured) is returned.
        """
        arm_positions = self._arm_entity.get_dofs_pos(dofs_idx_local=self._arm_dofs_idx)[0].detach().cpu().numpy()

        if self._gripper_entity is not None:
            gripper_dofs = self._gripper_entity.get_dofs_pos(dofs_idx_local=self._gripper_dofs_idx)[0]
            gripper_position = (
                ((gripper_dofs - self._gripper_open).abs().sum(dim=-1) / self.max_opening).detach().cpu().numpy()
            )
            joint_positions = np.concatenate([arm_positions, gripper_position.reshape(-1)])
        else:
            gripper_position = None
            joint_positions = arm_positions

        if dofs_pos_only:
            return joint_positions

        arm_velocities = self._arm_entity.get_dofs_vel(dofs_idx_local=self._arm_dofs_idx)[0].detach().cpu().numpy()
        if self._gripper_entity is not None:
            joint_velocities = np.concatenate([arm_velocities, np.zeros(1)])
        else:
            joint_velocities = arm_velocities

        # relative=False: world-frame EE pose (Genesis >=1.1.1 link getters default to user frame). Quaternion wxyz.
        ee_pos = self._ee_link.get_pos(relative=False)[0].detach().cpu().numpy()
        ee_quat = self._ee_link.get_quat(relative=False)[0].detach().cpu().numpy()
        ee_pos_quat = np.concatenate([ee_pos, ee_quat])

        return dict(
            joint_positions=joint_positions,
            joint_velocities=joint_velocities,
            ee_pos_quat=ee_pos_quat,
            gripper_position=gripper_position,
        )

    def get_raw_data(self) -> torch.Tensor:
        with self._command_lock:
            return torch.from_numpy(self._command.copy())

    def get_command(self) -> torch.Tensor:
        return self.get_raw_data()
