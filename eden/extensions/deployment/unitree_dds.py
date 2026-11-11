"""The DDS transport the Unitree SDK2 backends share: the G1 over ``unitree_hg``, the Go2 over ``unitree_go``.

Both robots speak the same low-level protocol: a ``LowState_`` stream on ``rt/lowstate``, a
CRC-stamped ``LowCmd_`` on ``rt/lowcmd``, and per-motor ``q`` / ``dq`` / ``tau`` / ``kp`` / ``kd``
slots addressed through the backend's Eden -> SDK motor map. :class:`_UnitreeDdsDeployment` holds
that plumbing; each backend supplies its message types, its packet header and its motor map, plus
anything only its robot has (the G1's ``mode_pr`` / ``mode_machine`` and motion-mode release).

The SDK is imported through :mod:`~eden.extensions.deployment.utils.unitree_sdk2`'s guarded
module-level import, and only touched at :meth:`_UnitreeDdsDeployment.connect`.
"""

from __future__ import annotations
from abc import abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar

import threading
import time

import numpy as np

from eden.extensions.deployment.base import DeploymentBase, SoftShutdownMixin
from eden.extensions.deployment.utils.republisher import CommandRepublisherMixin
from eden.extensions.deployment.utils.state import RobotCommand, RobotState
from eden.extensions.deployment.utils.unitree_sdk2 import (
    CRC,
    ChannelPublisher,
    ChannelSubscriber,
    init_unitree_channel,
    read_low_state,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from eden.envs.base import EnvBase
    from eden.options.extensions.deployment import DeploymentOptions


#: ``LowCmd_.level_flag`` for low-level (per-motor) control, as Unitree's SDK2 examples set it.
_LOW_LEVEL_FLAG = 0xFF
#: History depth of the ``rt/lowstate`` subscriber's reader queue.
_LOW_STATE_QUEUE_DEPTH = 10


class _UnitreeDdsDeployment(CommandRepublisherMixin, SoftShutdownMixin, DeploymentBase):
    """Drive a Unitree robot over the SDK2 DDS low-level interface.

    A backend sets :attr:`_ROBOT_NAME`, resolves ``_dof_index`` / ``_num_full_dof`` in
    ``_resolve_options``, declares its own ``publish_rate`` (see :class:`CommandRepublisherMixin`),
    and implements :meth:`_dds_messages` and :meth:`_shutdown_hold_pos`. It may extend
    :meth:`connect`, :meth:`_init_low_cmd`, :meth:`_prepare_low_cmd` and :meth:`_on_low_state`.

    network_interface: str | None
        NIC the DDS channel binds to. ``None`` lets the SDK choose.
    motor_mode: int
        ``motor_cmd.mode`` written on every mapped motor slot (``0x01`` = enabled, the PMSM servo
        mode Unitree's examples use).
    """

    network_interface: str | None = None
    motor_mode: int = 0x01

    #: How errors and the low-state reader name the robot (``"Unitree G1"``).
    _ROBOT_NAME: ClassVar[str]

    def __init__(self, env: EnvBase, options: DeploymentOptions) -> None:
        super().__init__(env, options)

        self._latest_msg: Any | None = None
        self._msg_event = threading.Event()
        self._subscriber = None
        self._publisher = None
        self._low_cmd = None
        self._crc = None
        self._init_republisher()
        # Whether ``_low_cmd`` holds a command some caller actually asked for. Until it does, a
        # re-send would push ``_init_low_cmd``'s limp all-zero packet -- harmless, but nothing
        # goes on the wire before the first ``send_payload``, republisher included.
        self._cmd_ready = False

    # ------------------------------------------------------------------
    # Per-robot hooks
    # ------------------------------------------------------------------

    @abstractmethod
    def _dds_messages(self) -> tuple[type, type, Callable[[], Any]]:
        """``(LowState_, LowCmd_, LowCmd_ factory)`` of this robot's IDL, read when :meth:`connect` runs."""

    def _init_low_cmd(self) -> None:
        """Fill the fresh ``_low_cmd`` as a limp packet; a backend extends it with its own header fields.

        The low-level flag, ``gpio`` 0, and every mapped motor slot at ``motor_mode`` with zero
        target, velocity, gains and torque.
        """
        self._low_cmd.level_flag = _LOW_LEVEL_FLAG
        self._low_cmd.gpio = 0
        for i in range(self._num_full_dof):
            m = self._low_cmd.motor_cmd[i]
            m.mode = self.motor_mode
            m.q = 0.0
            m.dq = 0.0
            m.kp = 0.0
            m.kd = 0.0
            m.tau = 0.0

    def _resolve_options(self) -> None:
        """Check the republish rate (see :meth:`DeploymentBase._resolve_options`); a backend resolves its motor map after."""
        super()._resolve_options()
        self._validate_publish_rate()

    def _prepare_low_cmd(self) -> None:
        """Refresh the packet's own fields under the command lock, before the motor targets. A no-op here."""

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect(self) -> None:
        self._latest_msg = None
        self._msg_event.clear()

        init_unitree_channel(self.network_interface)
        low_state_type, low_cmd_type, new_low_cmd = self._dds_messages()
        self._subscriber = ChannelSubscriber("rt/lowstate", low_state_type)
        self._subscriber.Init(self._on_low_state, _LOW_STATE_QUEUE_DEPTH)

        self._publisher = ChannelPublisher("rt/lowcmd", low_cmd_type)
        self._publisher.Init()
        self._low_cmd = new_low_cmd()
        self._init_low_cmd()
        self._crc = CRC()

        start_time = time.time()
        while not self._msg_event.is_set():
            if time.time() - start_time > self.connect_timeout_s:
                raise RuntimeError(f"Timed out waiting for {self._ROBOT_NAME} low state.")
            time.sleep(0.05)

        # Started once the state stream is up (so the G1's ``mode_machine`` is known), but gated on
        # ``_cmd_ready`` so nothing goes on the wire before the first ``send_payload``.
        self._cmd_ready = False
        self._start_publisher()

    def close(self) -> None:
        # Flush the debug recording and release the operator input device, then ramp to a
        # damping state before dropping the channel. The republisher keeps running through the
        # ramp on purpose -- the damping command wants the same packet rate as any other -- and
        # is stopped once it is done.
        self._teardown_common()
        self._soft_shutdown()
        self._stop_publisher()
        self._cmd_ready = False
        self._subscriber = None
        self._publisher = None
        self._low_cmd = None
        self._crc = None
        self._latest_msg = None
        self._msg_event.clear()

    # ------------------------------------------------------------------
    # Wire I/O
    # ------------------------------------------------------------------

    @property
    def _connected(self) -> bool:
        """Whether :meth:`connect` has opened the channel and :meth:`close` has not dropped it."""
        return self._publisher is not None and self._low_cmd is not None and self._crc is not None

    def read_state(self) -> RobotState:
        return read_low_state(
            self._latest_msg,
            robot_name=self._ROBOT_NAME,
            dof_index=self._dof_index,
            num_full_dof=self._num_full_dof,
        )

    def send_payload(self, payload: RobotCommand) -> None:
        if not self._connected:
            raise RuntimeError(f"{self._ROBOT_NAME} deployment is not connected.")

        # Low-pass the commanded targets (no-op unless --target_lpf < 1) before they
        # hit the wire, so every path (init/hold/policy) is smoothed.
        payload.dofs_pos = self._apply_target_lpf(np.asarray(payload.dofs_pos, dtype=np.float64))

        # Held under the command lock so a concurrent re-send never publishes a half-written
        # packet (see :class:`CommandRepublisherMixin`).
        with self._cmd_lock:
            self._prepare_low_cmd()

            for i, motor_idx in enumerate(self._dof_index):
                motor_cmd = self._low_cmd.motor_cmd[motor_idx]
                motor_cmd.q = float(payload.dofs_pos[i])
                motor_cmd.dq = float(payload.dofs_vel[i])
                motor_cmd.tau = float(payload.dofs_torque[i])
                if payload.dofs_kp is not None:
                    motor_cmd.kp = float(payload.dofs_kp[i])
                if payload.dofs_kd is not None:
                    motor_cmd.kd = float(payload.dofs_kd[i])

            self._low_cmd.crc = self._crc.Crc(self._low_cmd)
            self._publisher.Write(self._low_cmd)
            self._cmd_ready = True
            self._note_command_published()

    def _publish_held_command(self) -> None:
        """Re-send the held ``LowCmd_`` (see :class:`CommandRepublisherMixin`).

        No CRC recompute and no field refresh: neither robot's ``LowCmd_`` carries a timestamp
        or a sequence number Eden writes (the G1's is ``mode_pr``, ``mode_machine``,
        ``motor_cmd[35]``, ``reserve``, ``crc``; the Go2's ``sn`` / ``version`` words are never
        written by Eden or by Unitree's example), so the packet ``send_payload`` built is still
        byte-valid. SONIC's reference G1 writer rebuilds and re-CRCs its copy every tick, which
        comes to the same bytes.
        """
        if not self._cmd_ready or self._publisher is None or self._low_cmd is None:
            return
        self._publisher.Write(self._low_cmd)
        self._pub_count += 1

    def _on_low_state(self, msg: Any) -> None:
        # The SDK's reader callback hands over a freshly deserialized message per sample
        # (cyclonedds ``reader.take``), so this rebind is atomic and ``read_state`` never observes
        # a partially written buffer.
        self._latest_msg = msg
        self._msg_event.set()
