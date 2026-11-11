"""Unitree SDK2 helpers: CRC patch, DDS channel init, and low-state unpacking."""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

import numpy as np

from eden.extensions.deployment.utils.network import pick_unitree_network_interface
from eden.extensions.deployment.utils.state import RobotState

IS_UNITREE_SDK2PY_AVAILABLE = True
_UNITREE_SDK2PY_IMPORT_ERROR: Exception | None = None
try:
    from unitree_sdk2py.core.channel import (
        ChannelFactoryInitialize,
        ChannelPublisher,
        ChannelSubscriber,
    )
    from unitree_sdk2py.idl.default import (
        unitree_go_msg_dds__LowCmd_,
        unitree_hg_msg_dds__LowCmd_,
    )
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_ as LowCmd_go
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as LowState_go
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as LowCmd_hg
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as LowState_hg
    from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
    from unitree_sdk2py.utils.crc import CRC
except Exception as _exc:
    IS_UNITREE_SDK2PY_AVAILABLE = False
    _UNITREE_SDK2PY_IMPORT_ERROR = _exc
    ChannelFactoryInitialize = None
    ChannelPublisher = None
    ChannelSubscriber = None
    unitree_go_msg_dds__LowCmd_ = None
    unitree_hg_msg_dds__LowCmd_ = None
    LowCmd_go = None
    LowCmd_hg = None
    LowState_go = None
    LowState_hg = None
    MotionSwitcherClient = None
    CRC = None


__all__ = [
    "CRC",
    "ChannelPublisher",
    "ChannelSubscriber",
    "LowCmd_go",
    "LowCmd_hg",
    "LowState_go",
    "LowState_hg",
    "MotionSwitcherClient",
    "ensure_unitree_sdk",
    "init_unitree_channel",
    "read_low_state",
    "unitree_go_msg_dds__LowCmd_",
    "unitree_hg_msg_dds__LowCmd_",
    "IS_UNITREE_SDK2PY_AVAILABLE",
]


def ensure_unitree_sdk() -> None:
    if not IS_UNITREE_SDK2PY_AVAILABLE:
        cause = _UNITREE_SDK2PY_IMPORT_ERROR
        detail = f"{type(cause).__name__}: {cause}" if cause is not None else "import failed"
        raise RuntimeError(
            "Unitree SDK (unitree_sdk2py) failed to import. "
            "Install it and ensure it (and its native deps, e.g. libddsc.so.0 from CycloneDDS) are on the loader path.\n"
            f"Underlying error: {detail}"
        ) from cause


def _patch_unitree_crc_pyfallback() -> None:
    """Make ``CRC()`` survive when the SDK's native ``crc_*.so`` is missing.

    ``unitree_sdk2py`` ships a precompiled CRC library under
    ``unitree_sdk2py/utils/lib/crc_{amd64,aarch64}.so``. When the SDK is
    installed via ``pip install git+...`` that ``lib/`` directory is
    excluded from the wheel (no ``__init__.py`` → setuptools doesn't pick
    it up), so ``CRC.__init__`` crashes on first call with
    ``OSError: cannot open shared object``.

    The SDK already ships a correct pure-Python implementation
    (``CRC._crc_py``) and ``CRC.__Crc32`` routes to it whenever
    ``self.platform != "Linux"``. We override ``__init__`` to skip the
    ``ctypes.CDLL`` load and set ``self.platform`` to a sentinel value so
    the existing dispatch takes the Python path.

    Idempotent. No-op when the native lib is present. Pure-Python CRC is
    slower (~µs per byte vs ~ns) but well within budget for a 50 Hz
    control loop on the ~kB DOF-state messages.
    """
    if CRC is None or getattr(CRC, "_eden_pyfallback_patched", False):
        return

    import inspect
    import os
    import platform as _platform

    sdk_dir = os.path.dirname(inspect.getfile(CRC))
    arch = _platform.machine()
    so_name = {"x86_64": "crc_amd64.so", "aarch64": "crc_aarch64.so"}.get(arch)
    so_path = os.path.join(sdk_dir, "lib", so_name) if so_name else None
    if so_path and os.path.exists(so_path):
        CRC._eden_pyfallback_patched = True
        return  # SDK's fast path works — don't touch it.

    # Pack-format strings copied verbatim from ``unitree_sdk2py/utils/crc.py``.
    # Name-mangled because the SDK declares them as ``__packFmt*`` (double
    # underscore → ``_CRC__packFmt*`` at the instance level).
    _pack_fmts = {
        "_CRC__packFmtLowCmd": "<4B4IH2x" + "B3x5f3I" * 20 + "4B" + "55Bx2I",
        "_CRC__packFmtLowState": ("<4B4IH2x" + "13fb3x" + "B3x7fb3x3I" * 20 + "4BiH4b15H" + "8hI41B3xf2b2x2f4h2I"),
        "_CRC__packFmtHGLowCmd": "<2B2x" + "B3x5fI" * 35 + "5I",
        "_CRC__packFmtHGLowState": "<2I2B2xI" + "13fh2x" + "B3x4f2hf7I" * 35 + "40B5I",
    }

    def _patched_init(self):
        for attr, fmt in _pack_fmts.items():
            setattr(self, attr, fmt)
        # Anything ≠ "Linux" routes ``__Crc32`` through ``_crc_py``.
        self.platform = "PyFallback"
        self.crc_lib = None

    CRC.__init__ = _patched_init
    CRC._eden_pyfallback_patched = True

    msg = (
        f"Unitree SDK native CRC library not found at {so_path or '<unknown-arch>'}. "
        "Falling back to pure-Python CRC (slower but functional). "
        "Install the SDK from a wheel that bundles "
        "`unitree_sdk2py/utils/lib/*.so` to restore the fast path."
    )
    # Prefer Eden's logger so the warning matches the rest of the deploy
    # output. Fall back to the stdlib ``logging.getLogger("eden")`` if the
    # patch ran before ``en.init()`` set up ``eden.logger``.
    try:
        import eden as _en

        _en.logger.warning(msg)
    except (ImportError, AttributeError):
        import logging as _logging

        _logging.getLogger("eden").warning(msg)


_patch_unitree_crc_pyfallback()


def init_unitree_channel(interface: str | None = None) -> str:
    ensure_unitree_sdk()
    iface = interface or pick_unitree_network_interface()
    if iface is None:
        raise RuntimeError(
            "Could not auto-detect a usable network interface for Unitree DDS.\n"
            "Fix: set GS_UNITREE_NETWORK_INTERFACE to an active interface name."
        )
    ChannelFactoryInitialize(0, iface)
    return iface


def read_low_state(
    msg: Any | None,
    *,
    robot_name: str,
    dof_index: Sequence[int],
    num_full_dof: int,
) -> RobotState:
    """Unpack a Unitree SDK2 ``LowState_`` message into an Eden-order :class:`RobotState`.

    The ``unitree_go`` (Go2) and ``unitree_hg`` (G1) low-state messages expose the same
    ``motor_state`` / ``imu_state`` layout, so one unpacking path serves both.

    ``dof_index`` maps Eden DOF order to SDK motor index; ``full_dofs_pos`` stays in raw
    SDK order over ``num_full_dof`` motors, including any the policy does not drive.
    """
    if msg is None:
        raise RuntimeError(f"No {robot_name} low state received yet.")

    num_dofs = len(dof_index)
    dofs_pos = np.zeros(num_dofs, dtype=np.float64)
    dofs_vel = np.zeros(num_dofs, dtype=np.float64)
    dofs_torque = np.zeros(num_dofs, dtype=np.float64)
    full_dofs_pos = np.zeros(num_full_dof, dtype=np.float64)

    for i, motor_idx in enumerate(dof_index):
        dofs_pos[i] = msg.motor_state[motor_idx].q
        dofs_vel[i] = msg.motor_state[motor_idx].dq
        dofs_torque[i] = msg.motor_state[motor_idx].tau_est

    for i in range(num_full_dof):
        full_dofs_pos[i] = msg.motor_state[i].q

    return RobotState(
        stamp=time.time(),
        base_quat=np.array(msg.imu_state.quaternion, dtype=np.float64),  # wxyz
        base_ang_vel=np.array(msg.imu_state.gyroscope, dtype=np.float64),
        base_lin_acc=np.array(msg.imu_state.accelerometer, dtype=np.float64),
        dofs_pos=dofs_pos,
        dofs_vel=dofs_vel,
        dofs_torque=dofs_torque,
        full_dofs_pos=full_dofs_pos,
    )
