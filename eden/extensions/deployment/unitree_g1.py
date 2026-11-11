"""Unitree G1 real-robot deployment backend."""

from __future__ import annotations
from typing import Any, ClassVar, TYPE_CHECKING

import struct
import time

import numpy as np

import eden as en
from eden.extensions.deployment.base import (
    DEPLOYMENT_REGISTRY,
    bind_deployment,
    resolve_dof_index_map,
)
from eden.extensions.deployment.unitree_dds import _UnitreeDdsDeployment
from eden.options.robots.unitree_g1 import G1Dofs23, G1Dofs29, G1Dofs29NoHands
from eden.extensions.deployment.utils.operator_input import (
    CMD_INIT,
    CMD_KILL,
    CMD_NEXT_POLICY,
    CMD_START,
    CMD_STOP,
)
from eden.extensions.deployment.utils.unitree_sdk2 import (
    LowCmd_hg,
    LowState_hg,
    MotionSwitcherClient,
    unitree_hg_msg_dds__LowCmd_,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from eden.envs.base import EnvBase
    from eden.options.extensions.deployment import DeploymentOptions


G1_JOINT_IDX: dict[str, int] = {
    "left_hip_pitch_joint": 0,
    "left_hip_roll_joint": 1,
    "left_hip_yaw_joint": 2,
    "left_knee_joint": 3,
    "left_ankle_pitch_joint": 4,
    "left_ankle_roll_joint": 5,
    "right_hip_pitch_joint": 6,
    "right_hip_roll_joint": 7,
    "right_hip_yaw_joint": 8,
    "right_knee_joint": 9,
    "right_ankle_pitch_joint": 10,
    "right_ankle_roll_joint": 11,
    "waist_yaw_joint": 12,
    "waist_roll_joint": 13,
    "waist_pitch_joint": 14,
    "left_shoulder_pitch_joint": 15,
    "left_shoulder_roll_joint": 16,
    "left_shoulder_yaw_joint": 17,
    "left_elbow_joint": 18,
    "left_wrist_roll_joint": 19,
    "left_wrist_pitch_joint": 20,
    "left_wrist_yaw_joint": 21,
    "right_shoulder_pitch_joint": 22,
    "right_shoulder_roll_joint": 23,
    "right_shoulder_yaw_joint": 24,
    "right_elbow_joint": 25,
    "right_wrist_roll_joint": 26,
    "right_wrist_pitch_joint": 27,
    "right_wrist_yaw_joint": 28,
}


# Bit indices of the G1 wireless-remote ``keys`` uint16 (Unitree layout, shared
# with unitree_rl_gym). The 40-byte ``wireless_remote`` buffer carried inside
# every ``LowState_hg`` packs ``keys`` as a little-endian uint16 at offset 2.
_REMOTE_KEY_BITS: dict[str, int] = {
    "R1": 0, "L1": 1, "start": 2, "select": 3, "R2": 4, "L2": 5, "F1": 6, "F2": 7,
    "A": 8, "B": 9, "X": 10, "Y": 11, "up": 12, "right": 13, "down": 14, "left": 15,
}  # fmt: skip


class _RemoteInput:
    """Reads the G1 wireless remote from the deployment's latest low-state msg.

    No extra DDS channel is needed: the 40-byte ``wireless_remote`` buffer is
    already carried inside every ``LowState_hg`` packet the deployment
    subscribes to. Commands fire on the rising edge of a button press so a
    held button maps to a single intent.
    """

    def __init__(self, deployment: UnitreeG1Deployment) -> None:
        self._dep = deployment
        self._prev_keys = 0

    def start(self) -> bool:
        return True

    def poll(self) -> list[str]:
        msg = self._dep._latest_msg
        remote = getattr(msg, "wireless_remote", None) if msg is not None else None
        if remote is None:
            return []
        data = bytes(bytearray(remote))
        if len(data) < 4:
            return []
        keys = struct.unpack_from("<H", data, 2)[0]
        rising = keys & ~self._prev_keys
        self._prev_keys = keys

        def bit(name: str) -> int:
            return 1 << _REMOTE_KEY_BITS[name]

        cmds: list[str] = []
        if rising & bit("A"):
            cmds.append(CMD_START)
        if rising & bit("B"):
            cmds.append(CMD_STOP)
        if rising & bit("Y"):
            cmds.append(CMD_INIT)
        # Select cycles to the next loaded policy (matches Holosoma's NEXT_POLICY).
        # Ignore the L1+select combo so it doesn't collide with other chords.
        if (rising & bit("select")) and not (keys & bit("L1")):
            cmds.append(CMD_NEXT_POLICY)
        # KILL when L1 and R1 are held together (matches Holosoma's L1+R1).
        if (keys & bit("L1")) and (keys & bit("R1")):
            cmds.append(CMD_KILL)
        return cmds


@DEPLOYMENT_REGISTRY.register()
class UnitreeG1Deployment(_UnitreeDdsDeployment):
    """Drive a Unitree G1 over the ``unitree_hg`` DDS low-level interface.

    Configurable options (in addition to :class:`~eden.extensions.deployment.unitree_dds._UnitreeDdsDeployment`'s):

    publish_rate: float | None
        Rate at which the held ``LowCmd_`` is re-sent between control ticks (default
        500 Hz, Unitree's own low-level rate). ``None`` sends one packet per control tick.

        The G1's ankle *and* -- on the 3-DOF-waist 29-DOF build -- waist roll/pitch are
        parallel A/B linkages, driven here in PR (pitch/roll) mode via ``mode_pr``. That is
        the same joint class whose per-packet PD made the LimX Oli's waist buzz at one
        packet per 50 Hz tick: the derivative term acts on a 20 ms-old velocity,
        which at 25 Hz is anti-damping. 500 Hz is what both vendor references use --
        Unitree's own ``g1_low_level_example`` publishes at ``control_dt_ = 0.002``, and
        NVIDIA's SONIC G1 deploy runs a dedicated 500 Hz ``LowCommandWriter`` thread beside
        its 50 Hz control thread -- so it is the rate the robot is expected to be fed at,
        independent of how fast any policy runs.

        Unlike the Oli case this has **not** been measured on a G1: the mechanism and the
        rate are taken from those two references, not from a hardware fit. ``--publish_rate 0``
        sends one packet per tick, which is the A/B to run before concluding anything.
    """

    # Unitree's own low-level rate; see the class docstring for why it is not the control rate.
    publish_rate: float | None = 500.0

    _ROBOT_NAME = "Unitree G1"
    #: ``LowCmd_.mode_pr``: the ankle and waist parallel linkages driven in PR (pitch/roll) mode.
    _MODE_PR: ClassVar[int] = 0

    def __init__(self, env: EnvBase, options: DeploymentOptions) -> None:
        super().__init__(env, options)

        self._mode_machine = None

    def _resolve_options(self) -> None:
        """Resolve each deployment DOF's SDK motor index (see :meth:`DeploymentBase._resolve_options`)."""
        super()._resolve_options()
        self._dof_index = resolve_dof_index_map(self.dofs_name, G1_JOINT_IDX, owner=type(self).__name__)
        self._num_full_dof = max(G1_JOINT_IDX.values()) + 1

    def _dds_messages(self) -> tuple[type, type, Callable[[], Any]]:
        return LowState_hg, LowCmd_hg, unitree_hg_msg_dds__LowCmd_

    def connect(self) -> None:
        # Release any running high-level mode first, and re-learn ``mode_machine`` from the new stream.
        self._mode_machine = None
        self._release_motion_mode()
        super().connect()

    # ------------------------------------------------------------------
    # Operator remote and shutdown
    # ------------------------------------------------------------------

    def _remote_inputs(self) -> list[Any]:
        return [_RemoteInput(self)]

    def _shutdown_hold_pos(self) -> np.ndarray | None:
        """Hold the last pose the loop read (see :meth:`SoftShutdownMixin._shutdown_hold_pos`); nothing before it."""
        if not self._connected or self.state is None:
            return None
        return self.state.dofs_pos.copy()

    def _print_control_guide(self) -> None:
        en.logger.info("=" * 60)
        en.logger.info("G1 DEPLOY CONTROLS  (remote / keyboard)")
        en.logger.info("  A  / s+Enter    START  (hand control to the policy)")
        en.logger.info("  B  / x+Enter    STOP   (hold current pose with PD)")
        en.logger.info("  Y  / i+Enter    INIT   (interpolate to default pose)")
        en.logger.info("  L1+R1 / k+Enter KILL   (soft-damp and exit)")
        en.logger.info("  select / n+Enter SWITCH to next policy (if >1 loaded)")
        en.logger.info("  1-9 + Enter     SWITCH to that policy slot")
        en.logger.info("Robot boots in STOP/hold. Press INIT, then START.")
        en.logger.info("=" * 60)

    # ------------------------------------------------------------------
    # The G1's own packet fields (``mode_pr`` / ``mode_machine``) and motion-mode release
    # ------------------------------------------------------------------

    def _prepare_low_cmd(self) -> None:
        self._low_cmd.mode_pr = self._MODE_PR
        if self._mode_machine is not None:
            self._low_cmd.mode_machine = int(self._mode_machine)

    def _init_low_cmd(self) -> None:
        super()._init_low_cmd()
        self._low_cmd.mode_pr = self._MODE_PR
        self._low_cmd.mode_machine = 0

    def _on_low_state(self, msg: Any) -> None:
        # The robot's ``mode_machine``, latched before the base fires the event, so ``connect``
        # returns with it known.
        if self._mode_machine is None and hasattr(msg, "mode_machine"):
            self._mode_machine = msg.mode_machine
        super()._on_low_state(msg)

    def _release_motion_mode(self) -> None:
        # NOTE: release any currently running high-level mode before low-level control.
        if MotionSwitcherClient is None:
            return
        try:
            msc = MotionSwitcherClient()
            msc.SetTimeout(self.connect_timeout_s)
            msc.Init()
            _, result = msc.CheckMode()
            while result and result.get("name"):
                msc.ReleaseMode()
                time.sleep(0.5)
                _, result = msc.CheckMode()
        except Exception:
            return


# All G1 robot variants are driven by the same deployment backend; ``eden
# deploy`` looks these up via ``resolve_deployment_for(robot_options)``.
bind_deployment(G1Dofs23, "unitree_g1_deployment")
bind_deployment(G1Dofs29, "unitree_g1_deployment")
bind_deployment(G1Dofs29NoHands, "unitree_g1_deployment")
