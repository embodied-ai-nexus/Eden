"""Unitree Go2 real-robot deployment backend."""

from __future__ import annotations
from typing import TYPE_CHECKING, Any

import numpy as np

import eden as en
from eden.extensions.deployment.base import (
    DEPLOYMENT_REGISTRY,
    bind_deployment,
    resolve_dof_index_map,
)
from eden.extensions.deployment.unitree_dds import _UnitreeDdsDeployment
from eden.options.robots.unitree_go2 import Go2
from eden.extensions.deployment.utils.unitree_sdk2 import (
    LowCmd_go,
    LowState_go,
    unitree_go_msg_dds__LowCmd_,
)

if TYPE_CHECKING:
    from collections.abc import Callable


#: ``LowCmd_.head``: the two header bytes Unitree's Go2 examples stamp on every low-level command.
_GO2_LOW_CMD_HEAD = (0xFE, 0xEF)

# Go2 DDS motor index order: leg-major FR, FL, RR, RL, each [hip, thigh, calf].
# Deliberately independent of ``GO1_JOINT_IDX``, which currently holds the same twelve
# entries -- see the note there.
GO2_JOINT_IDX: dict[str, int] = {
    "FR_hip_joint": 0,
    "FR_thigh_joint": 1,
    "FR_calf_joint": 2,
    "FL_hip_joint": 3,
    "FL_thigh_joint": 4,
    "FL_calf_joint": 5,
    "RR_hip_joint": 6,
    "RR_thigh_joint": 7,
    "RR_calf_joint": 8,
    "RL_hip_joint": 9,
    "RL_thigh_joint": 10,
    "RL_calf_joint": 11,
}


@DEPLOYMENT_REGISTRY.register()
class UnitreeGo2Deployment(_UnitreeDdsDeployment):
    """Drive a Unitree Go2 over the ``unitree_go`` DDS low-level interface.

    Configurable options (in addition to :class:`~eden.extensions.deployment.unitree_dds._UnitreeDdsDeployment`'s):

    publish_rate: float | None
        Rate at which the held ``LowCmd_`` is re-sent between control ticks (default 500 Hz).
        ``None`` sends one packet per control tick.

        This is vendor-rate parity, not a fault fix. The Go2's twelve joints are all serial --
        its ``LowCmd_`` has no ``mode_pr``, as there is no parallel linkage to transform -- so
        the per-packet parallel-linkage fault the republisher exists for on the Oli and G1
        does not apply here. 500 Hz is what Unitree's own Go2 low-level example feeds the robot
        at (``go2_stand_example.py``: ``dt = 0.002`` and a 2 ms ``RecurrentThread`` writer),
        independent of any policy rate; one packet per 50 Hz tick is outside that example's own
        ``0.001~0.01`` s range. ``--publish_rate 0`` sends one packet per tick.
    """

    # Unitree's own Go2 low-level rate; see the class docstring for why it is not the control rate.
    publish_rate: float | None = 500.0

    # The republisher's failure messages default to the parallel-joint fault, which the Go2 cannot have.
    _PUB_DROPOUT_EFFECT = (
        "the same as --publish_rate 0, below Unitree's own 500 Hz low-level rate. No Go2 fault is known to "
        "follow, so stopping the run is your call"
    )
    _PUB_SLOW_EFFECT = "the robot was fed below Unitree's own low-level rate"

    _ROBOT_NAME = "Unitree Go2"
    _SOFT_SHUTDOWN_LABEL = "Go2 soft shutdown"

    def _resolve_options(self) -> None:
        """Resolve each deployment DOF's SDK motor index (see :meth:`DeploymentBase._resolve_options`).

        Go2W inherits this binding through the MRO but has no wheel-motor mapping, so its
        unmapped joints mean an unbuilt backend rather than a typo — hence NotImplementedError.
        """
        super()._resolve_options()
        self._dof_index = resolve_dof_index_map(
            self.dofs_name,
            GO2_JOINT_IDX,
            owner=type(self).__name__,
            error=NotImplementedError,
            hint="There is no wheeled Go2W deployment backend: its wheel motors have no confirmed SDK motor index.",
        )
        self._num_full_dof = len(GO2_JOINT_IDX)

    def _dds_messages(self) -> tuple[type, type, Callable[[], Any]]:
        return LowState_go, LowCmd_go, unitree_go_msg_dds__LowCmd_

    def _init_low_cmd(self) -> None:
        try:
            self._low_cmd.head[0] = _GO2_LOW_CMD_HEAD[0]
            self._low_cmd.head[1] = _GO2_LOW_CMD_HEAD[1]
        except (AttributeError, IndexError, TypeError):
            en.logger.warning("Could not set low_cmd header bytes; SDK message may lack a 'head' field.")
        super()._init_low_cmd()

    def _shutdown_hold_pos(self) -> np.ndarray | None:
        """Hold where the robot is now (see :meth:`SoftShutdownMixin._shutdown_hold_pos`); nothing once disconnected."""
        if not self._connected:
            return None
        return self.read_state().dofs_pos


bind_deployment(Go2, "unitree_go2_deployment")
# No wheeled Go2W backend: its wheel motors (FL/RL/FR/RR_foot_joint) have no
# confirmed SDK motor index or velocity-mode mapping, and guessing them for real hardware
# is unsafe. Go2W resolves here through the MRO, so __init__ raises for its unmapped joints.
