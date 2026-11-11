"""Robot plants for the SONIC teleop robot process: the body the control loop drives.

A plant is the controlled robot body behind one small interface — read this tick's
proprioception, apply one tick of joint targets, and the few rig operations only a simulator
has (the suspension band, a reset to the launch pose). Everything above this seam — the
operator link, planner, scheduler, policy, fall detector, pacing — is shared, so
``eden teleop sonic robot``, ``--mujoco`` and ``--real`` run the same control loop.

:class:`GenesisPlant` is the Genesis env itself: the 43-DOF G1 with Dex3 hands, hung from the
elastic band.

:class:`DeploymentPlant` is any :class:`~eden.extensions.deployment.base.DeploymentBase` —
the same backends, robot and gains ``eden deploy --policy sonic`` runs a motion clip on: the
29-DOF body-only G1 through the MuJoCo sim-to-sim backend or the Unitree G1 hardware backend.
The deployment keeps its own operator state machine (INIT / START / STOP / KILL), target
low-pass, joint disabling and debug recording; the hands are not driven.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from eden.envs.base import RLEnvBase
from eden.extensions.policies.sonic.band import ElasticBand, ElasticBandApplier
from eden.extensions.policies.sonic.policy import RobotStateSnapshot
from eden.options.robots.unitree_g1 import G1_DEX3_LEFT_DOFS_NAME, G1_DEX3_RIGHT_DOFS_NAME, G1_DOFS29_NAME
from eden.utils.torch import to_numpy

if TYPE_CHECKING:
    from eden.extensions.deployment.base import DeploymentBase
    from eden.extensions.deployment.mujoco import MujocoDeployment
    from eden.extensions.deployment.utils.state import RobotState

NUM_BODY_DOFS = len(G1_DOFS29_NAME)
NUM_HAND_DOFS = len(G1_DEX3_LEFT_DOFS_NAME) + len(G1_DEX3_RIGHT_DOFS_NAME)


@dataclass(frozen=True)
class PlantReading:
    """One control tick's proprioception, as float64 numpy.

    ``q_body`` / ``qd_body`` are the 29 body joints in MJCF order (the policy's);
    ``base_quat`` is the pelvis orientation (wxyz — the reference's free-joint / IMU
    quaternion, not a re-origined solver frame); ``base_ang`` the angular velocity in the
    pelvis frame (gyro convention); ``base_pos`` the pelvis position in the world, or zeros
    on a plant that cannot measure it; ``q_hands`` the 14 Dex3 joints (left then right),
    zeros on a body-only plant.
    """

    q_body: np.ndarray
    qd_body: np.ndarray
    base_quat: np.ndarray
    base_ang: np.ndarray
    base_pos: np.ndarray
    q_hands: np.ndarray

    @classmethod
    def from_robot_state(cls, state: RobotState) -> PlantReading:
        """The reading of a body-only deployment's :class:`RobotState`.

        ``base_pos`` comes from ``state.extra["base_pos"]`` when the body knows it (MuJoCo's
        ground truth) and is zeros otherwise (hardware).
        """
        base_pos = state.extra.get("base_pos")
        return cls(
            q_body=np.asarray(state.dofs_pos, dtype=np.float64),
            qd_body=np.asarray(state.dofs_vel, dtype=np.float64),
            base_quat=np.asarray(state.base_quat, dtype=np.float64),
            base_ang=np.asarray(state.base_ang_vel, dtype=np.float64),
            base_pos=np.zeros(3) if base_pos is None else np.asarray(base_pos, dtype=np.float64),
            q_hands=np.zeros(NUM_HAND_DOFS),
        )

    def snapshot(self, device: torch.device | str) -> RobotStateSnapshot:
        """The policy's per-tick state view: ``(1, ·)`` float32 tensors on ``device``."""

        def as_row(values: np.ndarray) -> torch.Tensor:
            return torch.as_tensor(values, dtype=torch.float32, device=device).unsqueeze(0)

        return RobotStateSnapshot(
            q_mj=as_row(self.q_body),
            qdot_mj=as_row(self.qd_body),
            base_ang_local=as_row(self.base_ang),
            pelvis_quat_wxyz=as_row(self.base_quat),
        )


class RobotPlant(ABC):
    """The robot body behind the SONIC control loop.

    ``step`` takes absolute joint targets keyed by the SONIC action terms:
    ``dofs_pos_controller`` ``(1, 29)``, plus ``hands_pos_controller`` ``(1, 14)`` on a plant
    with :attr:`has_hands`.
    """

    #: Control period in seconds.
    control_dt: float
    #: The suspension the robot hangs from until the operator releases it (``None``: none).
    band: ElasticBand | GantryBand | None = None
    #: Whether the body has Dex3 hands to read and drive.
    has_hands: bool = False
    #: Whether this body is a real robot: no :meth:`reset`, and hardware-only safety checks apply.
    is_hardware: bool = False

    @property
    def policy_active(self) -> bool:
        """Whether the targets passed to :meth:`step` reach the robot this tick (a deployment's START gate)."""
        return True

    def connect(self) -> None:
        """Open the body. Default: nothing to open."""

    @abstractmethod
    def reset(self) -> None:
        """Bring the body to its launch state (a simulator places the robot; hardware arms its operator gate)."""

    @abstractmethod
    def read(self) -> PlantReading:
        """This tick's proprioception, read fresh from the body."""

    @abstractmethod
    def step(self, action: dict[str, torch.Tensor]) -> None:
        """Apply one tick of joint targets and advance the body by one control period."""

    def consume_start_events(self) -> int:
        """Operator START presses since the last call (deployments only)."""
        return 0

    def render(self) -> None:
        """Refresh a viewer that :meth:`step` does not drive itself. Default: nothing."""

    def close(self) -> None:
        """Release the body. Default: nothing."""


# ---------------------------------------------------------------------------
# Genesis
# ---------------------------------------------------------------------------


class SonicTeleopEnv(RLEnvBase):
    """RLEnvBase that re-applies the elastic band before every scene step.

    Genesis clears applied external forces after each ``scene.step()``, and
    one control tick runs ``decimation`` scene steps — applying the band
    once per tick would leave it active for only the first substep.
    """

    band_applier: ElasticBandApplier | None = None

    def _scene_step_with_sensor_gating(self, is_last_substep: bool) -> None:
        if self.band_applier is not None:
            self.band_applier.apply()
        super()._scene_step_with_sensor_gating(is_last_substep)


class GenesisPlant(RobotPlant):
    """The Genesis env as the body: the 43-DOF G1 with Dex3 hands, hung from the elastic band."""

    has_hands = True

    def __init__(self, env: SonicTeleopEnv, entity_name: str = "robot") -> None:
        self._env = env
        self.control_dt = float(env.dt)
        robot = env.entities[entity_name]
        self._robot = robot
        _, body_idx = robot.find_named_dofs_idx_local(G1_DOFS29_NAME, preserve_order=True)
        _, hands_idx = robot.find_named_dofs_idx_local(
            G1_DEX3_LEFT_DOFS_NAME + G1_DEX3_RIGHT_DOFS_NAME, preserve_order=True
        )
        # One combined position read per tick (body + hands): each get_* is a GPU sync.
        self._all_idx = torch.as_tensor(list(body_idx) + list(hands_idx), dtype=torch.long, device=env.device)
        self._body_idx = torch.as_tensor(list(body_idx), dtype=torch.long, device=env.device)
        applier = ElasticBandApplier(env, entity_name=entity_name)
        env.band_applier = applier
        self.band = applier.band

    def reset(self) -> None:
        self._env.reset()

    def read(self) -> PlantReading:
        # relative=True reads the MJCF pelvis body frame (the reference sim's ground-truth
        # IMU / qpos frame), not the re-origined solver frame.
        robot = self._robot
        q_all = to_numpy(robot.get_dofs_pos(dofs_idx_local=self._all_idx)[0], np.float64)
        return PlantReading(
            q_body=q_all[:NUM_BODY_DOFS],
            qd_body=to_numpy(robot.get_dofs_vel(dofs_idx_local=self._body_idx)[0], np.float64),
            base_quat=to_numpy(robot.get_quat(relative=True)[0], np.float64),
            base_ang=to_numpy(robot.get_ang(frame="body")[0], np.float64),
            base_pos=to_numpy(robot.get_pos(relative=True)[0], np.float64),
            q_hands=q_all[NUM_BODY_DOFS:],
        )

    def step(self, action: dict[str, torch.Tensor]) -> None:
        self._env.step(action)  # the env subclass applies the band per scene step


# ---------------------------------------------------------------------------
# Deployment-backed (MuJoCo sim-to-sim, hardware)
# ---------------------------------------------------------------------------


class GantryBand:
    """The MuJoCo deployment's bring-up gantry behind the band interface (``9`` rigs / releases it).

    ``enable`` is a plain flag any thread may flip (the keyboard listener does);
    :meth:`apply` rigs or releases the rope on the control thread, between physics steps.
    The anchor height stays on the deployment's own ``+`` / ``-`` operator keys.
    """

    key_help = "9 release/rig gantry"

    def __init__(self, dep: MujocoDeployment) -> None:
        self._dep = dep
        self.enable = bool(dep.gantry)

    def handle_key(self, key: str) -> bool:
        if key != "9":
            return False
        self.enable = not self.enable
        print(f"[sonic robot] gantry {'rigged' if self.enable else 'released'}")
        return True

    def apply(self) -> None:
        """Bring the simulated gantry in line with ``enable``."""
        if self.enable != bool(self._dep.gantry):
            self._dep.rig_gantry(self.enable)


class DeploymentPlant(RobotPlant):
    """A :class:`DeploymentBase` driving the body-only G1 as the body — MuJoCo or hardware.

    ``read`` calls ``read_state`` fresh at the top of the tick, so on hardware the policy
    does not act on a state one control period old; ``step`` is the deployment's own
    ``step``, which runs its operator state machine (the action only reaches the robot
    under START — see :attr:`policy_active`), sends the payload and records it. The
    deployment must be constructed with ``sync=False`` (the SONIC loop paces itself) and
    ``auto_yaw_align=False`` (the policy captures its own heading on engage).

    ``is_hardware`` marks a real robot (the hardware-only safety checks apply). ``mirror``
    is the Genesis env and its original robot entity — the deployment swapped the env's for
    a state wrapper — given when the Genesis viewer should show the live robot.
    """

    def __init__(
        self,
        dep: DeploymentBase,
        *,
        is_hardware: bool,
        band: GantryBand | None = None,
        mirror: tuple[RLEnvBase, Any] | None = None,
    ) -> None:
        if tuple(dep.dofs_name) != tuple(G1_DOFS29_NAME):
            raise ValueError(
                f"The SONIC deployment plant drives the 29-DOF G1 body in MJCF order; the deployment drives "
                f"{len(dep.dofs_name)} DOF(s) starting {list(dep.dofs_name)[:3]}."
            )
        if dep.sync:
            raise ValueError("The SONIC loop paces itself; construct the deployment with sync=False.")
        if dep.auto_yaw_align:
            raise ValueError(
                "The SONIC policy captures its own heading; construct the deployment with auto_yaw_align=False."
            )
        self._dep = dep
        self._mirror = mirror
        self.band = band
        self.control_dt = float(dep.control_dt)
        self.is_hardware = is_hardware

    @property
    def policy_active(self) -> bool:
        return self._dep.policy_active

    def connect(self) -> None:
        self._dep.connect()

    def reset(self) -> None:
        if self.band is not None:
            self.band.apply()  # the deployment re-rigs its gantry from this on reset
        self._dep.reset()

    def read(self) -> PlantReading:
        return PlantReading.from_robot_state(self._dep.read_state())

    def step(self, action: dict[str, torch.Tensor]) -> None:
        if self.band is not None:
            self.band.apply()
        self._dep.step(action)
        for request in self._dep.consume_policy_switches():
            print(f"[sonic robot] policy switch request {request!r} ignored (single policy)")

    def consume_start_events(self) -> int:
        return self._dep.consume_start_events()

    def render(self) -> None:
        if self._mirror is None:
            return
        env, sim_robot = self._mirror
        state = self._dep.state
        sim_robot.set_quat(state.base_quat)
        sim_robot.set_dofs_pos(state.dofs_pos)
        env.scene.visualizer.update()

    def close(self) -> None:
        self._dep.close()
