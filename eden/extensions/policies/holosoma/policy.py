"""Holosoma velocity-command locomotion policy port (Unitree G1 29-DOF).

Wraps the released Holosoma ONNX locomotion policy (``ppo_g1_29dof.onnx`` /
``fastsac_g1_29dof.onnx``) as an Eden :class:`~eden.extensions.policies.base.
PolicyBase` so it can be driven inside a Genesis environment. This reproduces
the deployment-time observation construction, gait-phase clock, and action
transform from ``holosoma_inference.policies.locomotion.LocomotionPolicy`` in
batched torch.

The policy is command-driven: each step it consumes a 3-D velocity command
``[vel_x, vel_y, ang_vel_z]`` (set via :meth:`set_command`, defaulting to zero
→ standing) plus proprioception read directly from the simulator, and emits 29
joint actions (legs + waist + arms — the loco policy controls all DOFs).

I/O contract
------------
Observation (``actor_obs``, 100 dim, history length 1). Holosoma flattens the
group's terms in **sorted (alphabetical)** order, so the layout is::

    actions(29)            previous raw policy output            × 1.0
    base_ang_vel(3)        body-frame gyro                       × 0.25
    command_ang_vel(1)     vel_z command                        × 1.0
    command_lin_vel(2)     [vel_x, vel_y] command               × 1.0
    cos_phase(2)           cos(gait phase) [left, right]        × 1.0
    dof_pos(29)            q − default_dof_pos                   × 1.0
    dof_vel(29)            q̇                                    × 0.05
    projected_gravity(3)   R(base)ᵀ·[0,0,-1]                     × 1.0
    sin_phase(2)           sin(gait phase) [left, right]        × 1.0

Gait phase: a two-element clock (left foot starts at 0, right at π) advancing
``2π / (rl_rate · gait_period)`` per step. When the commanded velocity is ~0
both feet are pinned to π (standing); the clock re-initializes to ``[0, π]`` on
the first moving step after standing.

Action transform::

    target = default_dof_pos + policy_action_scale · action      (all 29 DOFs)

The 29-D absolute target is handed to Genesis's built-in PD via an
:class:`~eden.managers.terms.actions.joint_actions.ImplicitPDController` with
``reference_source=ZERO`` and ``scale=1.0``. PD gains are ``G1Dofs29``'s defaults,
bit-for-bit the released ONNX ``kp`` / ``kd`` metadata (see
:mod:`~eden.extensions.policies.holosoma.constants`).

Checkpoints
-----------
``checkpoint_path`` defaults to the hub path ``holosoma/ppo_g1_29dof.onnx`` on
``Kashu7100/eden_policies`` (md5-pinned, auto-download on first use). Point it at
local weights (which also clears the pinned hash) before instantiating::

    HolosomaLocomotionPolicy.use_local_checkpoint("/path/to/ppo_g1_29dof.onnx")
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from genesis.utils.geom import inv_transform_by_quat

from eden import actions
from eden.constants import ReferenceSource
from eden.extensions.policies.base import POLICY_REGISTRY, PolicyBase, PolicyOptions
from eden.extensions.policies.holosoma.constants import (
    G1_LOCO_DEFAULT_DOF_POS,
    NUM_DOFS,
    OBS_SCALE_BASE_ANG_VEL,
    OBS_SCALE_DOF_VEL,
)
from eden.extensions.policies.onnx import OnnxModule
from eden.options.managers.observations import ObservationGroupOptions
from eden.options.robots.unitree_g1 import G1_DOFS29_NAME, G1Dofs29

if TYPE_CHECKING:
    from eden.envs.base import EnvBase
    from eden.options.entities import RobotOptions
    from eden.options.managers.actions import ActionTermOptions


_OBS_DIM: int = 100
# Standing detection threshold on the command norm (matches LocomotionPolicy:
# ``np.linalg.norm(...) < 0.01`` on both lin- and ang-vel commands).
_STAND_EPS: float = 0.01


def holosoma_g1_robot(**overrides: Any) -> RobotOptions:
    """Build a ``G1Dofs29`` configured to match Holosoma's loco training setup.

    Overrides only the default joint pose (Holosoma's deeper-crouch stance). The
    PD gains are left to ``G1Dofs29``'s Raibert heuristic, which is bit-for-bit
    identical to the released ONNX ``kp`` / ``kd`` metadata, so re-specifying
    them would be redundant.

    Parameters
    ----------
    **overrides : Any
        Extra ``G1Dofs29`` fields to override (e.g. ``default_root_pos``).

    Returns
    -------
    RobotOptions
        A configured ``G1Dofs29`` instance.
    """
    default_dofs_pos = {name: G1_LOCO_DEFAULT_DOF_POS[i] for i, name in enumerate(G1_DOFS29_NAME)}
    return G1Dofs29(default_dofs_pos=default_dofs_pos, **overrides)


@POLICY_REGISTRY.register()
class HolosomaLocomotionPolicy(PolicyBase):
    """Holosoma velocity-command locomotion controller for G1 29-DOF (ONNX backend).

    Reads proprioception directly from the env each step and drives the frozen
    ONNX policy with the current 3-D velocity command (see :meth:`set_command`).
    Only the Unitree G1 29-DOF robot is supported; ``G1Dofs29.dofs_name`` order
    matches Holosoma's joint order, so no DOF remapping is needed.
    """

    entity_name: str = "robot"

    # Default to the PPO loco checkpoint mirrored on Kashu7100/eden_policies
    # (md5-pinned). The fastsac variant ships identical gains and I/O — point
    # ``checkpoint_path`` at it (and clear ``md5_hash``) to use it instead.
    checkpoint_path: str | None = "holosoma/ppo_g1_29dof.onnx"
    md5_hash: str | None = "cfe906f8c5a87b5c4b367b1241e8b217"

    # Control cadence / transforms (mirror task.locomotion defaults).
    policy_action_scale: float = 0.25
    rl_rate: float = 50.0
    gait_period: float = 1.0

    def __init__(self, env: EnvBase, options: PolicyOptions) -> None:
        """Load the ONNX graph, cache DOF indices, and allocate command/phase buffers."""
        super().__init__(env, options)

        device = env.device
        self.device = device
        self._dtype = torch.float32
        N = env.num_envs

        # Single-IO ONNX graph (actor_obs -> action). OnnxModule validates the
        # static input/output dims against these (no-op for a dynamic axis),
        # raising on a swapped / re-exported graph.
        self._policy = OnnxModule(
            self.checkpoint_path, "actor_obs", "action", device=str(device),
            label="Holosoma Loco ONNX", expected_input_dim=_OBS_DIM, expected_output_dim=NUM_DOFS,
        )  # fmt: skip

        robot = self.require_entity(self.entity_name)
        # All 29 DOFs in Holosoma / G1Dofs29 declaration order.
        _, dofs_idx = robot.find_named_dofs_idx_local(G1_DOFS29_NAME, preserve_order=True)
        self._dofs_idx = torch.as_tensor(dofs_idx, dtype=torch.long, device=device)

        self._default_dof_pos = torch.tensor(G1_LOCO_DEFAULT_DOF_POS, dtype=self._dtype, device=device)

        # Current 3-D command [vel_x, vel_y, ang_vel_z]; zero → standing.
        self._commands = torch.zeros(N, 3, device=device, dtype=self._dtype)
        # Previous raw policy output, fed back as the `actions` observation.
        self._last_action = torch.zeros(N, NUM_DOFS, device=device, dtype=self._dtype)

        # Gait phase clock (left foot at 0, right at π) and per-env standing flag.
        self._phase = torch.zeros(N, 2, device=device, dtype=self._dtype)
        self._phase[:, 1] = torch.pi
        self._is_standing = torch.zeros(N, dtype=torch.bool, device=device)
        self._phase_dt = 2.0 * torch.pi / (self.rl_rate * self.gait_period)

        # World-frame "down" vector, expanded once to the env batch.
        self._gravity_world = torch.tensor([0.0, 0.0, -1.0], device=device, dtype=self._dtype).expand(N, 3)

    # ------------------------------------------------------------------
    # Commands
    # ------------------------------------------------------------------

    def set_command(self, command: torch.Tensor) -> None:
        """Set the 3-D velocity command for all (or a slice of) environments.

        Parameters
        ----------
        command : torch.Tensor
            Either ``(3,)`` (broadcast to all envs) or ``(num_envs, 3)``. Layout
            is ``[vel_x, vel_y, ang_vel_z]``. A ~zero command makes the policy
            stand (gait phase pins to π).
        """
        command = torch.as_tensor(command, dtype=self._dtype, device=self.device)
        if command.ndim == 1:
            command = command.unsqueeze(0).expand(self._commands.shape[0], -1)
        self._commands[:] = command

    # ------------------------------------------------------------------
    # Env / action / observation specs
    # ------------------------------------------------------------------

    @classmethod
    def robot_options(cls) -> RobotOptions:
        """Return a ``G1Dofs29`` with Holosoma's training default joint pose."""
        return holosoma_g1_robot()

    @classmethod
    def env_options(cls) -> dict[str, Any]:
        """Return env options matching Holosoma's training sim: 200 Hz physics,
        decimation 4 → 50 Hz control (== ``rl_rate``). Holosoma trains all
        backends at ``fps=200, control_decimation=4, substeps=1``
        (``holosoma.config_values.simulator``), so reproduce that exact cadence
        rather than another (sim_dt, decimation) pair with the same 50 Hz product.
        """
        return dict(
            sim_dt=0.005,
            sim_substeps=1,
            decimation=4,
        )

    @classmethod
    def action_options(cls, entity_name: str = "robot") -> dict[str, ActionTermOptions]:
        """Configure a 29-DOF implicit PD controller consuming absolute targets.

        :meth:`step` already produces absolute joint-position targets
        (``default + scale * action``), so the controller uses
        ``reference_source=ZERO`` and ``scale=1.0`` (no extra offset / scaling).
        """
        return dict(
            dofs_pos_controller=actions.ImplicitPDController.configure(
                entity_name=entity_name,
                dofs_name=G1_DOFS29_NAME,
                scale=1.0,
                reference_source=ReferenceSource.ZERO,
            ),
        )

    @classmethod
    def observation_options(cls, entity_name: str = "robot") -> dict[str, ObservationGroupOptions]:
        """No env-computed observations.

        The policy builds its own 100-dim ``actor_obs`` from raw simulator state
        in :meth:`step` (the ``obs`` argument is ignored), so the env needs no
        observation terms — an empty group set builds and steps fine.
        """
        return {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self, envs_idx: (slice | torch.Tensor) | None = None) -> None:
        """Reset prev-action and gait-phase state for the given environments.

        Parameters
        ----------
        envs_idx : slice | torch.Tensor | None, optional
            Environments to reset; ``None`` resets all.
        """
        if envs_idx is None:
            self._last_action.zero_()
            self._phase[:, 0] = 0.0
            self._phase[:, 1] = torch.pi
            self._is_standing.zero_()
        else:
            self._last_action[envs_idx] = 0.0
            self._phase[envs_idx, 0] = 0.0
            self._phase[envs_idx, 1] = torch.pi
            self._is_standing[envs_idx] = False

    # ------------------------------------------------------------------
    # Gait phase
    # ------------------------------------------------------------------

    def _update_phase(self) -> None:
        """Advance the gait clock, pinning to a standing phase under ~zero command.

        Vectorized port of ``LocomotionPolicy.update_phase_time``.
        """
        self._phase = torch.remainder(self._phase + self._phase_dt + torch.pi, 2.0 * torch.pi) - torch.pi

        lin_norm = torch.linalg.norm(self._commands[:, :2], dim=-1)
        ang_norm = self._commands[:, 2].abs()
        standing = (lin_norm < _STAND_EPS) & (ang_norm < _STAND_EPS)

        # Standing: pin both feet to π.
        self._phase[standing] = torch.pi
        # First moving step after standing: re-init clock to [0, π].
        starting = (~standing) & self._is_standing
        if starting.any():
            self._phase[starting, 0] = 0.0
            self._phase[starting, 1] = torch.pi
        self._is_standing = standing

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _build_obs(self, robot) -> torch.Tensor:
        """Assemble the 100-dim ``actor_obs`` vector (sorted-term layout)."""
        dof_pos = robot.get_dofs_pos(dofs_idx_local=self._dofs_idx)
        dof_vel = robot.get_dofs_vel(dofs_idx_local=self._dofs_idx)
        base_ang_vel = robot.get_ang(frame="body")
        base_quat = robot.get_quat()
        projected_gravity = inv_transform_by_quat(self._gravity_world, base_quat)

        sin_phase = torch.sin(self._phase)
        cos_phase = torch.cos(self._phase)

        # Concatenate in Holosoma's sorted-term order.
        return torch.cat(
            [
                self._last_action,  # actions (29)
                base_ang_vel * OBS_SCALE_BASE_ANG_VEL,  # base_ang_vel (3)
                self._commands[:, 2:3],  # command_ang_vel (1)
                self._commands[:, :2],  # command_lin_vel (2)
                cos_phase,  # cos_phase (2)
                dof_pos - self._default_dof_pos,  # dof_pos (29)
                dof_vel * OBS_SCALE_DOF_VEL,  # dof_vel (29)
                projected_gravity,  # projected_gravity (3)
                sin_phase,  # sin_phase (2)
            ],
            dim=-1,
        )

    def step(self, obs: dict[str, torch.Tensor | dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        """Advance the gait clock, run the policy, and return 29-DOF targets.

        Parameters
        ----------
        obs : dict
            Ignored — raw simulator state is read directly from the env.

        Returns
        -------
        dict[str, torch.Tensor]
            ``{"dofs_pos_controller": (N, 29)}`` absolute joint-position targets.
        """
        # Read off the entity, not an observation group: this policy rides along as an extra deploy
        # slot on another policy's or task's env, whose groups are not its own (see the
        # "Declare the observation" paragraph of docs/guides/policy-porting.md).
        del obs
        robot = self.env.entities[self.entity_name]

        # Phase advances once per control step, before observation construction
        # (matches LocomotionPolicy.run: update_phase_time() then policy_action()).
        self._update_phase()

        actor_obs = self._build_obs(robot)
        action = self._policy(actor_obs)  # raw policy output, (N, 29)
        self._last_action = action

        targets = self._default_dof_pos + self.policy_action_scale * action
        return {"dofs_pos_controller": targets}
