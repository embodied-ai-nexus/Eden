"""ActionTerm driving a Genesis RigidEntity through Hill-type muscle dynamics."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import torch

from eden.extensions.musculoskeletal.hill_muscle import (
    MuscleParams,
    muscle_actuator_force,
    step_muscle_activation,
)
from eden.managers.action_manager import ACTION_TERM_REGISTRY, ActionTerm
from eden.utils.string import term_subject

if TYPE_CHECKING:
    from eden.envs.base import EnvBase
    from eden.options.managers.actions import ActionTermOptions

GeometryFn = Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor]]


@ACTION_TERM_REGISTRY.register()
class HillMuscleController(ActionTerm):
    """Drive controlled DOFs via Millard muscle activation dynamics and Hill FLV curves.

    Accepts ``num_muscles`` excitation commands in ``[-1, 1]`` or ``[0, 1]`` (clamped to ``[0, 1]``
    after ``action * scale + offset``), advances per-env muscle activations ``act`` on every physics
    substep inside :meth:`apply_actions`, evaluates active + passive muscle forces via
    :func:`~eden.extensions.musculoskeletal.hill_muscle.muscle_actuator_force`, and injects
    generalized joint torques ``J_muscle.T @ f_muscle`` via ``entity.control_dofs_force``.

    Parameters
    ----------
    entity_name : str
        Name of the controlled entity.
    dofs_name : str | list[str]
        Controlled single-DOF joint names.
    muscle_params : MuscleParams | None
        Extracted Hill muscle parameters (``nu`` muscles). Required before ``build()``.
    rest_length : tuple[float, ...] | list[float] | None
        Optional ``(nu,)`` reference tendon lengths used with a constant ``moment_arm`` matrix when
        ``geometry_fn`` is not provided: ``L(q) = rest_length + (q - q0) @ moment_arm.T`` (note MuJoCo's
        convention ``dL/dq = -moment_arm`` or direct moment arm depending on sign; here ``moment_arm``
        matches MuJoCo's ``actuator_moment`` where ``qfrc = moment_arm.T @ force`` and ``L_dot = moment_arm @ qvel``).
    moment_arm : list[list[float]] | None
        Constant ``(nu, n_dofs)`` moment-arm matrix used when ``geometry_fn`` is ``None``.
    geometry_fn : GeometryFn | None
        Optional callable ``(dofs_pos, dofs_vel) -> (length, velocity, moment_arm)`` returning tensors
        of shape ``(B, nu)``, ``(B, nu)``, and ``(B, nu, n_dofs)`` for nonlinear spatial-wrap paths.
    scale : float
        Scale applied to raw policy actions before clamping to ``[0, 1]``.
    offset : float
        Offset added after scaling (set ``scale=0.5, offset=0.5`` to map ``[-1, 1] -> [0, 1]``).
    initial_act : float
        Activation level assigned on episode reset.
    """

    muscle_params: MuscleParams | None = None
    rest_length: list[float] | None = None
    moment_arm: list[list[float]] | None = None
    geometry_fn: GeometryFn | None = None
    scale: float = 1.0
    offset: float = 0.0
    initial_act: float = 0.0

    def __init__(self, env: EnvBase, options: ActionTermOptions) -> None:
        super().__init__(env=env, options=options)
        self._act: torch.Tensor | None = None
        self._ctrl: torch.Tensor | None = None
        self._moment_const: torch.Tensor | None = None
        self._rest_len: torch.Tensor | None = None
        self._q0: torch.Tensor | None = None

    @property
    def action_dim(self) -> int:
        if self.muscle_params is None:
            raise ValueError(f"{term_subject(self)}: `muscle_params` must be provided.")
        return len(self.muscle_params.tau_act)

    @property
    def muscle_activation(self) -> torch.Tensor:
        """Current ``(num_envs, num_muscles)`` activation state."""
        return self._act

    def build(self) -> None:
        if self.muscle_params is None:
            raise ValueError(f"{term_subject(self)}: `muscle_params` is required to build HillMuscleController.")
        super().build()
        nu = self.action_dim
        if self.geometry_fn is None:
            if self.moment_arm is None or self.rest_length is None:
                raise ValueError(
                    f"{term_subject(self)}: provide either `geometry_fn` or both `moment_arm` and `rest_length`."
                )
            mat = torch.as_tensor(self.moment_arm, dtype=torch.float32, device=self.device)
            rlen = torch.as_tensor(self.rest_length, dtype=torch.float32, device=self.device)
            if mat.shape != (nu, self._n_dofs):
                raise ValueError(
                    f"{term_subject(self)}: `moment_arm` shape {tuple(mat.shape)} must match "
                    f"(num_muscles={nu}, num_dofs={self._n_dofs})."
                )
            if rlen.shape != (nu,):
                raise ValueError(f"{term_subject(self)}: `rest_length` shape {tuple(rlen.shape)} must be ({nu},).")
            self._moment_const = mat
            self._rest_len = rlen
            self._q0 = self.entity.get_default_dofs_pos(self.dofs_name)

        self._act = torch.full((self.num_envs, nu), float(self.initial_act), dtype=torch.float32, device=self.device)
        self._ctrl = torch.zeros((self.num_envs, nu), dtype=torch.float32, device=self.device)

    def reset(self, envs_idx: slice | torch.Tensor | None = None) -> None:
        idx = slice(None) if envs_idx is None else envs_idx
        self._raw_action[idx] = 0.0
        self._prev_action[idx] = 0.0
        self._processed_action[idx] = 0.0
        self._act[idx] = float(self.initial_act)
        self._ctrl[idx] = 0.0

    def compute(self, actions: torch.Tensor) -> None:
        self._prev_action[:] = self._raw_action
        self._raw_action[:] = actions
        self._ctrl = (actions * self.scale + self.offset).clamp(0.0, 1.0)

    def apply_actions(self) -> None:
        substep_dt = float(self.env.substep_dt)
        self._act, _ = step_muscle_activation(self._act, self._ctrl, self.muscle_params, dt=substep_dt, substeps=1)

        q = self.entity.get_dofs_pos(dofs_idx_local=self.genesis_dofs_idx)
        qd = self.entity.get_dofs_vel(dofs_idx_local=self.genesis_dofs_idx)
        if self.geometry_fn is not None:
            length, velocity, moment = self.geometry_fn(q, qd)
        else:
            dq = q - self._q0
            length = self._rest_len.unsqueeze(0) + dq @ self._moment_const.T
            velocity = qd @ self._moment_const.T
            moment = self._moment_const.unsqueeze(0).expand(self.num_envs, -1, -1)

        force = muscle_actuator_force(length, velocity, self._act, self.muscle_params)
        ctrl_torque = torch.einsum("buv,bu->bv", moment, force)
        self._processed_action = ctrl_torque
        self.verify_command_shape(ctrl_torque)
        self.entity.control_dofs_force(ctrl_torque, dofs_idx_local=self.genesis_dofs_idx)
