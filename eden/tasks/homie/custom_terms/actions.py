"""Action-side terms of the HOMIE task: OpenHomie's motor model and its upper-body pose curriculum."""

from __future__ import annotations

import math

import torch

from eden.managers.action_manager import ACTION_TERM_REGISTRY, ActionTerm
from eden.managers.modifiers.base import ACTION_MODIFIER_REGISTRY, ActionModifier
from eden.utils.misc import assign_over_envs


@ACTION_MODIFIER_REGISTRY.register()
class HomieMotorModel(ActionModifier):
    """OpenHomie's ``control_type='M'`` torque post-processing (``legged_robot.py`` ``_compute_torques``)::

        tau = clip(tau + actuation_offset + joint_injection, -tau_limit, tau_limit)

    ``tau`` is the PD torque :class:`~eden.managers.terms.actions.ExplicitPDController` computed.
    ``actuation_offset`` is a per-episode, per-DOF torque bias and ``joint_injection`` per-control-step,
    per-DOF torque noise, both drawn from ``±scale · tau_limit``. The injection is drawn once per
    control step (``modify_processed_action``) and held across the physics steps
    (``modify_ctrl_torque``), as OpenHomie draws it once in ``step``. Outside training
    (``env.training`` is False) both are zero and only the clip remains.
    """

    rfi_scale: float = 0.05  # per-control-step joint injection, a fraction of the torque limit
    offset_scale: float = 0.05  # per-episode actuation offset, a fraction of the torque limit

    def build(self, num_envs: int, device: str, entity=None, dofs_idx_local=None) -> None:
        if entity is None or dofs_idx_local is None:
            raise ValueError(
                f"{type(self).__name__} needs the controlled entity and DOFs; attach it to a joint "
                f"controller's `modifier=` (e.g. ExplicitPDController), not to a term without DOFs."
            )
        _, upper = entity.get_dofs_force_range(dofs_idx_local=dofs_idx_local)
        # The effort limits are not randomized, so one row serves every env.
        self._tau_lim = upper[0].abs().to(device=device, dtype=torch.float32).reshape(1, -1)
        n_dofs = self._tau_lim.shape[-1]
        self._bias = torch.zeros(num_envs, n_dofs, device=device)
        self._noise = torch.zeros(num_envs, n_dofs, device=device)
        self.reset(None)

    def _sample(self, like: torch.Tensor, scale: float) -> torch.Tensor:
        return (torch.rand_like(like) * 2.0 - 1.0) * scale * self._tau_lim

    def reset(self, envs_idx: (slice | torch.Tensor) | None = None) -> None:
        if not self._env.training:
            return
        idx = slice(None) if envs_idx is None else envs_idx
        # assign_over_envs, not ``bias[mask] = sample[mask]``: the auto-reset hands a bool mask, and
        # gathering with one is a GPU->CPU sync on nearly every step.
        assign_over_envs(self._bias, idx, self._sample(self._bias, self.offset_scale))

    def modify_processed_action(self, processed_action: torch.Tensor) -> torch.Tensor:
        if self._env.training:
            self._noise.copy_(self._sample(self._noise, self.rfi_scale))
        return processed_action

    def modify_ctrl_torque(
        self,
        ctrl_torque: torch.Tensor,
        dofs_vel: torch.Tensor,
        pos_err: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return (ctrl_torque + self._bias + self._noise).clamp(-self._tau_lim, self._tau_lim)


@ACTION_TERM_REGISTRY.register()
class UpperBodyPoseCurriculum(ActionTerm):
    """Random, curriculum-widened pose targets for the non-policy upper body (waist + arms).

    OpenHomie's upper-body disturbance (``legged_robot.py`` ``step`` and ``update_action_curriculum``):
    every ``resample_interval_s`` each joint draws a fraction of the way from its default toward a
    coin-flipped hard limit — a curriculum-tempered truncated exponential (concentrated near 0 early,
    about uniform on ``[0, 1]`` once ramped) times a uniform — and the joint ramps there linearly over
    the interval. The term takes no policy action (``action_dim = 0``) and drives its joints with
    Genesis position control. Outside training (``env.training`` is False) it holds the default pose,
    as the upper body is held at deployment.

    The one deviation is the curriculum ratio: a fixed linear ramp over ``curriculum_length_steps``
    control steps here, where OpenHomie steps it up on velocity-tracking performance. The ramp
    counter is not checkpointed, so a resumed run restarts it.

    Parameters
    ----------
    resample_interval_s : float
        Seconds between pose targets (OpenHomie ``upper_interval_s``).
    curriculum_length_steps : int
        Control steps over which the ratio ramps from 0 to ``max_ratio``.
    max_ratio : float
        The final curriculum ratio.
    action_scale : float
        The leg policy's action scale, to express the applied pose in the same normalized-action units
        (:attr:`normalized_action`) as the leg actions.
    """

    entity_name: str = "robot"
    dofs_name: list[str] = []
    resample_interval_s: float = 1.0
    curriculum_length_steps: int = 250_000
    max_ratio: float = 1.0
    action_scale: float = 0.25

    @property
    def action_dim(self) -> int:
        return 0

    @property
    def normalized_action(self) -> torch.Tensor:
        """The applied pose as ``(target - default) / action_scale``, shape ``(num_envs, num_dofs)``.

        OpenHomie's ``self.actions`` carries the curriculum's targets in these units next to the leg
        actions, and its ``action_rate`` / ``smoothness`` penalties span both.
        """
        return (self._processed_action - self._default) / self.action_scale

    def build(self) -> None:
        super().build()
        entity = self.entity
        dofs = entity.find_named_dofs(self.dofs_name)
        self._default = entity.get_default_dofs_pos(dofs)
        # OpenHomie samples toward the HARD limits (its action_min/max come from hard_dof_pos_limits). A full-width
        # get_dofs_limit read is laid out in columns, like the property tensors.
        lower, upper = entity.get_dofs_limit()
        self._lower = lower[..., dofs.columns].expand(self.num_envs, -1).clone()
        self._upper = upper[..., dofs.columns].expand(self.num_envs, -1).clone()
        self._n_interp = max(1, round(self.resample_interval_s / self._env.dt))
        self._processed_action[:] = self._default
        self._delta = torch.zeros_like(self._processed_action)
        self._step = 0

    def _ratio(self) -> float:
        return min(self.max_ratio, self.max_ratio * self._step / max(self.curriculum_length_steps, 1))

    def _resample_targets(self) -> None:
        ratio = min(1.0, max(0.0, self._ratio()))
        k = 1.0 - 0.99 * ratio  # the truncated exponential's rate is 20 k, k in [0.01, 1]
        u = torch.rand_like(self._processed_action)
        upper_ratio = (-1.0 / (20.0 * k)) * torch.log(1.0 - u + u * math.exp(-20.0 * k))  # inverse CDF on [0, 1]
        joint_ratio = upper_ratio * torch.rand_like(self._processed_action)
        bound = torch.where(torch.rand_like(self._processed_action) >= 0.5, self._lower, self._upper)
        target = (self._default + (bound - self._default) * joint_ratio).clamp(self._lower, self._upper)
        self._delta.copy_((target - self._processed_action) / self._n_interp)

    def compute(self, actions: torch.Tensor) -> None:
        if not self._env.training:
            return
        if self._step % self._n_interp == 0:  # one global resample every interval, as OpenHomie
            self._resample_targets()
        self._step += 1
        self._processed_action.add_(self._delta)

    def apply_actions(self) -> None:
        self.entity.control_dofs_pos(self._processed_action, dofs_idx_local=self.genesis_dofs_idx)

    def reset(self, envs_idx: (slice | torch.Tensor) | None = None) -> None:
        # Back to the default pose, ramp frozen until the next global resample (OpenHomie zeroes
        # the current upper action and its delta on reset).
        idx = slice(None) if envs_idx is None else envs_idx
        assign_over_envs(self._processed_action, idx, self._default)  # sync-free for a bool mask
        self._delta[idx] = 0.0
