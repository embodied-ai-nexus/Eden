"""LIBERO Panda gripper action term."""

from __future__ import annotations

from typing import TYPE_CHECKING

import genesis as gs
import torch

from eden.managers.action_manager import ACTION_TERM_REGISTRY, ActionTerm
from eden.options.robots.libero import LiberoPanda
from eden.utils.string import term_subject

if TYPE_CHECKING:
    from eden.envs.base import EnvBase
    from eden.options.managers.actions import ActionTermOptions


@ACTION_TERM_REGISTRY.register()
class LiberoPandaGripperController(ActionTerm):
    """Stateful one-dimensional LIBERO Panda gripper controller.

    Positive actions close and negative actions open the gripper. The controller follows the official LIBERO Panda
    opening range and advances its normalized state by 0.01 on every simulation step.
    """

    def __init__(self, env: EnvBase, options: ActionTermOptions) -> None:
        super().__init__(env=env, options=options)
        self._current_action: torch.Tensor | None = None
        self._action_direction: torch.Tensor | None = None
        self._position_midpoint: torch.Tensor | None = None
        self._position_half_range: torch.Tensor | None = None

    @property
    def action_dim(self) -> int:
        return 1

    def build(self) -> None:
        super().build()
        if tuple(self.dofs_name) != LiberoPanda.gripper_dofs_name:
            raise ValueError(
                f"{term_subject(self)}: the LIBERO gripper must control {LiberoPanda.gripper_dofs_name}, "
                f"got {self.dofs_name}"
            )

        open_position = torch.tensor(
            [LiberoPanda.open_dofs_pos[name] for name in self.dofs_name],
            dtype=gs.tc_float,
            device=self.device,
        ).unsqueeze(0)
        close_position = torch.tensor(
            [LiberoPanda.close_dofs_pos[name] for name in self.dofs_name],
            dtype=gs.tc_float,
            device=self.device,
        ).unsqueeze(0)

        self._current_action = torch.zeros((self.num_envs, 1), dtype=gs.tc_float, device=self.device)
        self._action_direction = torch.zeros_like(self._current_action)
        self._position_midpoint = 0.5 * (open_position + close_position)
        self._position_half_range = 0.5 * (open_position - close_position)

    def compute(self, actions: torch.Tensor) -> None:
        expected_shape = (self.num_envs, self.action_dim)
        if tuple(actions.shape) != expected_shape:
            raise ValueError(
                f"LIBERO Panda gripper actions must have shape {expected_shape}, got {tuple(actions.shape)}"
            )
        self._prev_action[:] = self._raw_action
        self._raw_action[:] = actions

    def reset(self, envs_idx: slice | torch.Tensor | None = None) -> None:
        if envs_idx is None:
            envs_idx = slice(None)
        self._prev_action[envs_idx] = 0.0
        self._raw_action[envs_idx] = 0.0
        assert self._position_midpoint is not None
        self._processed_action[envs_idx] = self._position_midpoint
        assert self._current_action is not None
        self._current_action[envs_idx] = 0.0

    def apply_actions(self) -> None:
        assert self._current_action is not None
        assert self._action_direction is not None
        assert self._position_midpoint is not None
        assert self._position_half_range is not None
        torch.sign(self._raw_action, out=self._action_direction)
        normalized_step = 0.01
        self._current_action.add_(self._action_direction, alpha=normalized_step).clamp_(-1.0, 1.0)
        torch.mul(self._current_action, -self._position_half_range, out=self._processed_action)
        self._processed_action.add_(self._position_midpoint)
        self.verify_command_shape(self._processed_action)
        self.entity.control_dofs_pos(self._processed_action, dofs_idx_local=self.genesis_dofs_idx)


__all__ = [
    "LiberoPandaGripperController",
]
