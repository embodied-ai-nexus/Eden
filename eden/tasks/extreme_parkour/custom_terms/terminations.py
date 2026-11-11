"""Termination terms for the Extreme Parkour teacher task.

All termination terms of this task (these two plus the reused ``time_out`` /
roll / pitch limits) are declared with ``time_out=True`` in the config, so the
runner bootstraps the value at episode end (the source treats every reset as a
truncation; Eden's rsl_rl wrapper forwards truncations as ``time_outs``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from eden.managers import TERMINATION_TERM_REGISTRY
from eden.tasks.extreme_parkour.custom_terms.commands import ParkourCommand

if TYPE_CHECKING:
    from eden.envs.base import RLEnvBase


@TERMINATION_TERM_REGISTRY.register()
def base_height_below(env: RLEnvBase, *, entity_name: str = "robot", threshold: float = -0.25) -> torch.Tensor:
    """Terminate when the base drops below an ABSOLUTE world height.

    The parkour terrain's z-origin is 0 and pits/gaps are carved to negative
    heights, so a base below ``threshold`` (default -0.25 m) has fallen into one.
    """
    return env.entities[entity_name].get_pos()[:, 2] < threshold


@TERMINATION_TERM_REGISTRY.register()
def all_goals_reached(env: RLEnvBase, *, command_name: str) -> torch.Tensor:
    """Terminate (truncate) once the env has advanced past its last real goal: ``ParkourCommand.goal_progress`` is 1."""
    command: ParkourCommand = env.command_manager.get_term(command_name)
    return command.goal_progress >= 1.0
