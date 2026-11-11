"""Termination terms for the HUSKY skateboarding task.

Ports of ``mjlab_husky.tasks.skater.mdp.terminations`` plus the source's
``bad_orientation`` computed in the pelvis (MJCF root) frame.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from eden.managers.termination_manager import TERMINATION_TERM_REGISTRY
from eden.tasks.husky.custom_terms.observations import pelvis_projected_gravity

if TYPE_CHECKING:
    from eden.envs.base import EnvBase
    from eden.tasks.husky.custom_terms.commands import SkateCommand


def _skate(env: EnvBase, command_name: str) -> SkateCommand:
    return env.command_manager.get_term(command_name)


@TERMINATION_TERM_REGISTRY.register()
def pelvis_fell_over(
    env: EnvBase, *, entity_name: str = "robot", limit_angle: float = math.radians(70.0)
) -> torch.Tensor:
    """Terminate when the pelvis tilts more than ``limit_angle`` from upright."""
    gravity_b = pelvis_projected_gravity(env, entity_name=entity_name)
    return torch.acos((-gravity_b[:, 2]).clamp(-1.0, 1.0)).abs() > limit_angle


@TERMINATION_TERM_REGISTRY.register()
def feet_off_board(env: EnvBase, *, command_name: str = "skate") -> torch.Tensor:
    """Terminate when neither foot registers contact with the deck."""
    return _skate(env, command_name).feet_contact_b.sum(dim=-1) == 0


@TERMINATION_TERM_REGISTRY.register()
def skate_illegal_contact(env: EnvBase, *, command_name: str = "skate") -> torch.Tensor:
    """Terminate on contact involving the shins, arms below the shoulder, or pelvis."""
    return _skate(env, command_name).illegal_contact
