"""GMT observation terms (ankle-zeroed joint velocities)."""

from __future__ import annotations
from typing import TYPE_CHECKING

import torch

from eden.managers import OBSERVATION_TERM_REGISTRY

if TYPE_CHECKING:
    from eden.envs.base import EnvBase
    from eden.entities.base import Entity


@OBSERVATION_TERM_REGISTRY.register()
def dofs_vel_no_ankle_gmt(
    env: EnvBase,
    *,
    entity_name: str,
    envs_idx: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return joint velocities of the entity with the ankle pitch/roll columns (4, 5, 10, 11) zeroed."""
    robot: Entity = env.entities[entity_name]
    dofs_vel = robot.get_dofs_vel(envs_idx=envs_idx).clone()
    dofs_vel[..., [4, 5, 10, 11]] = 0.0
    return dofs_vel
