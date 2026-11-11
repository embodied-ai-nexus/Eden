"""Termination terms for humanoid velocity tracking tasks."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from eden.managers.termination_manager import TERMINATION_TERM_REGISTRY
from eden.managers.terms.utils import gravity_vec, projected_gravity

if TYPE_CHECKING:
    from eden.envs.base import EnvBase
    from eden.entities.base import Entity


@TERMINATION_TERM_REGISTRY.register()
def fell_over(env: EnvBase, *, entity_name: str, limit_angle: float) -> torch.Tensor:
    """Terminate when the robot leans too far from vertical.

    Uses projected gravity to determine the lean angle. The robot is considered
    fallen when the angle between the gravity vector in body frame and the
    world z-axis exceeds ``limit_angle`` (in radians).
    """
    robot: Entity = env.get_entity(entity_name)
    proj_grav = projected_gravity(robot.get_quat(), gravity_vec(env.num_envs, env.device))
    # The z-component of projected gravity is -cos(lean_angle) (it is -1.0 when upright).
    cos_angle = -proj_grav[:, 2]
    return cos_angle < math.cos(limit_angle)
