"""Locomotion termination terms (base orientation thresholds, base height floor)."""

from __future__ import annotations
from typing import TYPE_CHECKING

import math
import torch
from genesis.utils.geom import quat_to_xyz

from eden.managers.termination_manager import TERMINATION_TERM_REGISTRY
from eden.managers.terms.utils import base_height_above_ground, resolve_terrain
from eden.utils.misc import resolve_named_entity

if TYPE_CHECKING:
    from eden.envs.base import EnvBase
    from eden.entities.base import Entity


@TERMINATION_TERM_REGISTRY.register()
def base_orientation_roll_greater_than(env: EnvBase, *, entity_name: str, threshold: float) -> torch.Tensor:
    """Terminate when the base roll exceeds ``threshold`` degrees."""
    robot: Entity = env.get_entity(entity_name)
    rpy = quat_to_xyz(robot.get_quat(), rpy=True)  # radians
    threshold_rad = math.radians(threshold)
    return torch.abs(rpy[:, 0]) > threshold_rad


@TERMINATION_TERM_REGISTRY.register()
def base_orientation_pitch_greater_than(env: EnvBase, *, entity_name: str, threshold: float) -> torch.Tensor:
    """Terminate when the base pitch exceeds ``threshold`` degrees."""
    robot: Entity = env.get_entity(entity_name)
    rpy = quat_to_xyz(robot.get_quat(), rpy=True)
    threshold_rad = math.radians(threshold)
    return torch.abs(rpy[:, 1]) > threshold_rad


@TERMINATION_TERM_REGISTRY.register()
def root_height_below_minimum(
    env: EnvBase,
    *,
    entity_name: str,
    minimum_height: float,
    terrain_entity_name: str | None = None,
    relative: bool = False,
) -> torch.Tensor:
    """Terminate when the entity's base sinks below ``minimum_height`` above the ground.

    The height floor half of a fall detector, complementing
    :func:`~eden.managers.terms.terminations.velocity_tracking.fell_over` (which watches the
    lean angle): a robot that collapses onto its belly while staying upright trips this one and not
    that one. Shares :func:`~eden.managers.terms.utils.base_height_above_ground` with the
    :func:`~eden.managers.terms.observations.locomotion.base_height` observation, so a policy that
    sees its height and an episode that ends on it agree on the number.

    Parameters
    ----------
    env : EnvBase
        The environment instance.
    entity_name : str
        Entity whose base height is checked.
    minimum_height : float
        Floor, in metres. The term fires while ``height < minimum_height`` — strictly below, so a
        robot resting exactly at the floor survives.
    terrain_entity_name : str | None, optional
        Terrain entity to measure against. ``None`` (the default) compares the world-frame ``z``,
        which matches Isaac Lab's ``root_height_below_minimum`` and is correct on flat ground only;
        naming a :class:`~eden.entities.terrain.Terrain` compares the height above the terrain under
        the base instead, which is what a rough-terrain task wants — a world-``z`` floor on a
        heightfield fires in every valley and never on a hill.
    relative : bool, optional
        Frame the base pose is read in — see
        :func:`~eden.managers.terms.observations.proprio.base_pos`. Defaults to ``False`` (the solver
        frame), matching the base-pose observations and
        :func:`~eden.managers.terms.rewards.locomotion.base_height_l2`. Pass ``True`` to threshold
        the authored base-link origin, the frame ``default_root_pos`` and Isaac Lab's ``root_pos_w``
        are in; the two differ by the base link's centre-of-mass offset.
    """
    terrain = resolve_terrain(env.entities, terrain_entity_name)
    robot: Entity = resolve_named_entity(env.entities, entity_name)
    return base_height_above_ground(robot, terrain, relative=relative) < minimum_height
