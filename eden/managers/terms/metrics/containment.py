"""Containment metric terms (object above height / inside region)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from eden.managers.metric_manager import METRIC_TERM_REGISTRY
from eden.managers.termination_manager import TERMINATION_TERM_REGISTRY
from eden.utils.torch import device_constant

if TYPE_CHECKING:
    from eden.envs.base import EnvBase
    from eden.entities.base import Entity


@TERMINATION_TERM_REGISTRY.register()
@METRIC_TERM_REGISTRY.register()
def object_above_height(
    env: EnvBase,
    *,
    entity_name: str,
    height: float,
    relative: bool = False,
) -> torch.Tensor:
    """Check if an entity's root z-position is above *height*.

    Returns a per-env bool tensor (True if above).

    Parameters
    ----------
    env: EnvBase
        The environment instance.
    entity_name: str
        Name of the entity.
    height: float
        Minimum height threshold.
    relative: bool
        Frame *height* is measured in. ``False`` (the default) reads the solver frame, which Genesis
        re-origins onto a rigid body's inertial frame; ``True`` reads the frame the asset was
        authored in. Pass ``True`` whenever *height* was reasoned from the asset's own geometry: the
        two frames differ by the body's centre-of-mass offset, and because that offset is expressed
        in the body frame, the reported z swings by it as the body tilts.

        Only a rigid entity has the second frame, so ``True`` against a deformable raises from
        :meth:`~eden.entities.base.Entity._reject_relative`. Unlike the observation terms, which the
        manager warm-computes while sizing their buffers, this is a metric term and so first runs on
        step 1 rather than at env construction.
    """
    entity: Entity = env.get_entity(entity_name)
    return entity.get_pos(relative=relative)[:, 2] > height


@METRIC_TERM_REGISTRY.register()
def object_bottom_above_height(
    env: EnvBase,
    *,
    entity_name: str,
    height: float,
) -> torch.Tensor:
    """Lowest point of the entity above *height*.

    Unlike :func:`object_above_height`, this does not depend on where the asset puts its origin (a
    scanned mesh's origin may sit at its base, its centre or off to one side), so ``height=0.03`` with
    the table at ``z = 0`` reads "lifted 3 cm off the table" for any object.

    Parameters
    ----------
    env: EnvBase
        The environment instance.
    entity_name: str
        Name of the entity.
    height: float
        World ``z`` [m] the entity's bounding box must clear.
    """
    entity: Entity = env.get_entity(entity_name)
    return entity.get_AABB()[:, 0, 2] > height


@TERMINATION_TERM_REGISTRY.register()
@METRIC_TERM_REGISTRY.register()
def object_in_region(
    env: EnvBase,
    *,
    entity_name: str,
    region_min: list[float],
    region_max: list[float],
    relative: bool = False,
) -> torch.Tensor:
    """Check if an entity's root position is inside an axis-aligned box.

    Returns a per-env bool tensor (True if inside, bounds inclusive).

    Parameters
    ----------
    env: EnvBase
        The environment instance.
    entity_name: str
        Name of the entity.
    region_min: list[float]
        Lower corner of the box ``[x_min, y_min, z_min]``.
    region_max: list[float]
        Upper corner of the box ``[x_max, y_max, z_max]``.
    relative: bool
        Frame the box is expressed in — see :func:`object_above_height`, including why ``True``
        needs a rigid entity and only says so on step 1. Pass ``True`` when the corners were
        reasoned from the asset's own geometry rather than measured in the solver frame.
    """
    entity: Entity = env.get_entity(entity_name)
    pos = entity.get_pos(relative=relative)
    # Built once per distinct box, not per call: an upload per step is a GPU->CPU sync.
    lo = device_constant(region_min, env.device).unsqueeze(0)
    hi = device_constant(region_max, env.device).unsqueeze(0)
    inside = (pos >= lo).all(dim=1) & (pos <= hi).all(dim=1)
    return inside
