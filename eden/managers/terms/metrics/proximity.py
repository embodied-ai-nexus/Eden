"""Proximity and relative-position metric terms between entities."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from genesis.typing import Vec3FType

from eden.constants import MetricDirection
from eden.managers.metric_manager import METRIC_TERM_REGISTRY, MetricTerm
from eden.managers.termination_manager import TERMINATION_TERM_REGISTRY
from eden.utils.torch import device_constant

if TYPE_CHECKING:
    from eden.entities.base import Entity
    from eden.envs.base import EnvBase


@TERMINATION_TERM_REGISTRY.register()
@METRIC_TERM_REGISTRY.register()
def object_a_is_on_b(
    env: EnvBase,
    *,
    entity_a_name: str,
    entity_b_name: str,
    xy_threshold: float = 0.03,
    height_threshold: float = 0.04,
    height_diff: float = 0.0,
):
    """Check if object_a is on object_b by comparing height and xy distance of object origins.

    This function can be used as a termination term or a metric term.

    Parameters
    ----------
    env: EnvBase
        The environment instance.
    entity_a_name: str
        The name of the entity_a.
    entity_b_name: str
        The name of the entity_b.
    xy_threshold: float
        The threshold for the xy distance.
    height_threshold: float
        The threshold for the height distance.
    height_diff: float
        The expected height difference.

    Returns
    -------
    success: torch.Tensor
        A boolean tensor indicating if the object_a is on the object_b.
    """
    object_a: Entity = env.get_entity(entity_a_name)
    object_b: Entity = env.get_entity(entity_b_name)

    pos_diff = object_a.get_pos() - object_b.get_pos()
    height_dist = torch.linalg.vector_norm(pos_diff[:, 2:], dim=1)
    xy_dist = torch.linalg.vector_norm(pos_diff[:, :2], dim=1)

    success = torch.logical_and(
        pos_diff[:, 2] > 0.0,  # object_a is above object_b
        torch.logical_and(xy_dist < xy_threshold, (height_dist - height_diff) < height_threshold),
    )
    return success


@METRIC_TERM_REGISTRY.register()
def object_a_within_b_xy(
    env: EnvBase,
    *,
    entity_a_name: str,
    entity_b_name: str,
    min_fraction: float = 0.8,
) -> torch.Tensor:
    """*min_fraction* of entity_a's footprint lies over entity_b's.

    Both footprints are the xy extent of the entity's world-frame bounding box (its collision geometry),
    so a cup over a cutting board, or a scissor dropped into a container, passes whatever its height:
    pair it with a height or contact check to require that it rests there. PolaRiS's ``is_within_xy``
    rubric checker (``eden.tasks.polaris``) is this test.

    Parameters
    ----------
    env: EnvBase
        The environment instance.
    entity_a_name: str
        The entity that must lie over the other.
    entity_b_name: str
        The entity it must lie over.
    min_fraction: float
        Smallest share of entity_a's footprint area that must overlap entity_b's, in ``(0, 1]``.
    """
    a = env.get_entity(entity_a_name).get_AABB()[..., :2]  # (num_envs, 2 [low, high], 2)
    b = env.get_entity(entity_b_name).get_AABB()[..., :2]
    overlap = (torch.minimum(a[:, 1], b[:, 1]) - torch.maximum(a[:, 0], b[:, 0])).clamp(min=0.0).prod(dim=1)
    area = (a[:, 1] - a[:, 0]).prod(dim=1)
    return overlap >= min_fraction * area


@METRIC_TERM_REGISTRY.register(direction=MetricDirection.LIB)
def entity_distance(
    env: EnvBase,
    *,
    entity_a_name: str,
    entity_b_name: str,
) -> torch.Tensor:
    """Distance between two entity root positions.

    Returns a float tensor of shape ``(num_envs,)`` with the Euclidean
    distance. Registered with ``direction=LIB`` (lower is better).

    Parameters
    ----------
    env: EnvBase
        The environment instance.
    entity_a_name: str
        Name of the first entity.
    entity_b_name: str
        Name of the second entity.
    """
    entity_a: Entity = env.get_entity(entity_a_name)
    entity_b: Entity = env.get_entity(entity_b_name)
    pos_diff = entity_a.get_pos() - entity_b.get_pos()
    return torch.linalg.vector_norm(pos_diff, dim=1)


@TERMINATION_TERM_REGISTRY.register()
@METRIC_TERM_REGISTRY.register()
def entity_near_target(
    env: EnvBase,
    *,
    entity_name: str,
    target_pos: Vec3FType | torch.Tensor,
    threshold: float = 0.05,
) -> torch.Tensor:
    """Check if an entity is within *threshold* of a fixed world position.

    Returns a per-env bool tensor (True if within threshold).

    Parameters
    ----------
    env: EnvBase
        The environment instance.
    entity_name: str
        Name of the entity to check.
    target_pos: list[float] | Tensor
        Target world position ``[x, y, z]``.  A plain list is converted to a
        tensor once per distinct value (``device_constant``) and reused.
    threshold: float
        Maximum allowed distance.
    """
    entity: Entity = env.get_entity(entity_name)
    if not isinstance(target_pos, torch.Tensor):
        target_pos = device_constant(target_pos, env.device).unsqueeze(0)
    dist = torch.linalg.vector_norm(entity.get_pos() - target_pos, dim=1)
    return dist < threshold


@METRIC_TERM_REGISTRY.register(direction=MetricDirection.LIB)
def entity_distance_xy(
    env: EnvBase,
    *,
    entity_a_name: str,
    entity_b_name: str,
) -> torch.Tensor:
    """Horizontal (XY) distance between two entity root positions.

    Like :func:`entity_distance` but ignores the Z axis.  Useful for
    placement tasks where objects sit at different heights (e.g. an apple
    on a plate).

    Parameters
    ----------
    env: EnvBase
        The environment instance.
    entity_a_name: str
        Name of the first entity.
    entity_b_name: str
        Name of the second entity.
    """
    entity_a: Entity = env.get_entity(entity_a_name)
    entity_b: Entity = env.get_entity(entity_b_name)
    pos_diff = entity_a.get_pos()[:, :2] - entity_b.get_pos()[:, :2]
    return torch.linalg.vector_norm(pos_diff, dim=1)


@TERMINATION_TERM_REGISTRY.register()
@METRIC_TERM_REGISTRY.register()
def entity_near_entity_xy(
    env: EnvBase,
    *,
    entity_a_name: str,
    entity_b_name: str,
    threshold: float = 0.05,
) -> torch.Tensor:
    """Check if two entities are within *threshold* horizontal (XY) distance.

    Returns a per-env bool tensor, True when the XY distance between root
    positions is below *threshold*.  Useful for placement tasks where objects
    naturally sit at different heights.

    Parameters
    ----------
    env: EnvBase
        The environment instance.
    entity_a_name: str
        Name of the first entity.
    entity_b_name: str
        Name of the second entity.
    threshold: float
        Maximum allowed horizontal distance.
    """
    entity_a: Entity = env.get_entity(entity_a_name)
    entity_b: Entity = env.get_entity(entity_b_name)
    pos_diff = entity_a.get_pos()[:, :2] - entity_b.get_pos()[:, :2]
    dist = torch.linalg.vector_norm(pos_diff, dim=1)
    return dist < threshold


@METRIC_TERM_REGISTRY.register()
class EeNearEntity(MetricTerm):
    """Check if the mean end-effector position is within *threshold* of a target entity.

    This is a class-based metric that caches link indices at ``build()`` time
    for efficiency.

    Parameters
    ----------
    robot_name: str
        Name of the robot entity.
    ee_link_names: list[str]
        Link names whose mean position is treated as the EE position.
    entity_name: str
        Name of the target entity.
    threshold: float
        Maximum allowed distance.
    """

    robot_name: str = "robot"
    ee_link_names: list[str] = []
    entity_name: str = "obj"
    threshold: float = 0.05

    def build(self):
        self._robot = self.resolve_entity(self.robot_name)
        self._target = self.resolve_entity(self.entity_name)
        _, ee_idx = self._robot.find_named_links_idx_local(self.ee_link_names, preserve_order=True)
        self._ee_idx_local = ee_idx

    def compute(self) -> torch.Tensor:
        ee_pos = self._robot.get_links_pos(ls_idx_local=self._ee_idx_local).mean(dim=1)
        obj_pos = self._target.get_pos()
        dist = torch.linalg.vector_norm(ee_pos - obj_pos, dim=1)
        return dist < self.threshold
