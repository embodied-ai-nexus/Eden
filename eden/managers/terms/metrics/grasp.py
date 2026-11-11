"""Grasp metric terms (object lifted, is-grasping)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from eden.managers.metric_manager import METRIC_TERM_REGISTRY, MetricTerm
from eden.managers.terms.utils import parallel_jaw_is_grasping
from eden.managers.termination_manager import TERMINATION_TERM_REGISTRY

if TYPE_CHECKING:
    from eden.envs.base import EnvBase
    from eden.entities.base import Entity
    from eden.options.managers.metrics import MetricTermOptions


@TERMINATION_TERM_REGISTRY.register()
@METRIC_TERM_REGISTRY.register()
def object_lifted(
    env: EnvBase,
    *,
    entity_name: str,
    min_height: float,
) -> torch.Tensor:
    """Check if an entity has been lifted above *min_height*.

    Returns a per-env bool tensor, True when the entity's root z-position
    (solver frame) exceeds *min_height*.

    Parameters
    ----------
    env: EnvBase
        The environment instance.
    entity_name: str
        Name of the entity.
    min_height: float
        World-z height threshold [m].
    """
    entity: Entity = env.get_entity(entity_name)
    return entity.get_pos()[:, 2] > min_height


@METRIC_TERM_REGISTRY.register()
class IsGrasping(MetricTerm):
    """Detect a parallel-jaw grasp via contact force alignment.

    Returns a per-env bool, True when **both** gripper pads have contact
    forces aligned with their closing direction above *force_threshold*.
    This correctly detects a grasp even when the fingers cannot fully
    close (e.g. when an object is between them).

    Shares the force test of :func:`~eden.managers.terms.utils.parallel_jaw_is_grasping` with
    :class:`~eden.managers.terms.actions.welding_actions.ParallelJawWelding`, but sums each pad's
    contacts over every link it touches. A grasp spanning two links of one object counts, and so
    do pads pressing two different objects. The weld action instead requires both pads on the one
    link it would weld, so this metric can read True in a step that welds nothing.

    Parameters
    ----------
    robot_name : str
        Name of the robot entity.
    left_gripper_link_name : str
        Link name of the left gripper pad (e.g. ``"left_gripper_pad"``).
    right_gripper_link_name : str
        Link name of the right gripper pad (e.g. ``"right_gripper_pad"``).
    force_threshold : float
        Minimum projected contact force magnitude on **each** pad to
        count as grasping.  Lower values are more sensitive.
    """

    robot_name: str = "robot"
    left_gripper_link_name: str = "left_gripper_pad"
    right_gripper_link_name: str = "right_gripper_pad"
    force_threshold: float = 1.0

    def __init__(self, env: EnvBase, options: MetricTermOptions):
        super().__init__(env=env, options=options)
        self._left_link_idx: int = -1
        self._right_link_idx: int = -1
        self._gripper_idx_local: list[int] = []

    def build(self):
        entity = self.resolve_entity(self.robot_name)
        self._entity = entity
        self._left_link_idx = entity.get_link(self.left_gripper_link_name).idx
        self._right_link_idx = entity.get_link(self.right_gripper_link_name).idx
        self._gripper_idx_local = [
            entity.get_link(self.left_gripper_link_name).idx_local,
            entity.get_link(self.right_gripper_link_name).idx_local,
        ]

    def compute(self) -> torch.Tensor:
        contact_info = self._entity.get_contacts(exclude_self_contact=True)
        return parallel_jaw_is_grasping(
            self._entity,
            contact_info,
            self._left_link_idx,
            self._right_link_idx,
            self._gripper_idx_local,
            self.force_threshold,
        )
