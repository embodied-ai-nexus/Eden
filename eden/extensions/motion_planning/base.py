"""Base class for motion planners."""

from __future__ import annotations
from abc import abstractmethod
from typing import TYPE_CHECKING, Any

import torch

from eden.utils.common import ConfigurableMixin
from eden.utils.misc import resolve_named_entity
from eden.options.extensions.motion_planning import PlannerOptions
from eden.utils.registry import Registry

if TYPE_CHECKING:
    from eden.entities.base import Entity
    from eden.envs.base import EnvBase


MOTION_PLANNER_REGISTRY = Registry("MOTION_PLANNER")


class MotionPlannerBase(ConfigurableMixin[PlannerOptions]):
    """Abstract base class for motion planners.

    Subclasses provide backend-specific goal representations (joint-space, task-space, etc.)
    and return planned waypoints in joint space. The shared option ``entity_name`` names the
    entity to plan motions for (resolved at construction).

    Parameters
    ----------
    env : EnvBase
        The environment instance.
    options : PlannerOptions
        From the backend's ``configure()``.
    """

    entity_name: str = ""

    def __init__(self, env: EnvBase, options: PlannerOptions) -> None:
        super().__init__(options)
        self.env = env
        self._entity = resolve_named_entity(
            env.entities, self.entity_name, hint="PlannerOptions.entity_name names it; set it to one of those."
        )

    @property
    def entity(self) -> Entity:
        return self._entity

    @property
    def allow_batch(self) -> bool:
        """Whether the planner allows batch processing."""
        return False

    @abstractmethod
    def plan_motion(self, **kwargs: Any) -> list[torch.Tensor]:
        """Plan motion to a target configuration.

        Subclasses define their own goal specification (joint-space, task-space, etc.)
        via keyword arguments.

        Returns
        -------
        list[torch.Tensor]
            Planned waypoints in joint space. The layout is backend-specific: one
            ``(num_waypoints, n_dof)`` tensor per environment (cuRobo), or one ``(n_qs,)`` tensor
            per waypoint of a single environment (OMPL).
        """
        pass
