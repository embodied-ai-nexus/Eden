"""Base class for retargeting backends."""

from __future__ import annotations
from typing import TYPE_CHECKING
from abc import abstractmethod

from eden.utils.common import ConfigurableMixin
from eden.utils.misc import resolve_named_entity
from eden.options.extensions.retargeting import RetargeterOptions

if TYPE_CHECKING:
    from eden.entities.base import Entity
    from eden.envs.base import EnvBase


class RetargetingBase(ConfigurableMixin[RetargeterOptions]):
    """Abstract base class for retargeting backends.

    Resolves ``entity_name`` against ``env.entities`` at construction (raising on an unknown name);
    subclasses implement :meth:`retarget_motion`.
    """

    entity_name: str = ""

    def __init__(self, env: EnvBase, options: RetargeterOptions) -> None:
        super().__init__(options)
        self.env = env
        self._entity = resolve_named_entity(
            env.entities, self.entity_name, hint="RetargeterOptions.entity_name names it; set it to one of those."
        )

    @property
    def entity(self) -> Entity:
        return self._entity

    @abstractmethod
    def retarget_motion(self):
        raise NotImplementedError
