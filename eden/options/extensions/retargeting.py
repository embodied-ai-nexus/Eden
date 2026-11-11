"""Configuration options for retargeting extensions."""

from __future__ import annotations

from eden.options.options import DynamicConfigurableOptions


class RetargeterOptions(DynamicConfigurableOptions):
    """Options for retargeting extensions.

    Parameters
    ----------
    entity_name : str
        The name of the entity to perform retargeting for.
    """

    entity_name: str = "robot"
