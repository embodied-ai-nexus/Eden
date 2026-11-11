"""Configuration options for motion-planning extensions."""

from __future__ import annotations

from eden.options.options import DynamicConfigurableOptions


class PlannerOptions(DynamicConfigurableOptions):
    """
    Options for motion planning extensions.

    Parameters
    ----------
    entity_name: str
        The name of the entity to plan motions for
    """

    entity_name: str = "robot"
