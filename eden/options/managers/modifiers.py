"""Configuration options for action/observation modifiers (including noise)."""

from eden.constants import NoiseOperation
from eden.options.managers.base import ManagerTermOptions


class NoiseOptions(ManagerTermOptions):
    """
    Noise options.

    Parameters
    ----------
    operation: NoiseOperation, optional
        The operation to apply to the noise. Default is NoiseOperation.ADD.
    """

    operation: NoiseOperation = NoiseOperation.ADD


class ActionModifierOptions(ManagerTermOptions):
    """Options for action modifiers."""

    ...


class ObservationModifierOptions(ManagerTermOptions):
    """Options for an observation modifier — one entry of an observation term's ``modifiers`` list.

    Built by ``<ModifierClass>.configure(...)``; the modifier's own parameters ride as extras, validated
    against the registered class like any term's.
    """

    ...
