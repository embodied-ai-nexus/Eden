"""Command manager and command-term configuration options."""

from eden.options.managers.base import ManagerOptions, ManagerTermOptions


class CommandTermOptions(ManagerTermOptions):
    """
    Command term specification.

    Parameters
    ----------
    resampling_time_range: tuple[float, float] | None
        The ``(min, max)`` time before a command is resampled [s]. Default is None.
    debug_vis: bool
        Whether to visualize debug information. Defaults to False.
    """

    resampling_time_range: tuple[float, float] | None = None
    debug_vis: bool = False


class CommandManagerOptions(ManagerOptions[CommandTermOptions]):
    """
    Command manager options.

    Parameters
    ----------
    <command_term_name>: CommandTermOptions
        The command terms configuration to be used.
    """
