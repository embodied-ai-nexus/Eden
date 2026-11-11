"""Event manager and event-term configuration options."""

from pydantic import field_validator

from eden.constants import EventMode
from eden.options.managers.base import FunctionTermOptions, ManagerOptions


class EventTermOptions(FunctionTermOptions):
    """
    Event term specification.

    Parameters
    ----------
    mode: EventMode | str
        The mode of the event term. Accepts either an ``EventMode`` member
        or its string value (e.g. ``"reset"``).
    interval_range_s: tuple[float, float] | None
        ``(min, max)`` of the time between firings in seconds, resampled uniformly after each firing.
        Required when ``mode`` is ``"interval"``.
    is_global_time: bool
        Interval mode only: one countdown shared by all envs (the term fires with no ``envs_idx``)
        instead of one countdown per env.
    min_step_count_between_reset: int
        Reset mode only: the minimum number of global env steps since the term last fired on an env
        before a reset fires it on that env again. ``0`` fires it on every reset.
    priority: int
        Lower numbers run first within the same mode. Default is 0.
    """

    mode: EventMode = EventMode.RESET
    interval_range_s: tuple[float, float] | None = None
    is_global_time: bool = False
    min_step_count_between_reset: int = 0
    priority: int = 0

    @field_validator("mode", mode="before")
    @classmethod
    def _coerce_mode(cls, v):
        return EventMode(v) if isinstance(v, str) else v


class EventManagerOptions(ManagerOptions[EventTermOptions]):
    """
    Event manager options.

    Parameters
    ----------
    <event_term_name>: EventTermOptions
        The event terms configuration to be used.
    """
