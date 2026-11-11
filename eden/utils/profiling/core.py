"""Shared instrumentation core for the per-term profilers.

Both front-ends attribute wall time to individual manager *terms* by the same
mechanism: find every term the step loop runs, wrap its ``compute()`` with a
CUDA-synced timer, and restore it afterwards. That mechanism lives here so there
is a single source of truth for

* the coupling to manager term storage (:func:`iter_terms`), and
* the reversible, sync-both-sides timing wrapper (:func:`time_call` times one
  call, :class:`Patch` installs and removes a wrapper, :func:`time_method` joins
  the two). The one-shot profiler uses :func:`time_method`; the live-loop one
  builds its own wrapper from :func:`time_call` and :class:`Patch`, to subtract
  the calls nested inside a call from its own time.

The two front-ends differ only in *policy* — what they measure and how they
report it — not in this *mechanism*:

* :class:`eden.utils.profiling.step_profiler.RLStepProfiler` attaches to a
  caller-driven loop (training / rollout), also times manager-level methods, and
  produces a full per-manager rollup. Use it to profile terms *in an actual loop*.
* :func:`eden.utils.profiling.term_profiler.profile_terms` owns a short,
  self-driven window and produces a compact, agent-readable ranking. Use it to
  profile *a term you just wrote*, in isolation.
"""

from __future__ import annotations

from time import perf_counter
from typing import TYPE_CHECKING, Any, Callable, Iterator

if TYPE_CHECKING:
    from eden.envs.base import EnvBase

# Every manager exposes a public term iterator: :attr:`FlatManagerBase.terms`
# for the flat managers, and ``iter_terms()`` on the observation (by group) and
# event (by mode) managers. :func:`iter_terms` composes those — it touches no
# private manager storage.
FLAT_TERM_MANAGERS = ("action", "reward", "termination", "command", "curriculum", "metric")


def resolve_sync(sync: bool | None) -> Callable[[], None] | None:
    """Return the CUDA-synchronize callable to use around timed calls, or ``None``.

    Parameters
    ----------
    sync : bool or None
        ``True`` to synchronize, ``False`` to skip, ``None`` to auto-enable only
        when a CUDA device is present. In every case a synchronize callable is
        returned only if CUDA is actually available (there is nothing to wait on
        otherwise).

    Returns
    -------
    callable or None
        ``torch.cuda.synchronize`` when syncing, else ``None``.
    """
    import torch

    if sync is None:
        sync = torch.cuda.is_available()
    return torch.cuda.synchronize if (sync and torch.cuda.is_available()) else None


def iter_terms(
    env: "EnvBase", *, include_event: bool = True, include_action: bool = True
) -> Iterator[tuple[str, str, Any]]:
    """Yield ``(manager, label, term)`` for every term the step loop runs.

    Composes the public term iterators of the three manager shapes: the
    observation manager's ``iter_terms()`` (per group), the event manager's
    ``iter_terms()`` (per mode), and the flat ``terms`` dict every
    :class:`~eden.managers.base.FlatManagerBase` exposes. Managers absent on this
    env (e.g. a task with no command manager) are skipped.

    Parameters
    ----------
    env : EnvBase
        A built environment.
    include_event : bool
        Whether to include event-manager terms. They fire on reset / interval
        through mode dispatch rather than once per step, so a per-step cost view
        (the one-shot profiler) excludes them; the live-loop profiler includes
        them.
    include_action : bool
        Whether to include action-manager terms. An action term's dominant cost
        is in ``apply_actions`` (per decimation substep), not the ``compute`` this
        module times, so a compute-only view (the one-shot profiler) excludes it
        to avoid reporting a misleading ~0; the live-loop profiler, which also
        times ``apply_actions``, includes it.

    Yields
    ------
    tuple of (str, str, object)
        The owning manager's name, a human-readable label (``"group.term"`` for
        observations, ``"mode.term"`` for events, the bare term name otherwise),
        and the term instance.
    """
    om = getattr(env, "observation_manager", None)
    if om is not None:
        for group, name, term in om.iter_terms():
            yield "observation", f"{group}.{name}", term
    if include_event:
        ev = getattr(env, "event_manager", None)
        if ev is not None:
            for mode, name, term in ev.iter_terms():
                yield "event", f"{mode}.{name}", term
    for mgr_name in FLAT_TERM_MANAGERS:
        if mgr_name == "action" and not include_action:
            continue
        mgr = getattr(env, f"{mgr_name}_manager", None)
        if mgr is None:
            continue
        for name, term in mgr.terms.items():
            yield mgr_name, name, term


def can_time(owner: Any) -> bool:
    """Whether ``owner`` can carry the per-instance ``compute`` override the timers install.

    :func:`time_method` shadows the method with an instance attribute, so the owner
    needs a writable ``__dict__``. A term defining ``__slots__`` without one cannot
    be wrapped — the ``setattr`` would raise. No built-in term is slotted, but an
    author could write one, so :func:`discover_timeable` skips such a term instead
    of crashing the profiler. Manager instances always qualify.
    """
    return hasattr(owner, "__dict__")


def discover_timeable(
    env: "EnvBase", *, include_event: bool = True, include_action: bool = True
) -> tuple[list[tuple[str, str, Any]], list[str]]:
    """Split :func:`iter_terms` into the terms the profilers can time and the labels they can't.

    Dedups by term identity (a term shared across groups is timed once, not
    double-counted) and drops any term whose ``compute`` is missing or which cannot
    be wrapped (:func:`can_time`). Both front-ends discover through this, so the
    "which terms are timeable" policy has a single home.

    Parameters
    ----------
    env : EnvBase
        A built environment.
    include_event, include_action : bool
        Forwarded to :func:`iter_terms`.

    Returns
    -------
    tuple of (list, list)
        ``(timeable, skipped_labels)`` — the ``(manager, label, term)`` triples to
        wrap, and the labels dropped as un-timeable. Pass the latter to
        :func:`warn_untimed` so the omission is surfaced rather than silent.
    """
    timeable: list[tuple[str, str, Any]] = []
    skipped: list[str] = []
    seen: set[int] = set()
    for manager, label, term in iter_terms(env, include_event=include_event, include_action=include_action):
        if id(term) in seen:
            continue
        seen.add(id(term))
        if not callable(getattr(term, "compute", None)):
            continue
        if not can_time(term):
            skipped.append(label)
            continue
        timeable.append((manager, label, term))
    return timeable, skipped


def warn_untimed(context: str, labels: list[str]) -> None:
    """Warn that ``labels`` were skipped as un-timeable; a no-op when empty.

    Keeps the message identical across both front-ends. Imports ``eden`` lazily so
    importing this module never triggers the package init.
    """
    if not labels:
        return
    import eden as en

    en.logger.warning(
        f"{context}: skipped {len(labels)} term(s) that define __slots__ (no __dict__) and "
        f"cannot be timed — omitted from the profile: {', '.join(labels)}"
    )


class Patch:
    """A reversible replacement of one instance method with a wrapper.

    Records whether the method was a per-instance attribute or an inherited class
    method, so :meth:`restore` puts back exactly what was there — deleting the
    instance shadow in the common class-method case rather than leaving a bound
    method attribute behind.
    """

    __slots__ = ("_owner", "_name", "_had_own", "_original")

    def __init__(self, owner: Any, name: str, wrapper: Callable) -> None:
        self._owner = owner
        self._name = name
        self._had_own = name in owner.__dict__
        self._original = owner.__dict__.get(name)  # None => inherited class method
        setattr(owner, name, wrapper)

    def restore(self) -> None:
        """Undo the patch, restoring the original bound/class method."""
        if self._had_own:
            setattr(self._owner, self._name, self._original)
        elif self._name in self._owner.__dict__:
            delattr(self._owner, self._name)


def time_call(fn: Callable[[], Any], on_dt: Callable[[float], None], synchronize: Callable[[], None] | None) -> Any:
    """Run ``fn`` once, reporting its wall time to ``on_dt``.

    Syncs on *both* sides when ``synchronize`` is given: the pre-sync drains
    kernels queued by earlier work so they are not charged here, the post-sync
    waits out this call's own kernels. Returns ``fn``'s result.

    A call that raises is still reported, with its wall time up to the raise, and
    the exception propagates. It skips the post-sync: after a CUDA error that sync
    would raise again, over the original exception.
    """
    if synchronize is not None:
        synchronize()
    t0 = perf_counter()
    try:
        out = fn()
    except BaseException:
        on_dt(perf_counter() - t0)
        raise
    if synchronize is not None:
        synchronize()
    on_dt(perf_counter() - t0)
    return out


def time_method(owner: Any, name: str, on_dt: Callable[[float], None], synchronize: Callable[[], None] | None) -> Patch:
    """Wrap ``owner.name`` so each call's wall time is reported to ``on_dt``.

    Returns a :class:`Patch`; call :meth:`Patch.restore` to remove the wrapper.
    Timing is delegated to :func:`time_call`, so both levels of instrumentation
    share one clock and one sync policy.
    """
    orig = getattr(owner, name)

    def wrapper(*args, **kwargs):
        return time_call(lambda: orig(*args, **kwargs), on_dt, synchronize)

    return Patch(owner, name, wrapper)
