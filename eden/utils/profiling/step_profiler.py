"""Live-loop per-term profiler for the RL step loop.

Instruments every manager *term*'s ``compute()`` — plus each manager's top-level
``compute()``, ``reset()`` and ``post_reset()`` and the other hooks the step loop
calls on it (the action manager's ``apply_actions()``, the motion manager's
``fill_data()``, ...) — so the wall-clock cost of observations / rewards /
terminations / commands / events / curriculum / metrics / actions / motion can be
attributed term by term *while a real loop drives the env*. Use it to surface the
framework-level overhead Eden spends dispatching and composing terms on top of the
raw Genesis physics step.

The headline numbers are:

* per-term mean/total wall time and share of the step,
* per-manager time split into ``terms`` (its term calls) and ``overhead`` (the
  manager's own methods around them) — no split for a manager whose terms are not
  timed (the motion manager's run inside ``fill_data``, not through ``compute``),
* the step-loop work no manager owns: the Genesis ``scene.step`` physics tick
  (``physics``) and the env's own reset work around the managers' resets
  (``reset``: ``scene.reset``, solver bookkeeping, buffer zeroing, index handling), and
* the ``unattributed`` remainder of the step — the Python in ``env.step`` between
  instrumented calls.

The per-term table and :meth:`RLStepProfiler.as_dict`'s ``total_ms_per_step`` are
inclusive: a call's wall time counts every instrumented call nested inside it. The
rollup instead charges each call only its *own* time (its wall time minus that of
the instrumented calls nested inside it), and only inside :meth:`RLStepProfiler.step`
windows, so its rows partition the measured step exactly. A RESET event that fills
the motion window is charged its own work under ``event`` and the fill under
``motion``; ``reset`` excludes everything instrumented inside it. With ``sync=True``
the syncs around a nested call land in its caller's own time. A call that raises is
still recorded, with its wall time up to the raise (no closing sync).

For the *one-shot* "profile the term I just wrote" workflow, see
:func:`eden.utils.profiling.term_profiler.profile_terms`. Both share the
term-discovery and timing mechanism in :mod:`eden.utils.profiling.core`.

GPU note
--------
Manager terms launch CUDA kernels asynchronously, so naive host-side timing
measures *launch* cost, not execution. With ``sync=True`` (default) the profiler
calls :func:`torch.cuda.synchronize` around every measured call, yielding true
wall time per term at the cost of serializing kernels that would otherwise
overlap — absolute numbers run high but the relative breakdown is faithful.
With ``sync=False`` you instead measure pure Python dispatch cost (useful for
spotting per-term CPU overhead). The two views answer different questions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from time import perf_counter
from typing import TYPE_CHECKING, Any, Iterator

from prettytable import PrettyTable

from eden.utils.profiling.core import Patch, discover_timeable, resolve_sync, time_call, warn_untimed

if TYPE_CHECKING:
    from eden.envs.base import EnvBase

# The step and the two halves of a reset (``_reset_idx`` fans ``reset`` out,
# ``_dispatch_post_reset`` ``post_reset``).
_STEP_AND_RESET = ("compute", "reset", "post_reset")

# Managers probed for the per-manager rollup, in step-loop execution order, each with the
# top-level methods the step loop calls on it — all wrapped, and every one must exist (a
# renamed hook raises at attach rather than moving its time to ``unattributed``). Each maps
# to the attribute name on the env (``<name>_manager``); missing/None managers are skipped
# (e.g. command/curriculum/metric are optional).
#
# The action manager writes targets to the sim in ``apply_actions`` (called once per
# decimation substep), which is where its real per-step cost lives. The motion manager's
# ``reset`` does not refill the reset envs' reference window: ``fill_data`` does, from
# ``compute`` every step and, on a reset, from inside the RSI event's init-state request
# (``data_start_from="current_time"``) or on leaving ``MotionManager.resetting`` — wrapped so
# the fill is charged to ``motion`` rather than to the event or the ``reset`` row it runs in.
# The recorder's ``compute`` raises by design; the env drives it through its ``_on_*`` hooks.
_MANAGER_METHODS: dict[str, tuple[str, ...]] = {
    "motion": (*_STEP_AND_RESET, "fill_data"),
    "action": (*_STEP_AND_RESET, "apply_actions"),
    "event": _STEP_AND_RESET,
    "metric": _STEP_AND_RESET,
    "termination": _STEP_AND_RESET,
    "reward": _STEP_AND_RESET,
    "curriculum": _STEP_AND_RESET,
    "command": (*_STEP_AND_RESET, "pre_physics_advance", "post_physics_refresh"),
    "observation": (*_STEP_AND_RESET, "post_physics_refresh"),
    "recorder": (
        "reset",
        "post_reset",
        "_on_step_started",
        "_on_step_finished",
        "_on_reset_started",
        "_on_reset_finished",
    ),
}

# Step-loop work no manager owns, as ``(row, owner attribute on the env or None for the env
# itself, method)``; each method keeps its own record, and the rollup sums a row's records by
# their own time. ``reset`` is therefore the env's own reset work around the managers'
# resets (``scene.reset``, solver bookkeeping, buffer zeroing, index handling), not the resets.
_LOOP_METHODS = (
    ("physics", "scene", "step"),
    ("reset", None, "_reset_idx"),
    ("reset", None, "_dispatch_post_reset"),
)

# One rollup row: ``(row, own_s, terms_s, overhead_s)``, the split ``None`` where there is none.
_RollupRow = tuple[str, float, float | None, float | None]


@dataclass
class TermTiming:
    """Running timing accumulator for a single instrumented callable."""

    manager: str
    label: str
    kind: str  # "term" | "manager" | "loop"
    calls: int = 0
    total_s: float = 0.0
    # Wall time minus that of the instrumented calls nested inside this one — the rollup's
    # number. RLStepProfiler adds it only for calls inside a ``step()`` window.
    self_s: float = 0.0
    sumsq_s: float = 0.0
    max_s: float = 0.0

    def add(self, dt: float, self_dt: float | None = None) -> None:
        self.calls += 1
        self.total_s += dt
        self.self_s += dt if self_dt is None else self_dt
        self.sumsq_s += dt * dt
        if dt > self.max_s:
            self.max_s = dt

    @property
    def mean_s(self) -> float:
        return self.total_s / self.calls if self.calls else 0.0

    @property
    def std_s(self) -> float:
        if self.calls < 2:
            return 0.0
        var = self.sumsq_s / self.calls - self.mean_s**2
        return var**0.5 if var > 0.0 else 0.0


class RLStepProfiler:
    """Attribute RL-step wall time to individual manager terms.

    Typical usage::

        profiler = RLStepProfiler(env)
        profiler.attach()
        for _ in range(warmup):
            env.step(sample_action())
        profiler.reset()            # discard warmup, keep instrumentation
        for _ in range(steps):
            with profiler.step():   # times the whole env.step()
                env.step(sample_action())
        print(profiler.report())
        profiler.detach()

    Or as a context manager that attaches/detaches automatically::

        with RLStepProfiler(env) as profiler:
            ...
    """

    def __init__(self, env: EnvBase, *, sync: bool = True):
        self._env = env
        # Only sync when there's actually a CUDA device to wait on.
        self._synchronize = resolve_sync(sync)
        self._stats: dict[tuple[str, str, str], TermTiming] = {}
        self._patches: list[Patch] = []
        # One entry per instrumented call in progress: the wall time of the instrumented calls
        # nested inside it so far, subtracted from its own to give its self time.
        self._nested_s: list[float] = []
        self._step = TermTiming(manager="", label="step", kind="step")
        # Whether a ``step()`` window is open. Own time accrues only inside one, so the rollup
        # partitions the measured step; instrumented calls outside (an ``env.reset()`` between
        # timed steps) still reach the per-term table, and their wall time is ``_outside_s``.
        self._in_step = False
        self._outside_s = 0.0
        self._attached = False

    @property
    def _sync(self) -> bool:
        """Whether CUDA syncing is active (reported in the breakdown)."""
        return self._synchronize is not None

    # -- (de)instrumentation ------------------------------------------------

    def _iter_managers(self) -> Iterator[tuple[str, Any]]:
        for mgr_name in _MANAGER_METHODS:
            mgr = getattr(self._env, f"{mgr_name}_manager", None)
            if mgr is not None:
                yield mgr_name, mgr

    def _step_loop_hooks(self) -> list[tuple[tuple[str, str, str], Any, str]]:
        """``(record key, owner, method)`` for every manager hook and step-loop method to wrap.

        Resolved in full before anything is patched, so a missing hook raises with the env untouched.
        """
        hooks: list[tuple[tuple[str, str, str], Any, str]] = []
        for mgr_name, mgr in self._iter_managers():
            for method in _MANAGER_METHODS[mgr_name]:
                if not callable(getattr(mgr, method, None)):
                    raise AttributeError(
                        f"RLStepProfiler: {type(mgr).__name__} (env.{mgr_name}_manager) has no method {method!r}, "
                        f"one of the hooks the step loop calls on it ({', '.join(_MANAGER_METHODS[mgr_name])}). "
                        f"If the step loop renamed or dropped it, update _MANAGER_METHODS['{mgr_name}'] in "
                        "eden/utils/profiling/step_profiler.py to match; skipping it instead would move its time "
                        "into 'unattributed' without a word."
                    )
                suffix = "" if method == "compute" else f".{method}"
                hooks.append(((mgr_name, f"<{mgr_name}{suffix}>", "manager"), mgr, method))
        for row, attr, method in _LOOP_METHODS:
            owner = self._env if attr is None else getattr(self._env, attr)
            label = method if attr is None else f"{attr}.{method}"
            hooks.append(((row, label, "loop"), owner, method))
        return hooks

    def _wrap(self, key: tuple[str, str, str], owner: Any, method: str) -> None:
        """Replace ``owner.method`` with a timing wrapper, remembering the original."""
        rec = self._stats.setdefault(key, TermTiming(manager=key[0], label=key[1], kind=key[2]))
        orig = getattr(owner, method)
        nested_s = self._nested_s

        def on_dt(dt: float) -> None:
            if self._in_step:
                rec.add(dt, dt - nested_s[-1])
            else:
                rec.add(dt, 0.0)
                if len(nested_s) == 1:
                    self._outside_s += dt
            if len(nested_s) > 1:
                nested_s[-2] += dt

        def wrapper(*args, **kwargs):
            nested_s.append(0.0)
            try:
                return time_call(lambda: orig(*args, **kwargs), on_dt, self._synchronize)
            finally:
                nested_s.pop()

        self._patches.append(Patch(owner, method, wrapper))

    def attach(self) -> RLStepProfiler:
        if self._attached:
            return self
        hooks = self._step_loop_hooks()
        # discover_timeable dedups a term shared across groups so it is wrapped once, not
        # double-counted, and drops any term that can't carry the timing override.
        timeable, skipped = discover_timeable(self._env, include_event=True)
        warn_untimed("RLStepProfiler", skipped)
        # A run of RESET terms that fires as one ``compute_batch`` never calls its terms' own ``compute``:
        # time the batch as one row, ``reset.<a>+<b>+...``, in their place.
        event_manager = getattr(self._env, "event_manager", None)
        batches = list(event_manager.iter_reset_batches()) if event_manager is not None else []
        batched = {id(term) for batch in batches for term in batch.terms}
        for mgr_name, label, term in timeable:
            if id(term) not in batched:
                self._wrap((mgr_name, label, "term"), term, "compute")
        for batch in batches:
            self._wrap(("event", f"reset.{'+'.join(batch.names)}", "term"), batch, "compute")
        for key, owner, method in hooks:
            self._wrap(key, owner, method)
        self._attached = True
        return self

    def detach(self) -> None:
        for patch in reversed(self._patches):
            patch.restore()
        self._patches.clear()
        self._attached = False

    # -- measurement --------------------------------------------------------

    def reset(self) -> None:
        """Zero all accumulators (e.g. to discard warmup) without detaching."""
        for rec in self._stats.values():
            rec.calls = 0
            rec.total_s = 0.0
            rec.self_s = 0.0
            rec.sumsq_s = 0.0
            rec.max_s = 0.0
        self._step = TermTiming(manager="", label="step", kind="step")
        self._outside_s = 0.0

    class _StepTimer:
        def __init__(self, profiler: RLStepProfiler):
            self._p = profiler

        def __enter__(self):
            if self._p._in_step:
                raise RuntimeError(
                    "RLStepProfiler.step() is already open: step windows cannot nest. Time each env.step() in its "
                    "own `with profiler.step():` — an inner window would close the outer one early, and every call "
                    "after it would read as made outside a step."
                )
            if self._p._synchronize is not None:
                self._p._synchronize()
            self._t0 = perf_counter()
            self._p._in_step = True
            return self

        def __exit__(self, *exc):
            self._p._in_step = False
            if self._p._synchronize is not None:
                self._p._synchronize()
            self._p._step.add(perf_counter() - self._t0)
            return False

    def step(self) -> _StepTimer:
        """Context manager timing one whole ``env.step()`` call."""
        return self._StepTimer(self)

    # -- reporting ----------------------------------------------------------

    @property
    def num_steps(self) -> int:
        return self._step.calls

    def _term_records(self) -> list[TermTiming]:
        return [r for r in self._stats.values() if r.kind == "term"]

    def _unattributed_s(self) -> float:
        """Step time outside every instrumented call — the own times partition the rest."""
        own_s = self._step.total_s - sum(r.self_s for r in self._stats.values())
        # Own time accrues only inside a step window, so this is >= 0 up to float rounding. Zero
        # only the rounding: a real overcount must still show, as a negative row.
        return 0.0 if -1e-9 < own_s < 0.0 else own_s

    def _rollup(self) -> tuple[list[_RollupRow], list[_RollupRow]]:
        """The rollup's rows, ``(row, own_s, terms_s, overhead_s)``: together they sum to the measured step.

        Returns the manager rows, then the rest: the step-loop rows and ``unattributed``. A manager's
        row is the own time of its terms (``terms_s``) and of its own methods (``overhead_s``); a
        manager with no timed term (the motion manager's run inside ``fill_data``), the step-loop rows
        and ``unattributed`` have no split (``None``).
        """
        terms_s: dict[str, float] = {}
        own_s: dict[str, float] = {}
        for r in self._stats.values():
            if r.kind == "term":
                terms_s[r.manager] = terms_s.get(r.manager, 0.0) + r.self_s
            own_s[r.manager] = own_s.get(r.manager, 0.0) + r.self_s
        manager_rows: list[_RollupRow] = []
        for mgr_name in _MANAGER_METHODS:
            if mgr_name in own_s:
                t = terms_s.get(mgr_name)
                manager_rows.append((mgr_name, own_s[mgr_name], t, None if t is None else own_s[mgr_name] - t))
        loop_rows: list[_RollupRow] = [
            (row, own_s.get(row, 0.0), None, None) for row in dict.fromkeys(row for row, _, _ in _LOOP_METHODS)
        ]
        return manager_rows, [*loop_rows, ("unattributed", self._unattributed_s(), None, None)]

    def _records_by_owner(self, kind: str) -> dict[str, dict[str, TermTiming]]:
        """The ``kind`` records (``"manager"`` or ``"loop"``), by manager or step-loop row, then label."""
        out: dict[str, dict[str, TermTiming]] = {}
        for r in self._stats.values():
            if r.kind == kind:
                out.setdefault(r.manager, {})[r.label] = r
        return out

    def as_dict(self) -> dict[str, Any]:
        """Machine-readable summary (JSON-serializable).

        ``terms``, ``managers`` (by manager) and ``loop`` (by step-loop row: ``physics``,
        ``reset``) hold one record per instrumented callable: every call, inclusive of the calls
        nested inside it (``total_ms_per_step``, ``pct_step``), plus its own time inside step
        windows (``self_ms_per_step``). ``rollup`` is the report's rollup table: own time per
        row, summing to ``step_ms``. ``outside_step_ms`` is the wall time of instrumented calls
        made outside :meth:`step`, which only the records count.
        """
        n = max(self.num_steps, 1)
        step_total = self._step.total_s

        def pack(r: TermTiming) -> dict[str, Any]:
            return {
                "manager": r.manager,
                "label": r.label,
                "kind": r.kind,
                "calls_per_step": r.calls / n,
                "mean_ms": r.mean_s * 1e3,
                "total_ms_per_step": r.total_s / n * 1e3,
                "self_ms_per_step": r.self_s / n * 1e3,
                "std_ms": r.std_s * 1e3,
                "max_ms": r.max_s * 1e3,
                "pct_step": (r.total_s / step_total * 100.0) if step_total > 0 else 0.0,
            }

        def pack_by_owner(kind: str) -> dict[str, dict[str, dict[str, Any]]]:
            return {
                owner: {label: pack(r) for label, r in entries.items()}
                for owner, entries in self._records_by_owner(kind).items()
            }

        terms = sorted(self._term_records(), key=lambda r: r.total_s, reverse=True)
        manager_rows, other_rows = self._rollup()
        return {
            "num_steps": self.num_steps,
            "sync": self._sync,
            "step_ms": self._step.mean_s * 1e3,
            "steps_per_s": (n / step_total) if step_total > 0 else 0.0,
            "terms": [pack(r) for r in terms],
            "managers": pack_by_owner("manager"),
            "loop": pack_by_owner("loop"),
            "rollup": {
                row: {
                    "ms_per_step": own_s / n * 1e3,
                    "terms_ms_per_step": None if t is None else t / n * 1e3,
                    "overhead_ms_per_step": None if o is None else o / n * 1e3,
                    "pct_step": (own_s / step_total * 100.0) if step_total > 0 else 0.0,
                }
                for row, own_s, t, o in [*manager_rows, *other_rows]
            },
            "outside_step_ms": self._outside_s * 1e3,
        }

    def report(self, *, top: int | None = None, sort_by: str = "total") -> str:
        """Build human-readable breakdown tables.

        Parameters
        ----------
        top : int, optional
            If given, only show the ``top`` slowest terms.
        sort_by : str, optional
            Term ordering, either ``"total"`` (default) or ``"mean"``.

        Returns
        -------
        str
            The formatted breakdown tables.
        """
        n = max(self.num_steps, 1)
        step_total = self._step.total_s
        step_ms = self._step.mean_s * 1e3

        key = (lambda r: r.mean_s) if sort_by == "mean" else (lambda r: r.total_s)
        terms = sorted(self._term_records(), key=key, reverse=True)
        if top is not None:
            terms = terms[:top]

        def pct(total_s: float) -> str:
            return f"{(total_s / step_total * 100.0):.1f}" if step_total > 0 else "-"

        term_tbl = PrettyTable()
        # Calls outside a step window still count here, over the timed steps: say so on the table itself.
        outside = ", incl. calls outside step()" if self._outside_s > 0.0 else ""
        term_tbl.title = f"Per-Term Timing  ({self.num_steps} steps, sync={self._sync}{outside})"
        term_tbl.field_names = ["Manager", "Term", "Calls/step", "Mean (ms)", "Total/step (ms)", "% step"]
        term_tbl.align["Term"] = "l"
        term_tbl.align["Manager"] = "l"
        for r in terms:
            term_tbl.add_row(
                [
                    r.manager,
                    r.label,
                    f"{r.calls / n:.2f}",
                    f"{r.mean_s * 1e3:.4f}",
                    f"{r.total_s / n * 1e3:.4f}",
                    pct(r.total_s),
                ]
            )

        # Per-manager rollup, by own time so the rows sum to the step: each manager's terms and its
        # own methods around them (dispatch/bookkeeping), then the step-loop rows no manager owns,
        # and whatever no instrumented call covers.
        mgr_tbl = PrettyTable()
        mgr_tbl.title = "Per-Manager Rollup  (own time; rows sum to the step)"
        mgr_tbl.field_names = ["Manager", "Own/step (ms)", "Terms (ms)", "Overhead (ms)", "% step"]
        mgr_tbl.align["Manager"] = "l"

        def add_rollup_row(row: str, own_s: float, terms_s: float | None, overhead_s: float | None) -> None:
            split = ["-", "-"] if terms_s is None else [f"{terms_s / n * 1e3:.4f}", f"{overhead_s / n * 1e3:.4f}"]
            mgr_tbl.add_row([row, f"{own_s / n * 1e3:.4f}", *split, pct(own_s)])

        manager_rows, other_rows = self._rollup()
        for rollup_row in manager_rows:
            add_rollup_row(*rollup_row)
        mgr_tbl.add_row(["─" * 7, "─" * 13, "─" * 10, "─" * 13, "─" * 6])
        for rollup_row in other_rows:
            add_rollup_row(*rollup_row)

        steps_per_s = (n / step_total) if step_total > 0 else 0.0
        header = (
            f"RL step profile — {self.num_steps} steps, {self._env.num_envs} envs, "
            f"sync={self._sync}\n"
            f"  step: {step_ms:.3f} ms/step  ({steps_per_s:,.0f} steps/s, "
            f"{steps_per_s * self._env.num_envs:,.0f} env-steps/s)\n"
        )
        if self._outside_s > 0.0:
            header += (
                f"  {self._outside_s * 1e3:.3f} ms of instrumented calls ran outside profiler.step() "
                "(e.g. an env.reset() between timed steps): counted in the per-term table, not the rollup\n"
            )
        return header + "\n" + mgr_tbl.get_string() + "\n\n" + term_tbl.get_string() + "\n"

    def save_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.as_dict(), f, indent=2)

    # -- context manager ----------------------------------------------------

    def __enter__(self) -> RLStepProfiler:
        return self.attach()

    def __exit__(self, *exc) -> bool:
        self.detach()
        return False
