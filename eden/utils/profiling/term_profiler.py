"""On-demand, one-shot per-term cost profiler for a single task.

Why this exists
---------------
A term that is *correct but slow* is invisible. Nothing in a normal run tells the
author that the observation term they just wrote allocates a ``torch.tensor(...)``
constant every step and now dominates rollout collection — the only symptom is a
training run that feels slow. Measured on the Go1 backflip benchmark, two runs of
the *same* task on the *same* GPU differed 35x in throughput purely because of
per-call allocations inside a task's own terms (see ``docs/guides/terms.md``).

This profiler makes that cost *investigable on demand*. ``eden smoke --task <name>
--profile`` builds the task, steps it a bounded window, attributes wall time to
each term the per-step managers run, and writes a ranked, machine-readable
``term_cost.json`` — so an author (or a coding agent that just wrote the term) can
see which term is the bottleneck and what fraction of the step it costs.

Relationship to :class:`~eden.utils.profiling.step_profiler.RLStepProfiler`
---------------------------------------------------------------------------
That profiler attaches to a *caller-driven* loop (training / rollout) and also
rolls up per-manager overhead; this one owns a short self-driven window and emits
a compact ranking with no manager rollup. They share the term-discovery and
timing mechanism in :mod:`eden.utils.profiling.core`, so both stay faithful to
the same manager internals and the same both-sides CUDA sync.

Purely additive, zero overhead when off
---------------------------------------
Nothing here runs unless a caller asks for it. The profiler wraps ``term.compute``
from the *outside* for the measurement window only, times it with a CUDA sync on
both sides so the number reflects the term's own GPU work, then restores every
original method. A run without ``--profile`` creates no profiler object and wraps
no method anywhere in the framework.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

from eden.utils.profiling.core import discover_timeable, resolve_sync, time_call, time_method, warn_untimed

if TYPE_CHECKING:
    from eden.envs.base import EnvBase
    from eden.utils.configs import EdenConfig


@dataclass
class _Timing:
    """Running wall-time accumulator for one instrumented ``compute``."""

    total_s: float = 0.0
    calls: int = 0

    def add(self, dt: float) -> None:
        self.total_s += dt
        self.calls += 1


@dataclass
class TermCost:
    """One term's measured cost, normalized per env step."""

    manager: str
    term: str
    us_per_step: float
    calls_per_step: float
    pct_step: float = 0.0


@dataclass
class ProfileReport:
    """Ranked per-term cost for a profiled window. Agent- and human-readable."""

    num_steps: int
    step_us: float
    sync: bool
    rows: list[TermCost] = field(default_factory=list)

    #: One-line pointer to the fix, carried in both the JSON and the console footer.
    HINT = (
        "Hot-path rules: resolve indices and allocate constant tensors in build(), "
        "not compute(); avoid .item()/.cpu()/bool(tensor). See docs/guides/terms.md."
    )

    #: Measurement caveat, carried alongside HINT so a consumer does not misread the absolutes.
    NOTE = (
        "With sync=true, us/step and step_us are wall time with CUDA synced on both sides, so "
        "they run higher than an unsynced run; the pct_step shares are the faithful signal."
    )

    def as_dict(self) -> dict:
        """Machine-readable summary (JSON-serializable), terms ranked costliest-first."""
        return {
            "num_steps": self.num_steps,
            "step_us": round(self.step_us, 1),
            "sync": self.sync,
            "terms": [
                {
                    "manager": r.manager,
                    "term": r.term,
                    "us_per_step": round(r.us_per_step, 1),
                    "pct_step": round(r.pct_step, 1),
                    "calls_per_step": round(r.calls_per_step, 2),
                }
                for r in self.rows
            ],
            "hint": self.HINT,
            "note": self.NOTE,
        }

    def to_json(self, path: str) -> None:
        """Write :meth:`as_dict` to ``path`` (creating parent dirs)."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as fh:
            json.dump(self.as_dict(), fh, indent=2)

    def render(self, top: int | None = 15) -> str:
        """Human-readable table, costliest term first (``top`` rows, then a rollup)."""
        term_total = sum(r.us_per_step for r in self.rows)
        lines = [
            f"Per-term compute cost — {self.num_steps} steps, "
            f"{self.step_us:.0f} us/env-step ({term_total:.0f} us/step in terms).",
            f"  {'us/step':>10}  {'%step':>6}  {'calls':>5}  term",
        ]
        shown = self.rows if top is None else self.rows[:top]
        for r in shown:
            lines.append(
                f"  {r.us_per_step:>10.0f}  {r.pct_step:>5.1f}%  {r.calls_per_step:>5.1f}  {r.manager}:{r.term}"
            )
        if top is not None and len(self.rows) > top:
            rest = sum(r.us_per_step for r in self.rows[top:])
            lines.append(f"  {rest:>10.0f}  {'':>6}  {'':>5}  ({len(self.rows) - top} more terms)")
        lines.append("  " + self.HINT)
        if self.sync:
            lines.append("  " + self.NOTE)
        return "\n".join(lines)


def profile_terms(
    env: "EnvBase",
    *,
    warmup: int = 10,
    measure: int = 50,
    step: Callable[[], object] | None = None,
    sync: bool | None = None,
) -> ProfileReport:
    """Attribute wall time to each per-step term over a bounded window.

    Runs ``warmup`` steps uninstrumented (kernel compilation and lazy allocation
    land there), then wraps every term's ``compute``, runs ``measure`` steps, and
    restores the originals. The env is left stepped, not reset — the caller owns it.

    Event terms are excluded (they fire on reset / interval, not per step) and so
    are action terms (their dominant cost is in ``apply_actions``, not the timed
    ``compute``). The reported ``step_us`` is the whole ``env.step()`` measured
    under the same both-sides sync, so it runs high in absolute terms but the
    per-term ``pct_step`` shares are faithful (see
    :class:`~eden.utils.profiling.step_profiler.RLStepProfiler` for the same caveat).

    Parameters
    ----------
    env : EnvBase
        A built, reset env.
    warmup : int
        Uninstrumented steps before measuring.
    measure : int
        Instrumented steps; per-term cost is normalized by this (a term computed
        more than once per step accumulates all its calls into that step's cost).
    step : callable, optional
        Zero-arg step to call each iteration. Defaults to ``lambda: env.step(None)``.
    sync : bool, optional
        Synchronize CUDA around each timed call. Defaults to whether CUDA is present.

    Returns
    -------
    ProfileReport
        Ranked per-term cost plus the measured env-step time.
    """
    synchronize = resolve_sync(sync)
    step = step or (lambda: env.step(None))

    for _ in range(warmup):
        step()

    timeable, skipped = discover_timeable(env, include_event=False, include_action=False)
    warn_untimed("profile_terms", skipped)

    records: dict[tuple[str, str], _Timing] = {}
    patches = []  # list[core.Patch]
    for manager, label, term in timeable:
        rec = records.setdefault((manager, label), _Timing())
        patches.append(time_method(term, "compute", rec.add, synchronize))

    step_time = _Timing()
    try:
        for _ in range(measure):
            time_call(step, step_time.add, synchronize)
    finally:
        for patch in patches:
            patch.restore()

    steps = max(measure, 1)
    step_us = step_time.total_s / steps * 1e6
    rows = [
        TermCost(
            manager=manager,
            term=label,
            us_per_step=rec.total_s / steps * 1e6,
            calls_per_step=rec.calls / steps,
        )
        for (manager, label), rec in records.items()
    ]
    rows.sort(key=lambda r: r.us_per_step, reverse=True)
    for r in rows:
        r.pct_step = (r.us_per_step / step_us * 100.0) if step_us > 0 else 0.0
    return ProfileReport(num_steps=measure, step_us=step_us, sync=synchronize is not None, rows=rows)


def run_smoke_profile(
    task: "str | EdenConfig",
    *,
    modifier_kwargs: dict[str, Any] | None = None,
    num_envs: int = 2,
    warmup: int = 10,
    measure: int = 50,
    eval_mode: bool = False,
    out_dir: str | None = None,
) -> tuple[ProfileReport, str]:
    """Build a task and profile its per-term cost. The smoke ``--profile`` entry point.

    Parameters
    ----------
    task : str | EdenConfig
        Registered task name or a constructed config.
    modifier_kwargs : dict[str, Any], optional
        TaskMod overrides for a task *name*, forwarded to
        :func:`eden.utils.smoke.build_and_step` (and from there to
        ``TASK_REGISTRY.build``); rejected alongside a constructed config.
    num_envs : int
        Parallel envs to build.
    warmup : int
        Uninstrumented warmup steps.
    measure : int
        Instrumented steps.
    eval_mode : bool
        Build in eval mode (training/randomization off).
    out_dir : str, optional
        Directory for ``term_cost.json``. Defaults to ``eden.get_log_dir()``.

    Returns
    -------
    tuple of (ProfileReport, str)
        The report and the path the JSON was written to.
    """
    import eden as en

    from eden.utils.smoke import build_and_step

    env = build_and_step(task, modifier_kwargs=modifier_kwargs, num_envs=num_envs, num_steps=0, eval_mode=eval_mode)
    report = profile_terms(env, warmup=warmup, measure=measure)
    path = os.path.join(out_dir or en.get_log_dir(), "term_cost.json")
    report.to_json(path)
    return report, path
