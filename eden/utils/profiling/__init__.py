"""Per-term cost profilers for Eden's step loop.

Two front-ends over one shared instrumentation core
(:mod:`eden.utils.profiling.core`), for two workflows:

* :class:`RLStepProfiler` — attach to a *live, caller-driven* loop (training or
  rollout), time every term and manager method, and get a full per-manager
  rollup. Driven by ``examples/recipes/terms/rl_step_profile.py``.
* :func:`run_smoke_profile` / :func:`profile_terms` — *one-shot*: build a task,
  step a short window, and rank the per-step terms by cost. The engine behind
  ``eden smoke --task <name> --profile``; use it to profile a term you just wrote.

Both discover terms and time each call identically (see :func:`~eden.utils.profiling.core.iter_terms`
and :func:`~eden.utils.profiling.core.time_call`); they differ only in what
they measure and how they report. :class:`RLStepProfiler` builds its own wrapper
from :func:`~eden.utils.profiling.core.time_call` and :class:`~eden.utils.profiling.core.Patch`,
to subtract nested calls from their caller.
"""

from eden.utils.profiling.core import iter_terms
from eden.utils.profiling.step_profiler import RLStepProfiler, TermTiming
from eden.utils.profiling.term_profiler import ProfileReport, TermCost, profile_terms, run_smoke_profile

__all__ = [
    "RLStepProfiler",
    "TermTiming",
    "ProfileReport",
    "TermCost",
    "profile_terms",
    "run_smoke_profile",
    "iter_terms",
]
