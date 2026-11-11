"""Deciding whether — and how — to smoke-run an example from the catalog.

Shared by the recipe test harness (``tests/utils/test_example_smoke.py``) and the
``eden examples smoke`` CLI. An example is smoke-runnable only if it needs no
hardware/device, its optional deps are installed, the machine has the backend it
needs, it is instrumented with :func:`eden.is_smoke`, and its canonical ``run``
needs no arguments. :func:`skip_reason` returns why an example is *not* runnable
(or ``None`` if it is), so callers can report every skip rather than silently drop it;
a missing optional dep also names its install command.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

from eden.utils.examples_catalog import OPTIONAL_DEPS, REPO_ROOT, ExampleSpec


def has_gpu() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def _dep_installed(dep: str) -> bool:
    info = OPTIONAL_DEPS.get(dep)
    if info is None:
        return False  # unknown dep — conservatively treat as unavailable (skip, don't fail)
    try:
        return importlib.util.find_spec(info.module) is not None
    except (ImportError, ValueError):
        return False


def _run_needs_args(spec: ExampleSpec) -> bool:
    """True if the canonical ``run`` passes a positional/value argument (not just -flags)."""
    toks = spec.run.split()
    fname = spec.path.rsplit("/", 1)[-1]
    tail = toks[next((k for k, t in enumerate(toks) if t.endswith(fname)), len(toks)) + 1 :]
    return any(not t.startswith("-") for t in tail)


def skip_reason(spec: ExampleSpec, *, has_gpu: bool) -> str | None:
    """Why this example can't be auto-smoke-run here, or None if it can."""
    if spec.hardware == "real-robot":
        return "needs real robot hardware"
    if any(c.startswith("teleop.") for c in spec.capabilities):
        return "needs a teleoperation device"
    for dep in spec.optional_deps:
        if not _dep_installed(dep):
            info = OPTIONAL_DEPS.get(dep)
            hint = f" ({info.install})" if info else ""
            return f"optional dependency '{dep}' not installed{hint}"
    if spec.hardware in ("gpu", "gpu-fp64") and not has_gpu:
        return "needs a CUDA GPU"
    if "is_smoke" not in (REPO_ROOT / spec.path).read_text():
        return "not smoke-instrumented (no en.is_smoke() step cap)"
    if _run_needs_args(spec):
        return "canonical run requires arguments"
    return None


def smoke_run(
    spec: ExampleSpec, *, timeout: int = 900, capture: bool = True, cwd: Path | None = None
) -> subprocess.CompletedProcess:
    """Run the example headless in smoke mode (few steps) as a subprocess.

    Parameters
    ----------
    spec : ExampleSpec
        The catalog entry to run.
    timeout : int
        Seconds before the run is killed.
    capture : bool
        Capture stdout/stderr into the result instead of passing them through.
    cwd : Path | None
        The directory the example runs in, and so where it writes its outputs (``logs/``, ``renders/``, images,
        HTML recordings), which examples put under the working directory. ``None`` is the caller's.
    """
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / spec.path)],
        cwd=cwd,
        env={**os.environ, "EDEN_SMOKE": "1"},
        capture_output=capture,
        text=True,
        timeout=timeout,
    )
