"""Lightweight build-and-step smoke harness for any registered task.

``build_and_step("<task>")`` (or ``eden smoke --task <name>``) constructs a task
config by name, builds its Genesis env, and steps it a few times with no
action — a one-call end-to-end sanity check for refactors and cheap coverage.
Task variants take the same route as the CLI's mod flags:
``build_and_step("move_forward", modifier_kwargs={"robot": "go1"})`` is ``eden smoke --task
move_forward --robot go1``.

It encapsulates the steps that are otherwise easy to get wrong:

* initialize Genesis first (done here if the caller hasn't),
* pick ``MimicRLEnvBase`` vs ``RLEnvBase`` from the config type,
* ``build()`` then ``reset()`` before stepping,
* ``step(None)`` advances the sim with no action tensor.
"""

import os
from typing import TYPE_CHECKING, Any, Union

if TYPE_CHECKING:
    from eden.envs.base import RLEnvBase
    from eden.utils.configs import EdenConfig


def is_smoke() -> bool:
    """True in "smoke" mode — the few-step, headless run used for CI/sanity checks.

    Controlled by ``EDEN_SMOKE`` (the canonical knob, set to ``1`` by ``eden examples smoke``
    and the recipe test harness); ``PYTEST_VERSION`` (set by pytest) is a fallback so a bare
    test run is treated as smoke. Examples branch their step horizon on it:
    ``horizon = 5 if en.is_smoke() else 1000``.

    ``EDEN_SMOKE`` is parsed as a boolean, not by presence: ``EDEN_SMOKE=0`` / ``false`` / ``no``
    is a real off-switch. An explicitly-set value is authoritative — it wins over
    ``PYTEST_VERSION``, so a test can opt back out of smoke mode with ``EDEN_SMOKE=0``. An unset
    or empty value falls through to the ``PYTEST_VERSION`` presence check. (Falsey vocabulary
    matches ``eden.extensions.policies.onnx._env_bool``.)
    """
    eden_smoke = os.environ.get("EDEN_SMOKE", "").strip().lower()
    if eden_smoke:
        return eden_smoke not in ("0", "false", "no")
    return bool(os.environ.get("PYTEST_VERSION"))


def build_and_step(
    task: "Union[str, EdenConfig]",
    *,
    modifier_kwargs: dict[str, Any] | None = None,
    num_envs: int = 2,
    num_steps: int = 2,
    eval_mode: bool = False,
    show_viewer: bool = False,
    backend=None,
) -> "RLEnvBase":
    """Build a task's env and step it ``num_steps`` times with no action.

    One-shot: each call builds a fresh scene and does not destroy it, so it is
    meant for a single sanity check (or a session-scoped test), not a tight
    loop over many tasks in one process.

    Parameters
    ----------
    task : str | EdenConfig
        A registered task name, or an already-constructed config object. A name
        goes through ``TASK_REGISTRY.build`` so the task's TaskMods run — a
        mod-bound task is not buildable without them.
    modifier_kwargs : dict[str, Any], optional
        TaskMod overrides forwarded to ``TASK_REGISTRY.build(task, **modifier_kwargs)``
        — the Python side of ``eden smoke --task <name> --<flag> <value>``, e.g.
        ``{"robot": "go1"}``. Unknown flags raise the registry's ``ValueError``. Only
        a task *name* can take them: a config object is already built, mods included,
        so passing both raises ``TypeError`` rather than silently ignoring the kwargs.
    num_envs : int
        Number of parallel envs to build.
    num_steps : int
        Number of no-action ``step(None)`` calls to run after the initial reset.
    eval_mode : bool
        Build in eval mode (training/randomization off). Defaults to False so the
        full training path is exercised.
    show_viewer : bool
        Open the Genesis viewer.
    backend : genesis backend, optional
        Passed to ``en.init`` only when Genesis isn't initialized yet. ``None``
        leaves ``en.init``'s default; pass ``gs.gpu`` / ``gs.cpu`` to choose.

    Returns
    -------
    RLEnvBase
        The built, stepped env — returned so callers can inspect it. Any error
        in the build/step path propagates; surfacing it is the point.
    """
    import eden as en
    import genesis as gs

    from eden.envs.base import env_class_for_config
    from eden.tasks import TASK_REGISTRY
    from eden.utils.configs import EdenConfig

    if isinstance(task, EdenConfig) and modifier_kwargs:
        raise TypeError(
            f"build_and_step() got modifier_kwargs {sorted(modifier_kwargs)} together with an already-built "
            f"{type(task).__name__}. TaskMods run while a task is built from its name, so either pass the task "
            "name here or apply them yourself with TASK_REGISTRY.build(name, **modifier_kwargs) and pass the result."
        )

    if not getattr(gs, "_initialized", False):
        en.init(**({"backend": backend} if backend is not None else {}))

    cfg = task if isinstance(task, EdenConfig) else TASK_REGISTRY.build(task, **(modifier_kwargs or {}))
    cfg.env_options.num_envs = num_envs

    env = env_class_for_config(cfg).from_config(cfg, show_viewer=show_viewer, eval_mode=eval_mode)
    env.build()
    env.reset()
    for _ in range(num_steps):
        env.step(None)
    return env
