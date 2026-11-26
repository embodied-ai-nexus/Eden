"""rsl_rl helpers for tests that must run on every rsl-rl-lib release the pin admits."""

from __future__ import annotations

import copy
from collections.abc import Callable


def runner_unpacks_stats() -> bool:
    """Whether the installed runner takes ``(loss_dict, stats_dict)`` from ``update()``, by the version number.

    ``eden.envs.wrappers.rsl_rl_compat`` decides the same question from ``Logger.log``'s signature, never from the
    version, so a test that compares the two checks the probe instead of restating it.
    """
    from eden.options.learning.rsl_rl.translate import detect_rsl_rl_version

    return detect_rsl_rl_version() >= (6,)


def resolve_class(cfg: dict) -> tuple[Callable, dict]:
    """``(class, constructor kwargs)`` of ``cfg["class_name"]``, leaving ``cfg`` as it is.

    ``rsl_rl.utils.resolve_class`` does the same from rsl-rl-lib 5.5.0 only; the models and algorithms these tests
    build (SimBa, MuonPPO, the distributions) run on every 5.x, so the tests resolve through ``resolve_callable``,
    which all of them ship.
    """
    from rsl_rl.utils import resolve_callable

    kwargs = copy.deepcopy(cfg)
    return resolve_callable(kwargs.pop("class_name")), kwargs
