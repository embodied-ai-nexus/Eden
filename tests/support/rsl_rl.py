"""rsl_rl helpers for tests that must run on every rsl-rl-lib 5.x the pin admits."""

from __future__ import annotations

import copy
from collections.abc import Callable


def resolve_class(cfg: dict) -> tuple[Callable, dict]:
    """``(class, constructor kwargs)`` of ``cfg["class_name"]``, leaving ``cfg`` as it is.

    ``rsl_rl.utils.resolve_class`` does the same from rsl-rl-lib 5.5.0 only; the models and algorithms these tests
    build (SimBa, MuonPPO, the distributions) run on every 5.x, so the tests resolve through ``resolve_callable``,
    which all of them ship.
    """
    from rsl_rl.utils import resolve_callable

    kwargs = copy.deepcopy(cfg)
    return resolve_callable(kwargs.pop("class_name")), kwargs
