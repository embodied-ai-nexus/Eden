"""``eden.options.learning.clip_actions``: the action bound every runner's options validate at construction."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from eden.learning.wrappers.rsl_rl_contrack import RslRlContrackRunnerOptions
from eden.options.learning.rl_games import RlGamesPpoRunnerOptions
from eden.options.learning.rsl_rl.rl import RslRlBaseRunnerOptions

RUNNER_OPTIONS = [RslRlBaseRunnerOptions, RlGamesPpoRunnerOptions, RslRlContrackRunnerOptions]


@pytest.mark.parametrize("options_cls", RUNNER_OPTIONS)
@pytest.mark.parametrize("bound", [0.0, -1.0, float("inf"), float("nan")])
def test_a_bound_that_clamps_nothing_sensibly_is_refused_at_construction(options_cls, bound):
    """Zero clamps every action to zero, a negative bound inverts the interval, and inf or NaN bounds nothing."""
    with pytest.raises(ValidationError, match=r"clip_actions=.* must be a finite positive bound, or None"):
        options_cls(clip_actions=bound)


@pytest.mark.parametrize("options_cls", RUNNER_OPTIONS)
def test_a_finite_positive_bound_is_kept(options_cls):
    assert options_cls(clip_actions=4.8).clip_actions == 4.8


@pytest.mark.parametrize("options_cls", [RslRlBaseRunnerOptions, RlGamesPpoRunnerOptions])
def test_none_leaves_actions_unclamped(options_cls):
    assert options_cls(clip_actions=None).clip_actions is None
