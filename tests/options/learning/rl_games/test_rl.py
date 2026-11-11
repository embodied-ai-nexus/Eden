"""``eden.options.learning.rl_games.rl``: rl_games runner options."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from eden.options.learning.rl_games import RlGamesPpoRunnerOptions


def test_a_negative_seed_is_refused_at_construction():
    """A negative seed is not a "draw one" sentinel: refused, rather than silently drawing the same value every run."""
    with pytest.raises(ValidationError, match="seed must be >= 0"):
        RlGamesPpoRunnerOptions(seed=-1)


@pytest.mark.parametrize("keep", [0, -1])
def test_keeping_fewer_than_one_checkpoint_is_refused_at_construction(keep):
    """Each save would delete the checkpoint it just wrote."""
    with pytest.raises(ValueError, match=rf"keep_last_checkpoints={keep}\).*count >= 1.*or None"):
        RlGamesPpoRunnerOptions(keep_last_checkpoints=keep)


def test_a_run_keeps_only_its_newest_checkpoint_by_default():
    assert RlGamesPpoRunnerOptions().keep_last_checkpoints == 1
