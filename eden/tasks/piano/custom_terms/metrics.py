"""How well the song is being played: RoboPianist's precision, recall and F1 over the keys, per control step.

Each is the episode mean of a per-step score, which is how RoboPianist reports them (scikit-learn's scores on each
step's 88-key vector, averaged over the episode).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from eden.managers import METRIC_TERM_REGISTRY

if TYPE_CHECKING:
    from eden.envs.base import EnvBase


def _counts(env: EnvBase, command_name: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    song = env.command_manager.get_term(command_name)
    hit = (song.key_down & song.goal).sum(dim=-1).float()
    return hit, song.key_down.sum(dim=-1).float(), song.goal.sum(dim=-1).float()


def _ratio(hit: torch.Tensor, total: torch.Tensor) -> torch.Tensor:
    # RoboPianist's convention: a step with nothing to count scores 1, so silence played as silence is perfect.
    return torch.where(total > 0, hit / total.clamp(min=1.0), torch.ones_like(hit))


@METRIC_TERM_REGISTRY.register()
def key_press_precision(env: EnvBase, *, command_name: str = "song") -> torch.Tensor:
    """Of the keys that are down, the fraction that should be."""
    hit, down, _ = _counts(env, command_name)
    return _ratio(hit, down)


@METRIC_TERM_REGISTRY.register()
def key_press_recall(env: EnvBase, *, command_name: str = "song") -> torch.Tensor:
    """Of the keys that should be down, the fraction that are."""
    hit, _, due = _counts(env, command_name)
    return _ratio(hit, due)


@METRIC_TERM_REGISTRY.register()
def key_press_f1(env: EnvBase, *, command_name: str = "song") -> torch.Tensor:
    """Harmonic mean of precision and recall: RoboPianist's headline number."""
    hit, down, due = _counts(env, command_name)
    return _ratio(2.0 * hit, down + due)


__all__ = ["key_press_f1", "key_press_precision", "key_press_recall"]
