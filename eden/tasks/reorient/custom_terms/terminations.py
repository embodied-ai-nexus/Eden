"""Termination terms for in-hand cube reorientation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from eden.managers import TERMINATION_TERM_REGISTRY

if TYPE_CHECKING:
    from eden.envs.base import RLEnvBase


@TERMINATION_TERM_REGISTRY.register()
def cage_drop(env: RLEnvBase, *, metric_name: str = "cage_escape") -> torch.Tensor:
    """End the episode once the cube has been outside the hand's cage long enough.

    Thresholds the *counter* rather than the instantaneous escape distance, so a momentary
    overshoot the policy recovers from does not kill the episode -- only a sustained
    excursion does. The counter is maintained by the shared cage metric term, whose
    ``max_outside_steps`` sets the threshold, so the penalty and the termination can never
    drift apart.
    """
    term = env.metric_manager.get_term(metric_name)
    return term.penalty_counter >= float(term.max_outside_steps)


__all__ = ["cage_drop"]
