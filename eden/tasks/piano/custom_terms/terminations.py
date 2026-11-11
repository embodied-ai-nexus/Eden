"""Termination terms for the piano task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from eden.managers import TERMINATION_TERM_REGISTRY

if TYPE_CHECKING:
    from eden.envs.base import RLEnvBase


@TERMINATION_TERM_REGISTRY.register()
def song_over(env: RLEnvBase, *, command_name: str = "song") -> torch.Tensor:
    """The score has been played to its end. A time-out, not a failure."""
    return env.command_manager.get_term(command_name).song_over


__all__ = ["song_over"]
