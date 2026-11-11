"""Termination terms for the manipulation tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from eden.managers import TERMINATION_TERM_REGISTRY

if TYPE_CHECKING:
    from eden.envs.base import RLEnvBase


@TERMINATION_TERM_REGISTRY.register()
def object_fallen(env: "RLEnvBase", *, object_name: str, threshold: float = -0.1) -> torch.Tensor:
    """Terminate if the object falls below ``threshold`` (off the surface)."""
    return env.entities[object_name].get_pos()[:, 2] < threshold
