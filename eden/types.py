"""Shared type aliases used across Eden."""

from typing import Any, Literal, TypeAlias

import torch
from genesis.options.morphs import Morph


MorphLike: TypeAlias = Morph | list[Morph]
"""A morph or a list of morphs (used to seed deformable/rigid primitives)."""

TaskTag: TypeAlias = Literal[
    "locomotion",
    "motion_tracking",
    "manipulation",
    "dexterous",
    "loco_manipulation",
    "deformable",
    "benchmarks",
]
"""A task category tag (``@TASK_REGISTRY.register(tags=(...))``). The rule for each is in
:data:`eden.tasks.registry.TASK_TAGS`, which holds the same names in the same order."""

VisMode: TypeAlias = Literal["visual", "collision", "particle", "sdf", "recon"]
"""Rendering/visualization mode for an entity."""

VecEnvObs: TypeAlias = dict[str, torch.Tensor]
"""Observation payload returned by vectorized environments."""

VecEnvReset: TypeAlias = tuple[VecEnvObs, dict[str, Any]]
"""Return type of ``EnvBase.reset``: ``(obs, info)``."""

VecEnvStep: TypeAlias = tuple[
    VecEnvObs,
    torch.Tensor | None,  # reward (None for non-RL EnvBase)
    torch.Tensor | None,  # terminated
    torch.Tensor | None,  # timeouts
    dict[str, Any],
]
"""Return type of ``EnvBase.step`` / ``RLEnvBase.step``: ``(obs, reward, terminated, timeouts, info)``."""


__all__ = [
    "MorphLike",
    "TaskTag",
    "VecEnvObs",
    "VecEnvReset",
    "VecEnvStep",
    "VisMode",
]
