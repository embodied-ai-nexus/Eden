"""Locomotion curricula (e.g. average-episode-length tracking)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from eden.managers.curriculum_manager import (
    CURRICULUM_TERM_REGISTRY,
    CurriculumTerm,
    CurriculumTermOptions,
)

if TYPE_CHECKING:
    from eden.envs.base import RLEnvBase


# Reference env count used to scale the tracker's effective EMA window with `num_envs`,
# so the same `num_compute_average_epl` config value yields a `num_envs`-invariant
# time-constant. Matches Eden's default `EnvOptions.num_envs=4096`.
BASE_NUM_ENVS = 4096


@CURRICULUM_TERM_REGISTRY.register()
class AverageEpisodeLengthTracker(CurriculumTerm):
    """Track the running mean of episode length.

    Maintains a single device-side scalar updated with an EMA whose effective
    time-constant is `num_compute_average_epl` episodes' worth of resets at the
    reference env count `BASE_NUM_ENVS`, scaled linearly with `env.num_envs`.

    Other curriculum terms read the current estimate via
    ``env.curriculum_manager.get_state(<this term's name>)``. The state dict
    carries both ``average_episode_length`` and an ``is_primed`` flag — the
    latter is False until at least one ``reset()`` call has actually fed
    episode lengths into the EMA, so consumers can avoid acting on the
    initial-zero estimate before any episode has finished.

    Declare this term *before* any consumer in ``curriculum_options`` so the
    consumer's ``compute()`` sees a fresh value on the same step. Declared
    after, ``get_state`` raises and names the ordering rather than reading as
    "no value yet".

    Parameters
    ----------
    num_compute_average_epl: int
        Reference EMA window in number of resets, expressed at `BASE_NUM_ENVS`.
        The effective window scales as
        ``max(1, int(num_compute_average_epl * num_envs / BASE_NUM_ENVS))``.
    """

    num_compute_average_epl: int = 1000

    def __init__(self, env: RLEnvBase, options: CurriculumTermOptions):
        super().__init__(env=env, options=options)
        self._scaled_window = max(1, int(self.num_compute_average_epl * self._env.num_envs / BASE_NUM_ENVS))
        self.average_episode_length = torch.tensor(0.0, device=self.device, dtype=torch.float)
        self._is_primed = False

    def reset(self, envs_idx: torch.Tensor | None = None) -> None:
        # ``envs_idx is None`` is treated as "no episodes to record" rather than
        # "all envs". CurriculumManager always passes the indices of envs that
        # just terminated; a None here would mean a startup-style call where
        # ``episode_length_buf`` is all-zero (or mid-episode), and feeding that
        # into the EMA would corrupt the running mean. Same semantics as
        # PenaltyCurriculum.reset().
        if envs_idx is None:
            return
        ep_lens = self._env.episode_length_buf[envs_idx].float()
        n = ep_lens.numel()
        if n == 0:
            return
        weight = min(n / self._scaled_window, 1.0)
        current_average = ep_lens.mean()
        # Device-side EMA update. ``lerp_`` is in-place: it avoids per-reset
        # tensor allocation on the running estimate.
        self.average_episode_length.lerp_(current_average, weight)
        self._is_primed = True

    def compute(self, *args, **kwargs) -> dict[str, float | bool]:
        return {
            "average_episode_length": float(self.average_episode_length),
            "is_primed": self._is_primed,
        }

    def state_dict(self) -> dict:
        return {
            "average_episode_length": self.average_episode_length.detach().cpu().clone(),
            "is_primed": self._is_primed,
        }

    def load_state_dict(self, state: dict) -> None:
        avg = state.get("average_episode_length")
        if avg is not None:
            self.average_episode_length = torch.as_tensor(float(avg), device=self.device, dtype=torch.float)
        if "is_primed" in state:
            self._is_primed = bool(state["is_primed"])
