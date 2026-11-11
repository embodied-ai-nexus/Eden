"""Adaptive curricula for in-hand cube reorientation.

Port of ``wuji-mjlab``'s ``curriculums.py``. Both terms are *outcome-driven* rather than
step-scheduled: they read what the policy actually achieved last episode and move a scalar
up or down. That scalar is then read by the disturbance event, so the physical difficulty
of the task tracks competence instead of wall-clock.

Upstream keeps the scalars on the env (``env._reorient_success_curriculum``); here each
term owns its own, which is what gives them reset semantics and keeps two tasks in one
process from sharing state.

Both grade on the episode that just ended, so the work happens in ``reset`` (where the
pre-reset counters are still readable -- ``EnvBase._reset_idx`` zeroes
``episode_length_buf`` only after every manager has reset) and ``compute`` just publishes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from eden.managers import CURRICULUM_TERM_REGISTRY
from eden.managers.curriculum_manager import CurriculumTerm

if TYPE_CHECKING:
    from eden.envs.base import EnvBase
    from eden.options.managers.curricula import CurriculumTermOptions


@CURRICULUM_TERM_REGISTRY.register()
class ReorientSuccessCurriculum(CurriculumTerm):
    """Track the fraction of episodes that reach at least ``count_threshold`` goals.

    Nudges a scalar in [0, 1] up by ``delta_per_loop / num_envs`` for every episode that
    banked enough goals and down by the same for every one that did not, so the value
    settles where roughly half the population is clearing the bar. Published for logging
    and available to any term that wants to scale with competence.

    Parameters
    ----------
    command_name:
        Key of the reorientation command term.
    count_threshold:
        Goals an episode must reach to count as a success.
    delta_per_loop:
        Total step size applied across a full batch of resets.
    """

    command_name: str = "reorient"
    count_threshold: int = 3
    delta_per_loop: float = 0.08

    def __init__(self, env: EnvBase, options: CurriculumTermOptions):
        super().__init__(env, options)
        self._value = 0.0
        self._last_success_frac = 0.0
        self._last_mean_goals = 0.0

    def reset(self, envs_idx: torch.Tensor | None = None) -> None:
        command = self._env.command_manager.get_term(self.command_name)
        # The snapshot, not the live counter: command_manager resets before
        # curriculum_manager in the reset walk, so the live count is already zero here.
        goals = command.goal_reach_count_at_reset
        goals = goals if envs_idx is None else goals[envs_idx]
        if goals.numel() == 0:
            return
        succeeded = goals >= self.count_threshold
        delta = (2.0 * succeeded.float() - 1.0).sum().item() * self.delta_per_loop / self._env.num_envs
        self._value = max(0.0, min(1.0, self._value + delta))
        self._last_success_frac = succeeded.float().mean().item()
        self._last_mean_goals = goals.float().mean().item()

    def compute(self) -> dict[str, float]:
        return {
            "value": self._value,
            "success_env_frac": self._last_success_frac,
            "mean_goal_reach_count": self._last_mean_goals,
        }


@CURRICULUM_TERM_REGISTRY.register()
class AdaptiveEpisodeCurriculum(CurriculumTerm):
    """Scale physical difficulty by how long episodes are surviving.

    Rises by ``inc_rate`` for each episode that reaches ``target_steps`` and falls by up to
    ``dec_rate`` for each early drop, weighted by how early it was -- so dropping the cube
    immediately costs the full amount and dropping it just short of the target costs almost
    nothing. Floored at ``min_scale`` so disturbances never switch off entirely.

    The scalar is read by ``ObjectVelocityDisturbance``: the perturbations only grow while
    the policy is actually holding on to the cube, which stops the ramp from stalling
    training before a grasp is learned at all.

    Parameters
    ----------
    target_steps:
        Episode length [steps] that counts as surviving.
    inc_rate:
        Increase per surviving episode, before the 1/num_envs normalization.
    dec_rate:
        Maximum decrease per early drop, before the 1/num_envs normalization.
    min_scale:
        Lower bound (and initial value) of the published scalar.
    drop_gamma:
        Exponent shaping how sharply an early drop is punished relative to a late one.
    """

    target_steps: int = 800
    inc_rate: float = 0.1
    dec_rate: float = 0.2
    min_scale: float = 0.05
    drop_gamma: float = 1.0

    def __init__(self, env: EnvBase, options: CurriculumTermOptions):
        super().__init__(env, options)
        self._value = self.min_scale
        self._last_mean_steps = 0.0
        self._last_early_drop_frac = 0.0

    def build(self) -> None:
        if self.target_steps < 1:
            raise ValueError(f"AdaptiveEpisodeCurriculum: target_steps must be >= 1, got {self.target_steps}.")

    def reset(self, envs_idx: torch.Tensor | None = None) -> None:
        lengths = self._env.episode_length_buf
        lengths = lengths if envs_idx is None else lengths[envs_idx]
        if lengths.numel() == 0:
            return
        lengths = lengths.float()
        progress = (lengths / float(self.target_steps)).clamp(0.0, 1.0)
        reached = lengths >= float(self.target_steps)

        terminated = getattr(self._env.termination_manager, "terminated", None)
        if terminated is None:
            early_drop = torch.zeros_like(reached)
        else:
            terminated = terminated if envs_idx is None else terminated[envs_idx]
            early_drop = terminated & ~reached

        increase = reached.float() * self.inc_rate
        penalty = early_drop.float() * (1.0 - progress).pow(self.drop_gamma) * self.dec_rate
        delta = (increase - penalty).sum().item() / self._env.num_envs
        self._value = max(self.min_scale, min(1.0, self._value + delta))
        self._last_mean_steps = lengths.mean().item()
        self._last_early_drop_frac = early_drop.float().mean().item()

    @property
    def value(self) -> float:
        """The current difficulty scalar, readable in both training and eval.

        ``CurriculumManager`` freezes terms whole at eval, so ``get_state()`` raises for the
        entire rollout there rather than hand a consumer a ``None`` it would have to invent a
        silent fallback for. The term's own level is the honest answer in both modes: the adapted
        value while training, the configured ``min_scale`` while frozen.
        """
        return self._value

    def compute(self) -> dict[str, float]:
        return {
            "value": self._value,
            "mean_episode_steps": self._last_mean_steps,
            "early_drop_frac": self._last_early_drop_frac,
        }


__all__ = ["AdaptiveEpisodeCurriculum", "ReorientSuccessCurriculum"]
