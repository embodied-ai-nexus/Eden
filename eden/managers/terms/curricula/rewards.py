"""Reward-weight curricula (staged ramp-up, penalty scheduling)."""

from __future__ import annotations
from typing import TYPE_CHECKING

import torch

import eden as en
from eden.managers.curriculum_manager import (
    CURRICULUM_TERM_REGISTRY,
    CurriculumTerm,
    CurriculumTermOptions,
)

if TYPE_CHECKING:
    from eden.envs.base import RLEnvBase


@CURRICULUM_TERM_REGISTRY.register()
class StageRewardWeightCurriculum(CurriculumTerm):
    """Curriculum that sets a reward term's weight at given training steps.

    Parameters
    ----------
    reward_term_name: str
        The name of the reward term.
    weight_stages: list[tuple[int, float]]
        ``(step, weight)`` stages, in any order: the latest stage whose step
        ``env.common_step_counter`` has reached sets the weight.
    """

    reward_term_name: str = ""
    weight_stages: list[tuple[int, float]] = []

    def __init__(self, env: RLEnvBase, options: CurriculumTermOptions):
        super().__init__(env=env, options=options)
        # make sure the stages are sorted in descending order (copy; don't mutate config)
        self.weight_stages = sorted(self.weight_stages, key=lambda x: x[0], reverse=True)

    def build(self) -> None:
        # The reward manager is built before the curricula (``EnvBase._manager_order``), so an
        # unknown reward_term_name raises at env construction instead of on the first compute().
        self._reward_term = self._env.reward_manager.get_term(self.reward_term_name)

    def compute(self, *args, **kwargs) -> dict[str, float]:
        # NOTE: the stages are sorted in descending order, so we can break the loop
        for stage_step, stage_weight in self.weight_stages:
            if self._env.common_step_counter >= stage_step:
                self._reward_term.weight = stage_weight
                break
        # Publish the weight actually in force -- the reward term's configured weight until the
        # first stage's step is reached, the applied stage after. Returning nothing would read back
        # through ``CurriculumManager.get_state`` as ``None`` (indistinguishable from "never computed")
        # and log no ``Curriculum/`` key at all.
        return {"weight": float(self._reward_term.weight)}


@CURRICULUM_TERM_REGISTRY.register()
class PenaltyCurriculum(CurriculumTerm):
    """Curriculum that gradually scales reward weights for a set of reward terms.

    Monitors average episode length and adjusts a scale factor applied to all
    targeted reward terms. When episodes are short (agent struggling), the scale
    decreases to ease penalties. When episodes are long (agent succeeding), the
    scale increases to enforce stricter behaviour.

    The scale factor multiplies the *configured* weight of each term, so the
    effective weight is ``configured_weight * scale``.

    Targets can be specified by name OR by tag — exactly one entry path must be
    used per curriculum instance. Mixing the two would split intent across two
    fields and make refactors error-prone (e.g. renaming a term forgets the
    name-list, but the tag-list still matches). Prefer ``reward_term_tags`` when
    the same curriculum should apply to a group of penalties whose roster may
    grow (just tag new terms); use ``reward_term_names`` for explicit, audited
    targets.

    Parameters
    ----------
    reward_term_names: list[str]
        Explicit names of reward terms to scale. Mutually exclusive with
        ``reward_term_tags``.
    reward_term_tags: list[str]
        Tags to match on ``RewardTermOptions.tags``. A term is selected if any
        of its tags appears in this list. Mutually exclusive with
        ``reward_term_names``.
    initial_scale: float
        Starting scale factor. Default 0.1 (10% of configured weight).
    min_scale: float
        Minimum allowed scale. Default 0.0.
    max_scale: float
        Maximum allowed scale. Default 1.0.
    level_up_threshold: float
        Average episode length (in steps) above which the scale increases.
    level_down_threshold: float
        Average episode length (in steps) below which the scale decreases.
    degree: float
        Amount to adjust the scale per curriculum compute call.
    num_episode_average: int
        Number of most recent episode lengths to average over. Ignored when
        ``tracker_name`` is set (the named tracker owns the EMA).
    tracker_name: str | None
        Optional name of an ``AverageEpisodeLengthTracker`` curriculum term
        whose ``average_episode_length`` should drive this curriculum instead
        of the per-instance buffer. The tracker must be declared **before**
        this PenaltyCurriculum in ``curriculum_options`` so the tracker's
        compute() runs first within a step. When ``None`` (default), a
        per-instance CPU buffer of the last ``num_episode_average`` lengths is used.
    """

    reward_term_names: list[str] = []
    reward_term_tags: list[str] = []
    initial_scale: float = 0.1
    min_scale: float = 0.0
    max_scale: float = 1.0
    level_up_threshold: float = 750.0
    level_down_threshold: float = 150.0
    degree: float = 0.00025
    num_episode_average: int = 1000
    tracker_name: str | None = None

    def __init__(self, env: RLEnvBase, options: CurriculumTermOptions):
        super().__init__(env=env, options=options)
        if not self.reward_term_names and not self.reward_term_tags:
            raise ValueError("PenaltyCurriculum requires either `reward_term_names` or `reward_term_tags` to be set.")
        if self.reward_term_names and self.reward_term_tags:
            raise ValueError(
                "PenaltyCurriculum: `reward_term_names` and `reward_term_tags` are mutually exclusive — "
                "set exactly one. Use names for explicit targets, tags for groups whose roster may grow."
            )
        self._scale = self.initial_scale
        self._targeted_terms: list = []
        self._base_weights: list[float] = []
        if self.tracker_name is None:
            # Rolling buffer for episode lengths. Kept on CPU on purpose: the running
            # mean is consumed by compute() every env step and used to branch on
            # ``level_up_threshold`` / ``level_down_threshold``. A device-side mean
            # would force a GPU→CPU sync per step, defeating async kernel launch.
            # Episode lengths are tiny ints; the CPU buffer is cheap.
            self._episode_lengths = torch.zeros(self.num_episode_average, device="cpu")
            self._ep_cursor = 0
            self._ep_count = 0

    def build(self) -> None:
        """Resolve the targeted reward terms and install ``initial_scale`` on them.

        The reward manager is built before the curricula (``EnvBase._manager_order``), so an
        unknown ``reward_term_names`` entry raises at env construction rather than on the first
        reset. Installing the scale here — rather than on whichever of ``reset()`` / ``compute()``
        runs first — is also what keeps the initial scale applied at eval:
        ``CurriculumManager`` freezes both of those there, so a term that only ever installed its
        starting level from one of them would grade an eval rollout at the *unscaled* configured
        weights.
        """
        reward_manager = self._env.reward_manager
        if self.reward_term_names:
            # Explicit names: error if any is missing.
            self._targeted_terms = [reward_manager.get_term(name) for name in self.reward_term_names]
        else:
            # Tag match: scan all terms and keep those carrying any listed tag.
            tag_set = set(self.reward_term_tags)
            self._targeted_terms = [
                term
                for _, term in reward_manager.iter_terms()
                if tag_set.intersection(getattr(term, "tags", None) or [])
            ]
        self._base_weights = [term.weight for term in self._targeted_terms]
        # Empty resolution is allowed (e.g. tag-based curriculum on a task that doesn't
        # use that tag yet) but warn loudly so it isn't a silent no-op.
        if not self._targeted_terms:
            en.logger.warning(
                "PenaltyCurriculum resolved zero target reward terms "
                f"(reward_term_names={self.reward_term_names!r}, reward_term_tags={self.reward_term_tags!r}). "
                "The curriculum will be a no-op — check that names exist and that target terms carry the listed tags."
            )
        self._apply_scale()

    def reset(self, envs_idx: torch.Tensor | None = None) -> None:
        if self.tracker_name is not None:
            # The named tracker owns the EMA; nothing to update here.
            return
        if envs_idx is None:
            return
        # Pull episode lengths to CPU once per reset and update the running mean
        # incrementally (avoids per-step ``.mean().item()`` syncs in compute()).
        ep_lens = self._env.episode_length_buf[envs_idx].float().cpu()
        n = ep_lens.shape[0]
        if n == 0:
            return
        buf_size = self.num_episode_average
        indices = (torch.arange(n, device=self._episode_lengths.device) + self._ep_cursor) % buf_size
        self._episode_lengths[indices] = ep_lens
        self._ep_cursor = (self._ep_cursor + n) % buf_size
        self._ep_count = min(self._ep_count + n, buf_size)
        # Refresh cached running mean as a Python float — no per-step sync below.
        self._avg_ep_len = float(self._episode_lengths[: self._ep_count].mean())

    def _read_tracker_state(self) -> tuple[float, bool]:
        """Return ``(average_episode_length, is_primed)`` from the named tracker.

        ``CurriculumManager.get_state`` raises when the named term has not computed yet — a
        declaration-order mistake in ``curriculum_options``, named as such.
        """
        state = self._env.curriculum_manager.get_state(self.tracker_name)
        if "average_episode_length" not in state:
            raise KeyError(
                f"PenaltyCurriculum(tracker_name={self.tracker_name!r}): the named curriculum term's "
                "compute() output does not contain 'average_episode_length'. Expected an "
                "AverageEpisodeLengthTracker (or compatible term) at this slot."
            )
        # ``is_primed`` is opt-in: a third-party tracker without the flag is assumed primed.
        is_primed = bool(state.get("is_primed", True))
        return float(state["average_episode_length"]), is_primed

    def compute(self, *args, **kwargs) -> dict[str, float]:
        if self.tracker_name is not None:
            # A tracker that is registered but hasn't computed yet raises inside get_state: it must
            # be declared *before* this term in curriculum_options, and the manager says so.
            avg_ep_len, is_primed = self._read_tracker_state()
            # Same gate as the buffer path's ``_ep_count > 0``: only move the scale once
            # the tracker has actually consumed at least one reset. Otherwise the
            # initial-zero estimate would push ``_scale`` toward ``min_scale`` on
            # every pre-first-reset step.
            if is_primed:
                if avg_ep_len > self.level_up_threshold:
                    self._scale = min(self._scale + self.degree, self.max_scale)
                elif avg_ep_len < self.level_down_threshold:
                    self._scale = max(self._scale - self.degree, self.min_scale)
                self._apply_scale()
            return {"scale": self._scale, "avg_episode_length": avg_ep_len}

        avg_ep_len = getattr(self, "_avg_ep_len", 0.0)

        # Only update the scale once we have at least one recorded episode length.
        # Otherwise compute() runs every env.step() from start of training and the
        # default avg_ep_len=0.0 would push _scale toward min_scale before any
        # real episode data exists.
        if self._ep_count > 0:
            if avg_ep_len > self.level_up_threshold:
                self._scale = min(self._scale + self.degree, self.max_scale)
            elif avg_ep_len < self.level_down_threshold:
                self._scale = max(self._scale - self.degree, self.min_scale)
            self._apply_scale()
        return {"scale": self._scale, "avg_episode_length": avg_ep_len}

    def _apply_scale(self) -> None:
        for term, base_w in zip(self._targeted_terms, self._base_weights):
            term.weight = base_w * self._scale
