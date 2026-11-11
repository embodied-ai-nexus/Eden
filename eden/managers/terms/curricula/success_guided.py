"""Success Guided Sampling (SGS): choose each episode's reset configuration from its recent success rate.

Zhang et al., "A Balanced Data Diet: Addressing the Exploration Bottleneck in Mega-Scale RL for Robot
Control" (CoRL 2026, arXiv:2610.12465), Sec. 3.2 and Appendix B.2. A task owns a fixed set of ``N``
reset configurations. SGS keeps the last ``history`` success/failure outcomes of every configuration,
scores each configuration's success rate ``p`` with a Beta-shaped kernel peaked at ``target``, and draws
the next configuration of every resetting env from a softmax over those scores. Most episodes go to
configurations the policy solves sometimes; every configuration keeps a non-zero probability, so the
estimates of the ones it cannot solve yet (or has mastered) keep being refreshed.

The term is a :class:`~eden.managers.resets.sampling.ResetConfigSampler`: the RESET event that owns the
configuration set binds it and calls :meth:`SuccessGuidedSampling.sample`, because the RESET events run
before any manager's ``reset()`` (see :mod:`eden.managers.resets.sampling`). ``sample`` grades the episode
that just ended before it draws the next one -- the termination manager still holds the finished step's
verdicts then, since its own ``reset()`` comes later in the same walk.
"""

from __future__ import annotations

import torch

from eden.managers.curriculum_manager import CURRICULUM_TERM_REGISTRY, CurriculumTerm
from eden.managers.resets.sampling import ResetConfigSampler


def success_guided_probabilities(
    success_rate: torch.Tensor, *, target: float, concentration: float, temperature: float, floor: float
) -> torch.Tensor:
    """Sampling probability of each configuration given its success rate (paper Eq. 1-2, Appendix B.2).

    ``w = (p + eps)^(k t) * (1 - p + eps)^(k (1 - t))`` -- the Beta(1 + k t, 1 + k (1 - t)) density in
    mode-concentration form, up to a constant -- floored at ``eps``; the logit is ``log(w + eps)`` and the
    probabilities a softmax over the logits at temperature ``max(temperature, 1)``.
    """
    p = success_rate.float()
    weight = (p + floor).pow(concentration * target) * (1.0 - p + floor).pow(concentration * (1.0 - target))
    logits = torch.log(weight.clamp(min=floor) + floor)
    return torch.softmax(logits / max(temperature, 1.0), dim=-1)


@CURRICULUM_TERM_REGISTRY.register()
class SuccessGuidedSampling(CurriculumTerm, ResetConfigSampler):
    """Draw each resetting env's reset configuration with probability peaked at a target success rate.

    A reset event that owns a configuration set names this term (its ``sampler`` parameter), binds it to
    the set size in its ``build()`` and calls :meth:`sample` from every reset. A term no reset event binds
    fails env construction.
    A configuration never graded reads ``p = 0``, so before any outcome is in, sampling is uniform.

    An outcome is recorded for an env when its episode *ended*: it is in the reset selection, the
    termination manager has it done, and it ran at least one step. A success is the ``success_term``
    termination having fired on that last step. Not graded: an env's first spawn, an ``env.reset()``
    that cut an episode short, and an episode the simulator flagged as corrupt
    (``termination_manager.solver_errors``) -- that one ended on the solver, not on the task. When more than
    ``history`` envs finish one configuration in the same reset, a uniformly random ``history`` of them are
    recorded.

    At eval (``env.training`` False) configurations are drawn uniformly and nothing is recorded:
    evaluation grades the policy on the task distribution, not on the sampler's current emphasis.

    The defaults are the paper's locomotion settings; it uses ``target=0.5, concentration=1.0,
    floor=1e-4`` for manipulation.

    Attributes
    ----------
    success_term
        Termination term whose firing marks an episode a success (e.g. ``all_goals_reached``).
    history
        Outcomes kept per configuration (a ring buffer); ``p`` is their mean.
    target
        Success rate the kernel peaks at, in [0, 1].
    concentration
        Kernel sharpness ``k >= 0``; 0 samples uniformly.
    temperature
        Softmax temperature, effective value ``max(temperature, 1)``.
    floor
        ``eps > 0``: keeps the kernel finite at ``p`` in {0, 1} and every probability non-zero.
    """

    success_term: str | None = None
    history: int = 100
    target: float = 0.66
    concentration: float = 5.0
    temperature: float = 2.0
    floor: float = 1e-8

    # ------------------------------------------------------------------ build

    def build(self) -> None:
        self._validate_params()
        terminations = getattr(self._env, "termination_manager", None)
        if terminations is None:
            raise ValueError(
                "SuccessGuidedSampling grades episodes by a termination term, but this env has no termination "
                "manager. Fix: build it from an EdenRLConfig with termination_options."
            )
        if self.success_term is None or self.success_term not in terminations.active_terms:
            raise ValueError(
                f"SuccessGuidedSampling: success_term={self.success_term!r} is not an active termination term. "
                f"Active termination terms: {', '.join(terminations.active_terms) or '(none)'}.\n"
                f"Fix: SuccessGuidedSampling.configure(success_term='<one of those>') -- the term whose firing "
                "means the episode succeeded."
            )
        with torch.inference_mode(False):
            self._env_arange = torch.arange(self.num_envs, device=self.device)
        self._require_bound_at_build(self._env)

    def _validate_params(self) -> None:
        problems = []
        if isinstance(self.history, bool) or not isinstance(self.history, int) or self.history < 1:
            problems.append(f"history={self.history!r} (a positive int)")
        if not 0.0 <= self.target <= 1.0:
            problems.append(f"target={self.target!r} (in [0, 1])")
        if self.concentration < 0.0:
            problems.append(f"concentration={self.concentration!r} (>= 0)")
        if self.temperature <= 0.0:
            problems.append(f"temperature={self.temperature!r} (> 0)")
        if self.floor <= 0.0:
            problems.append(f"floor={self.floor!r} (> 0; it keeps every probability non-zero)")
        if problems:
            raise ValueError(f"SuccessGuidedSampling: invalid {'; '.join(problems)}.")

    # ----------------------------------------------------------- the sampler

    def _on_bind(self, num_configs: int) -> None:
        with torch.inference_mode(False):
            # One extra "sink" row, num_configs: envs with nothing to record write there, so the
            # recording path never compacts a mask (a GPU->CPU sync per reset step). The sink is
            # garbage: its slots take racing duplicate writes and its count grows without bound, so
            # every reader of per-configuration state slices [:num_configs] and never sees it.
            self._outcomes = torch.zeros(num_configs + 1, self.history, dtype=torch.bool, device=self.device)
            self._count = torch.zeros(num_configs + 1, dtype=torch.long, device=self.device)
            # Running sum of each ring, so a reset updates p for the configurations it touched instead of
            # re-reducing all num_configs x history outcomes.
            self._success_sum = torch.zeros(num_configs + 1, dtype=torch.long, device=self.device)
            self._success_rate = torch.zeros(num_configs, device=self.device)
            self._probs = torch.full((num_configs,), 1.0 / num_configs, device=self.device)

    def _draw(self, selected: torch.Tensor) -> torch.Tensor:
        if self._env.training:
            self._record_finished_episodes(selected)
            draw = torch.multinomial(self._probs, self.num_envs, replacement=True)
        else:
            draw = torch.randint(self._num_configs, (self.num_envs,), device=self.device)
        return draw

    def _record_finished_episodes(self, selected: torch.Tensor) -> None:
        terminations = self._env.termination_manager
        ended = (
            selected
            & ~self.first_spawn
            & terminations.dones
            & ~terminations.solver_errors
            & (self._env.episode_length_buf > 0)
        )
        succeeded = terminations.get_term_dones(self.success_term)

        # Each ended env takes the next free slot of its configuration's ring; envs that finished the
        # same configuration in this batch take consecutive slots, in a random order (shuffled before a
        # stable sort), so when more than ``history`` finish together the ones kept are not biased toward
        # low env indices. Ranks past ``history`` go to the sink row, so no two writes share a slot and the
        # running sums stay exact.
        sink = self._num_configs
        key = torch.where(ended, self._assignment, sink)
        shuffle = torch.randperm(self.num_envs, device=self.device)
        sorted_key, order = torch.sort(key[shuffle], stable=True)
        order = shuffle[order]
        rank = self._env_arange - torch.searchsorted(sorted_key, sorted_key)
        sorted_key = torch.where(rank < self.history, sorted_key, sink)
        slot = (self._count[sorted_key] + rank) % self.history
        outcome = succeeded[order]
        replaced = self._outcomes[sorted_key, slot]
        self._outcomes[sorted_key, slot] = outcome
        self._success_sum.index_add_(0, sorted_key, outcome.long() - replaced.long())
        self._count.index_add_(0, sorted_key, torch.ones_like(sorted_key))
        self._refresh_distribution()

    def _refresh_distribution(self) -> None:
        """Recompute ``p`` and the sampling distribution from the outcome rings."""
        n = self._num_configs
        graded = self._count[:n].clamp(max=self.history)
        self._success_rate.copy_(self._success_sum[:n] / graded.clamp(min=1))
        self._probs.copy_(
            success_guided_probabilities(
                self._success_rate,
                target=self.target,
                concentration=self.concentration,
                temperature=self.temperature,
                floor=self.floor,
            )
        )

    # -------------------------------------------------------------- reading

    @property
    def success_rate(self) -> torch.Tensor:
        """``(num_configs,)``: mean of each configuration's recorded outcomes; 0 where none is recorded."""
        self._require_bound("success_rate")
        return self._success_rate

    @property
    def num_outcomes(self) -> torch.Tensor:
        """``(num_configs,)`` long: outcomes recorded per configuration over the run (not capped at history)."""
        self._require_bound("num_outcomes")
        return self._count[: self._num_configs]

    @property
    def probabilities(self) -> torch.Tensor:
        """``(num_configs,)``: the distribution the next training reset draws from."""
        self._require_bound("probabilities")
        return self._probs

    def compute(self) -> dict[str, torch.Tensor]:
        """Publish the sampler's state; device tensors, so logging it costs no sync.

        ``mean_success`` averages ``p`` over the configurations graded at least once;
        ``frontier_frac`` is the share of the whole set with ``0 < p < 1`` -- solved sometimes, the
        configurations SGS exists to find; ``sampling_ess_frac`` is the effective sample size of the
        sampling distribution as a fraction of the set (1 = uniform).
        """
        self._require_bound("compute()")
        visited = self.num_outcomes > 0
        num_visited = visited.sum()
        p = self._success_rate
        return {
            "mean_success": (p * visited).sum() / num_visited.clamp(min=1),
            "visited_frac": num_visited / self._num_configs,
            "frontier_frac": ((p > 0.0) & (p < 1.0)).float().mean(),
            "sampling_ess_frac": 1.0 / (self._probs.square().sum() * self._num_configs),
        }

    # ------------------------------------------------------------ checkpoint

    def _sampler_state(self) -> dict:
        """The per-configuration outcome history; the per-env assignment is not carried (envs respawn)."""
        return {
            "history": self.history,
            "outcomes": self._outcomes[: self._num_configs].detach().cpu().clone(),
            "count": self.num_outcomes.detach().cpu().clone(),
        }

    def _load_sampler_state(self, state: dict) -> None:
        if state.get("history") != self.history:
            raise ValueError(
                f"SuccessGuidedSampling: the checkpoint's rings hold history={state.get('history')!r} outcomes per "
                f"configuration, this term keeps history={self.history}. Fix: resume with the history the checkpoint "
                "was trained with. Do NOT drop this term's checkpoint state to get past it: a ring of another length "
                "would misplace every slot."
            )
        n = self._num_configs
        self._outcomes[:n].copy_(state["outcomes"].to(self.device))
        self._count[:n].copy_(state["count"].to(self.device))
        # Never-written slots are False, so a whole-ring sum is the sum of the recorded outcomes.
        self._success_sum[:n].copy_(self._outcomes[:n].sum(dim=1))
        self._refresh_distribution()
