"""Curriculum manager for updating environment quantities subject to a training curriculum."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from eden.managers.base import (
    ConfigurableTermBase,
    ConfigurableTermFuncWrapper,
    FlatManagerBase,
    ManagerTermFuncWrapperBase,
)
from eden.options.managers.curricula import (
    CurriculumManagerOptions,
    CurriculumTermOptions,
)
from eden.utils.registry import TermRegistry

if TYPE_CHECKING:
    from eden.envs.base import EnvBase


class CurriculumTerm(ConfigurableTermBase[CurriculumTermOptions]):
    """Base class for curriculum terms.

    ``compute()`` returns the term's published state: a dict of scalars, logged as
    ``Curriculum/<term_name>/<key>`` and served to other terms by
    :meth:`CurriculumManager.get_state`. **Publish it on every step.** A term that returns a dict
    on some steps and ``None`` on others makes ``get_state`` raise on exactly the steps it skipped
    — the contract is "never ``None``", so there is no quiet no-value for a consumer to branch on.
    A term with genuinely nothing to share simply returns ``None`` always, and its state
    is then not readable at all, which ``get_state`` says in as many words.
    """


class CurriculumTermFuncWrapper(ConfigurableTermFuncWrapper[CurriculumTermOptions]):
    """Base class for curriculum terms defined as a function."""


CURRICULUM_TERM_REGISTRY = TermRegistry(
    "CURRICULUM_TERM", CurriculumTerm, CurriculumTermFuncWrapper, populated_by=("eden.managers.terms.curricula",)
)


def _published(value):
    """A term's cached state as a log value: a tensor stays on the device, as a snapshot.

    ``.item()`` would be a GPU->CPU sync per key on every step the auto-reset fires; the loggers
    take a 0-dim tensor as they take a float. A copy, because a term may update the tensor it cached in
    place from its own ``reset()``, which runs right after this publish.
    """
    return value.detach().clone() if isinstance(value, torch.Tensor) else value


class CurriculumManager(FlatManagerBase[CurriculumManagerOptions]):
    """Curriculum manager for managing the curriculum of the environment."""

    _registry = CURRICULUM_TERM_REGISTRY

    def __init__(self, env: EnvBase, options: CurriculumManagerOptions):
        super().__init__(env=env, options=options)

        # Keyed only once a term has actually computed, so "never computed" is a missing key
        # rather than a ``None`` value indistinguishable from "computed and published nothing".
        # :meth:`get_state` turns each of those into its own raise.
        self._curriculum_state: dict[str, Any] = dict()
        #: Name of the term currently inside :meth:`compute`, or ``None`` outside the loop. Lets
        #: :meth:`get_state` tell "declared after its consumer" from "read before the run's first
        #: compute()" exactly, rather than guessing from an empty cache -- the cache is *also*
        #: empty for the first term in the loop on the first step, which is the common ordering bug.
        self._computing: str | None = None
        #: ``common_step_counter`` at each term's last publish, written beside the value it dates.
        #: Per-term rather than one stamp for the loop so a term whose ``compute()`` *raised* keeps
        #: the date of the step it last succeeded on, instead of being credited with the step it
        #: died on -- the loop's bookkeeping must not lie to a caller that caught the exception.
        self._published_at: dict[str, int] = dict()

    def summary(self) -> str:
        return self._format_summary_table(
            title="Active Curriculum Terms",
            field_names=["Index", "Name"],
            rows=([index, name] for index, name in enumerate(self._term_names)),
            align={"Name": "l"},
        )

    def reset(self, envs_idx: torch.Tensor | slice | None = None) -> dict[str, float | torch.Tensor]:
        """Publish the cached per-term state, then feed the finished episodes to the terms.

        The term half is frozen at eval, exactly like :meth:`compute` — a curriculum term's
        ``reset()`` is how it accumulates the statistics ``compute()`` would adapt from (an episode-
        length EMA, a success rate), so running one without the other would have the internal state
        drift away from the published one for the whole rollout, on evidence nothing is allowed to
        act on.

        The publish half reports whatever the last :meth:`compute` cached — and **an eval env has
        no last compute()**: ``env.training`` is fixed at construction, so an eval rollout logs no
        ``Curriculum/*`` key at all, from its first reset to its last. That is deliberate: the
        level an eval was graded at is the one its config pins, which the
        saved config already records; publishing it from here would mean running ``compute()`` at
        build, and the two terms whose ``compute()`` has side effects —
        ``StageRewardWeightCurriculum`` and ``VelocityCommandCurriculum`` — would then *overwrite*
        the configured reward weights and command ranges with their step-0 stage.
        """
        extras = {}
        for term_name, term_state in self._curriculum_state.items():
            if term_state is not None:
                if isinstance(term_state, dict):
                    for key, value in term_state.items():
                        extras[f"Curriculum/{term_name}/{key}"] = _published(value)
                else:
                    extras[f"Curriculum/{term_name}"] = _published(term_state)
        if not self._env.training:
            return extras
        for term in self._reset_terms:
            term.reset(envs_idx=envs_idx)
        return extras

    def compute(self) -> None:
        """Adapt every term and cache what it published — a no-op at eval.

        Curricula adapt task difficulty from training statistics, so an eval rollout freezes them:
        otherwise its difficulty is non-stationary, e.g. an EMA-gated goal curriculum at
        ``num_envs=1`` collapses its window to a single episode, so one failed episode retreats the
        level mid-rollout and every later episode is graded on an easier task than the one being
        reported. Terms still apply their configured initial level at build, so eval entry points
        pin difficulty through the term's own config.

        The freeze is all-or-nothing and lives here rather than at the call site so it cannot go
        asymmetric: :meth:`reset` is gated the same way. Nothing is cached either, so
        :meth:`get_state` raises for every term for the whole rollout rather than handing a
        consumer a ``None`` that reads like "not ready yet".
        """
        if not self._env.training:
            return
        try:
            for name, term in self._terms.items():
                # Recorded so a term that reads a not-yet-computed sibling gets an error naming
                # *which config entry* to move -- the traceback carries the consumer's class, but a
                # task can hold several entries of one class (``penalty_torques`` and
                # ``penalty_smoothness`` in docs/guides/curricula.md), and the entry key is what you edit.
                self._computing = name
                self._curriculum_state[name] = term.compute()
                self._published_at[name] = self._env.common_step_counter
        finally:
            self._computing = None

    def get_state(self, term_name: str) -> dict:
        """Return the most recent ``compute()`` output cached for the named term.

        **Never returns ``None``.** Every way there can be no state is a raise, because the one
        thing a consumer must not do is read the absence as "not ready yet" and fall back to a
        default: that fallback then stands for the whole run, with nothing logged and nothing
        raised. When you want a level that is correct in *both* modes, read it off the term
        itself — ``curriculum_manager.get_term(name)`` — which is what ``ObjectVelocityDisturbance``
        does; ``get_state`` is only for consuming what another term *published* this step, and it
        enforces that rather than only asking for it: a cached value from an *earlier* step raises
        too. Without that the same read is treated two ways -- the run's first step raises, and
        every step after silently serves the previous step's level, which is the intermittency this
        contract exists to remove.

        Freshness is dated against ``env.common_step_counter``, which advances once per step with
        the physics (:meth:`RLEnvBase.step`), so a read from phases 5-20 is placed exactly. Phases
        1-3 run *before* that advance and so still date as the previous step; there is no earlier
        per-step marker to key off, and what they would read is in any case the newest publish that
        can exist, since this step's :meth:`compute` has not run.

        Raises
        ------
        KeyError
            If no curriculum term is named ``term_name``, so a typo cannot pass for "not computed
            yet" and leave a consumer that treats that as "not ready" on its default for the whole run.
        RuntimeError
            If the named term has no state this step. The five causes get five messages, each
            naming the one edit that fixes it: the env is at eval, so nothing ever computes; the
            term is declared after the consumer now reading it; the read happens before the run's
            first ``compute()`` at all; the term's ``compute()`` returns ``None``; or the read runs
            earlier in the step than phase 16, so the cache still holds the previous step's
            publish.
        """
        if term_name not in self._terms:
            # One line on purpose: KeyError.__str__ is repr(args[0]), so a newline here would
            # reach the reader as a literal "\n" in an escaped blob.
            raise KeyError(
                f"No curriculum term named '{term_name}'. Declared curriculum terms: "
                f"{', '.join(self._terms) or '(none)'}. Fix: use one of those keys, or add "
                f"'{term_name}' to curriculum_options."
            )
        if term_name in self._curriculum_state:
            state = self._curriculum_state[term_name]
            if state is None:
                raise RuntimeError(
                    f"Curriculum term '{term_name}' ({self._term_label(term_name)}) computed but published "
                    f"nothing: its compute() returned None.\n"
                    f"Fix: return a dict of scalars from {self._term_label(term_name)}.compute() — it is both "
                    f"what get_state() serves and what is logged as Curriculum/{term_name}/<key>.\n"
                    "A term that publishes on some steps and not others raises here intermittently; publish "
                    "every step, or do not let anything read this term's state."
                )
            published_at = self._published_at[term_name]
            step = self._env.common_step_counter
            if published_at != step:
                raise RuntimeError(
                    f"Curriculum term '{term_name}' last published at step {published_at}, and it is now step "
                    f"{step}: this cached value is one control step stale.\n"
                    "reading from: a phase earlier in the step than CurriculumManager.compute(), which is "
                    "phase 16 (docs/guides/environments.md) — a curriculum term's reset(), a metric, a "
                    "termination and a reward all run before it\n"
                    f"Fix: read the level off the term itself — curriculum_manager.get_term('{term_name}'), a "
                    f"{self._term_label(term_name)} — which is current at every phase; or move this read into "
                    "a hook that runs after phase 16, where get_state() serves the current step's publish.\n"
                    "Do NOT call CurriculumManager.compute() yourself to refresh it: terms adapt as a side "
                    "effect of computing — PenaltyCurriculum steps its scale by `degree` toward a threshold "
                    "bound on each call — so a second call inside one step advances the curriculum twice."
                )
            return state
        if not self._env.training:
            raise RuntimeError(
                f"Curriculum term '{term_name}' has no state: this env was built with eval_mode=True, and "
                "CurriculumManager.compute() is a no-op at eval, so no term computes and nothing is ever "
                "cached — for the whole rollout, not just this step.\n"
                f"Fix: read the level off the term itself — curriculum_manager.get_term('{term_name}'), a "
                f"{self._term_label(term_name)} — which holds the configured level in training and at eval "
                "alike; that is what ObjectVelocityDisturbance._adaptive_scale does.\n"
                "Do NOT catch this and fall back to a default: the default would stand for every step of the "
                "eval, grading it at a difficulty no training step used, with nothing logged and nothing "
                "raised. Closing that silent fallback is why this raises at all."
            )
        if self._computing is not None:
            raise RuntimeError(
                f"Curriculum term '{term_name}' has not computed yet, so it has no state to read.\n"
                f"curriculum_options order: {', '.join(self._term_names)}\n"
                f"'{self._computing}' is reading '{term_name}', but '{term_name}' is declared after it\n"
                f"Fix: move '{term_name}' above '{self._computing}' in curriculum_options — "
                "CurriculumManager computes terms in declaration order, so a producer declared after its "
                "consumer would feed it a value one control step stale."
            )
        raise RuntimeError(
            f"Curriculum term '{term_name}' has no state yet: this read happens before the run's first "
            "CurriculumManager.compute(), which is step phase 16 (docs/guides/environments.md).\n"
            "reading from: outside the curriculum compute loop — build(), the first reset(), or task setup\n"
            f"Fix: read the term's own level off curriculum_manager.get_term('{term_name}'), a "
            f"{self._term_label(term_name)}, which is valid from build() onward. get_state() only ever "
            "serves what a term has already published."
        )

    def _term_label(self, term_name: str) -> str:
        """A greppable symbol for the term: its class name, or ``func()`` for a function term.

        Error messages carry this rather than only the config key so the reader — increasingly an
        agent working from a traceback — has the exact symbol to open, with no second lookup.

        Total by construction. Code that builds an error message must not be able to raise one of
        its own: an ``AttributeError`` from here would replace the diagnostic the caller needs with
        a traceback into the diagnostic itself, which is strictly worse than no label at all.
        """
        term = self._terms[term_name]
        if isinstance(term, ManagerTermFuncWrapperBase):
            return f"{getattr(term._func, '__name__', '<function>')}()"
        return type(term).__name__
