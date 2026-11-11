"""Which reset configuration each env starts its next episode from.

A *reset configuration* is one row of a fixed, indexed set of episode starts the task can reset
to: a stored simulator state (robot and object poses), a start state paired with a goal, a scene
variant such as an object instance or a terrain patch. What a row means is the consumer's business;
the sampler sees only indices. Two roles meet at reset time and this module is the seam between them:

- the **consumer** -- a RESET event term that owns the set, knows its size, and turns an index
  into a reset (writes the state, places the robot, sets the goal);
- the **sampler** -- whatever decides which index each resetting env gets. Uniform by default;
  an adaptive sampler (Success Guided Sampling, a curriculum term) when the task opts in.

They meet through :class:`ResetConfigSampler`. The consumer resolves its sampler in ``build()``
(:func:`resolve_reset_config_sampler`), tells it the set size once (:meth:`ResetConfigSampler.bind`),
and asks for indices from its RESET ``compute()`` (:meth:`ResetConfigSampler.sample`).

Why a RESET event drives the sampler rather than ``CurriculumManager.reset``: the scene and the
RESET events run *before* any manager reset (``EnvBase._reset_idx``), so by the time a curriculum
term's own ``reset()`` runs, the env has already been placed.

The assignment is returned as the full ``(num_envs,)`` buffer, not as the selected rows: a reset
selector may be a bool mask, and compacting one to its selected rows is a GPU->CPU sync on every
step that resets anything. Consumers index the buffer with the same ``envs_idx``
they were handed.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch

from eden.utils.misc import is_env_mask

if TYPE_CHECKING:
    from eden.envs.base import EnvBase


class ResetConfigSampler(ABC):
    """Chooses a reset-configuration index for every env being reset.

    Subclassed by :class:`UniformResetConfigSampler` and by adaptive samplers, which are curriculum terms
    (``class SuccessGuidedSampling(CurriculumTerm, ResetConfigSampler)``). A consumer calls :meth:`bind`
    exactly once from its ``build()``, then :meth:`sample` from every RESET ``compute()``.

    The base owns everything every sampler shares: bind validation, the ``(num_envs,)`` assignment buffer,
    which envs' next reset is a first spawn (:attr:`first_spawn`), and turning a reset selector into a mask. A
    subclass implements :meth:`_draw` -- one proposed configuration per env, of which only the selected rows are
    kept -- and may allocate per-configuration state in :meth:`_on_bind`. It provides ``num_envs`` and ``device``,
    which a manager term already has.

    The base also owns the checkpoint: :meth:`state_dict` / :meth:`load_state_dict` record and check the size of
    the configuration set, and a subclass with state of its own adds it in :meth:`_sampler_state` /
    :meth:`_load_sampler_state`. Every load ends with every env's next reset a first spawn, whatever the subclass:
    the episodes running at the load started from the pre-load state, and grading them into the loaded one would
    mix the two. Both methods are final: a subclass that overrides either one, or whose MRO resolves either one
    to another class first (a default added to a curriculum-term base, say), raises ``TypeError`` when the class
    is defined, instead of silently skipping the size check, the first-spawn reset or the sampler's own state.

    The bound state lives in class-level defaults rather than in an ``__init__``: a curriculum term's
    ``__init__`` chain calls its bases explicitly and never reaches this one.
    """

    _num_configs: int | None = None
    _consumer: str | None = None
    _assignment: torch.Tensor | None = None
    _first_spawn: torch.Tensor | None = None
    _selected: torch.Tensor | None = None

    def __init_subclass__(cls, **kwargs) -> None:
        """Refuse a subclass on which ``state_dict`` / ``load_state_dict`` are not the base's own."""
        super().__init_subclass__(**kwargs)
        for name in ("state_dict", "load_state_dict"):
            if getattr(cls, name) is getattr(ResetConfigSampler, name):
                continue
            owner = next(base for base in cls.__mro__ if name in vars(base))
            fix = (
                f"move its state into _sampler_state / _load_sampler_state and delete {name}() from {cls.__name__}"
                if issubclass(owner, ResetConfigSampler)
                else f"remove that {name}() from {owner.__qualname__}, or re-point it in {cls.__name__} with "
                f"`{name} = ResetConfigSampler.{name}`"
            )
            raise TypeError(
                f"{cls.__name__}: {name}() resolves to {owner.__qualname__}.{name}, not ResetConfigSampler.{name}, so "
                "a load would skip the configuration-set size check and the first-spawn reset (or the save would drop "
                f"the sampler's state). Fix: {fix}."
            )

    @property
    @abstractmethod
    def num_envs(self) -> int:
        """The number of parallel environments."""

    @property
    @abstractmethod
    def device(self) -> torch.device | str:
        """The torch device the assignment lives on."""

    def bind(self, num_configs: int, consumer: str) -> None:
        """Fix the size of the configuration set this sampler draws from, then run :meth:`_on_bind`.

        ``consumer`` labels the binding term in error messages, e.g. ``ResetFromStateBuffer(path='…')``.
        A sampler bound twice raises: two consumers sharing one sampler would draw indices into two
        different sets while the sampler's statistics conflate them.
        """
        label = type(self).__name__
        if self._consumer is not None:
            raise RuntimeError(
                f"{label} is already bound to reset consumer '{self._consumer}'; '{consumer}' tried to bind it "
                "too. One sampler serves one configuration set -- give each consumer its own sampler term."
            )
        if not isinstance(num_configs, int) or isinstance(num_configs, bool) or num_configs < 1:
            raise ValueError(
                f"{label}.bind(): '{consumer}' passed num_configs={num_configs!r}; expected a positive int "
                "(the number of rows in its reset-configuration set)."
            )
        self._num_configs = num_configs
        self._consumer = consumer
        # Normal tensors, not inference tensors: sample() runs both inside and outside torch.inference_mode.
        with torch.inference_mode(False):
            self._assignment = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self._first_spawn = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
            self._selected = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._on_bind(num_configs)

    def _on_bind(self, num_configs: int) -> None:
        """Allocate per-configuration state. Called once, from :meth:`bind`, after validation; a no-op by default."""

    @property
    def num_configs(self) -> int:
        """Size of the bound configuration set."""
        self._require_bound("num_configs")
        return self._num_configs

    @property
    def assignment(self) -> torch.Tensor:
        """``(num_envs,)`` long: the configuration each env's current episode started from."""
        self._require_bound("assignment")
        return self._assignment

    @property
    def first_spawn(self) -> torch.Tensor:
        """``(num_envs,)`` bool: envs whose next reset finishes no episode this sampler should learn from.

        True for every env until its first :meth:`sample`, and again for every env after a
        :meth:`load_state_dict`. Where it is False, :attr:`assignment` is the configuration the env's running
        episode started from, drawn by this sampler's current state.
        """
        self._require_bound("first_spawn")
        return self._first_spawn

    def sample(self, envs_idx: slice | torch.Tensor) -> torch.Tensor:
        """Assign a configuration to each env in ``envs_idx`` and return the assignment buffer.

        ``envs_idx`` is a sanitized selector (``eden.utils.misc.sanitize_envs_idx``): a slice, a
        long index tensor, or a ``(num_envs,)`` bool mask. Returns the sampler's ``(num_envs,)``
        long buffer of current assignments, with only the ``envs_idx`` rows changed by this call.
        The buffer is live -- the consumer must index it now, not keep it for later. The selected
        envs leave :attr:`first_spawn` after the draw, so :meth:`_draw` still sees which are first spawns.
        """
        self._require_bound("sample()")
        selected = self._selection_mask(envs_idx)
        # Proposed for every env and blended in under the mask, so no selector costs a GPU->CPU sync.
        torch.where(selected, self._draw(selected), self._assignment, out=self._assignment)
        self._first_spawn.logical_and_(selected.logical_not())
        return self._assignment

    @abstractmethod
    def _draw(self, selected: torch.Tensor) -> torch.Tensor:
        """Propose a configuration for every env; :meth:`sample` keeps the ``selected`` rows.

        ``selected`` is the ``(num_envs,)`` bool mask of envs being reset. For the envs not in :attr:`first_spawn`,
        :attr:`assignment` still holds the configuration each just finished, so this is where a sampler grades
        the ended episode before choosing the next; an env in :attr:`first_spawn` finished nothing to grade.
        Returns a ``(num_envs,)`` long tensor in ``[0, num_configs)``.
        """

    def state_dict(self) -> dict:
        """The sampler's checkpoint: the size of the set it was bound to, plus :meth:`_sampler_state`."""
        self._require_bound("state_dict()")
        return {"num_configs": self._num_configs, **self._sampler_state()}

    def load_state_dict(self, state: dict) -> None:
        """Restore a :meth:`state_dict`, then make every env's next reset a first spawn.

        Raises if the checkpoint was saved for a configuration set of another size: its per-configuration state
        would index configurations it was never recorded on. The episodes running now started from the pre-load
        state, so none of them is graded into the loaded one.
        """
        self._require_bound("load_state_dict()")
        saved = state.get("num_configs")
        if saved != self._num_configs:
            raise ValueError(
                f"{type(self).__name__}: the checkpoint was saved for a set of {saved} reset configurations, but "
                f"{self._consumer} bound this one to {self._num_configs}. Fix: resume with the task the checkpoint "
                "was trained on. Do NOT drop this term's checkpoint state to get past it: its statistics would be "
                "read against configurations they were never recorded on."
            )
        self._load_sampler_state(state)
        self._first_spawn.fill_(True)

    def _sampler_state(self) -> dict:
        """A subclass's own checkpoint entries, beside ``num_configs``; none by default."""
        return {}

    def _load_sampler_state(self, state: dict) -> None:
        """Restore what :meth:`_sampler_state` saved; ``num_configs`` is already checked. A no-op by default."""

    def _selection_mask(self, envs_idx: slice | torch.Tensor) -> torch.Tensor:
        if is_env_mask(envs_idx):
            return envs_idx
        self._selected.zero_()
        if isinstance(envs_idx, slice):
            self._selected[envs_idx] = True
        else:
            # Not ``mask[index] = True``: on CUDA that assignment synchronizes; index_fill_ does not.
            self._selected.index_fill_(0, envs_idx.long(), True)
        return self._selected

    def _require_bound(self, what: str) -> None:
        """Raise if no consumer has bound this sampler yet; ``what`` names the access, for the message."""
        if self._num_configs is None:
            raise RuntimeError(
                f"{type(self).__name__} reached {what} unbound: no reset event has bound it. A sampler draws "
                "from a configuration set owned by a RESET event, which calls bind(num_configs, consumer=<its "
                "term name>) from its own build().\n"
                "Fix: for a sampler term, name it in the `sampler` parameter of that RESET event, or remove the "
                "term; for a sampler a consumer constructs itself, call bind() from the consumer's build()."
            )

    def _require_bound_at_build(self, env: EnvBase) -> None:
        """Fail env construction if no consumer binds this sampler; call it from a sampler term's ``build()``.

        A sampler term builds before the event manager whose RESET event binds it, so it cannot check in
        its own ``build()``; the post-build hooks run once every manager is built.
        """
        env.register_post_build_hook(lambda: self._require_bound("the end of env construction"))


class UniformResetConfigSampler(ResetConfigSampler):
    """Draws every reset configuration with equal probability -- the default when a task opts out.

    Not a manager term: a consumer with no sampler configured constructs one of these itself, so
    the uniform path has no config entry, no logging and no checkpoint state. Draws from the global
    ``torch`` RNG, like every other reset-time randomization, so seeding the run seeds it.
    """

    def __init__(self, num_envs: int, device: torch.device | str) -> None:
        self._num_envs = num_envs
        self._device = device

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def device(self) -> torch.device | str:
        return self._device

    def _draw(self, selected: torch.Tensor) -> torch.Tensor:
        return torch.randint(self._num_configs, (self._num_envs,), device=self._device)


class CategoryWeightedSampler(ResetConfigSampler):
    """Draws a category by weight, then a configuration uniformly within it (OmniReset's reset-family mix).

    ``category`` labels each configuration (``(num_configs,)`` long, values index ``weights``); a category with
    weight 0 is never drawn, and its configurations with it. Unlike uniform sampling over the whole set, a category's
    share does not grow with its size. Like :class:`UniformResetConfigSampler`, a consumer constructs it itself and
    it draws from the global ``torch`` RNG. Sync-free: the configurations are sorted by category once, at bind, and
    again only on :meth:`relabel`.
    """

    def __init__(
        self, num_envs: int, device: torch.device | str, category: torch.Tensor, weights: torch.Tensor
    ) -> None:
        self._num_envs = num_envs
        self._device = device
        self._weights = weights.to(device=device, dtype=torch.float32)
        if (
            self._weights.ndim != 1
            or not torch.isfinite(self._weights).all()
            or (self._weights < 0).any()
            or not (self._weights > 0).any()
        ):
            raise ValueError(
                f"CategoryWeightedSampler weights must be finite and non-negative with at least one positive, got "
                f"{self._weights.tolist()}."
            )
        self._label(category)

    def relabel(self, category: torch.Tensor) -> None:
        """Take new labels for the same configurations, e.g. after ``ResetStateBuffer.write_rows`` changed some rows'.

        Re-checks the labels and re-sorts the configurations by them: a few device-to-host reads, so call it when the
        labels change, not on every reset.
        """
        self._label(category)
        if self._num_configs is not None:
            self._on_bind(self._num_configs)

    def _label(self, category: torch.Tensor) -> None:
        self._category = category.to(device=self._device, dtype=torch.long)
        counts = torch.bincount(self._category, minlength=len(self._weights))
        empty = [i for i in range(len(self._weights)) if self._weights[i] > 0 and counts[i] == 0]
        if len(counts) > len(self._weights) or empty:
            raise ValueError(
                f"CategoryWeightedSampler: categories with configurations {torch.nonzero(counts).flatten().tolist()} "
                f"do not match weights for {len(self._weights)} categories (positive weight but no configuration: "
                f"{empty})."
            )

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def device(self) -> torch.device | str:
        return self._device

    def _on_bind(self, num_configs: int) -> None:
        if self._category.shape != (num_configs,):
            raise ValueError(
                f"CategoryWeightedSampler: {tuple(self._category.shape)} category labels for {num_configs} configurations."
            )
        self._order = torch.argsort(self._category, stable=True)
        self._count = torch.bincount(self._category, minlength=len(self._weights))
        self._start = torch.cumsum(self._count, 0) - self._count

    def _draw(self, selected: torch.Tensor) -> torch.Tensor:
        category = torch.multinomial(self._weights, self._num_envs, replacement=True)
        offset = (torch.rand(self._num_envs, device=self._device) * self._count[category]).long()
        offset = torch.minimum(offset, self._count[category] - 1)  # rand() can round up to 1.0 in float32
        return self._order[self._start[category] + offset]


def resolve_reset_config_sampler(env: EnvBase, sampler_name: str | None, consumer: str) -> ResetConfigSampler:
    """The sampler a reset consumer named, or a fresh uniform one when it named none.

    ``sampler_name`` is a curriculum term's config key. Resolved from the consumer's ``build()``:
    the curriculum manager is constructed before the event manager, so the term already exists, and
    a wrong name raises at env construction rather than on the first reset.
    """
    if sampler_name is None:
        return UniformResetConfigSampler(env.num_envs, env.device)
    manager = getattr(env, "curriculum_manager", None)  # absent on a plain EnvBase
    if manager is None:
        raise ValueError(
            f"Reset consumer '{consumer}' names sampler='{sampler_name}', but this env has no curriculum "
            f"manager. Fix: add '{sampler_name}' to curriculum_options, or drop sampler= to sample "
            "configurations uniformly."
        )
    term = manager.get_term(sampler_name)  # KeyError listing the declared terms on a typo
    if not isinstance(term, ResetConfigSampler):
        raise TypeError(
            f"Reset consumer '{consumer}' names sampler='{sampler_name}', but that curriculum term is a "
            f"{type(term).__name__}, which is not a ResetConfigSampler subclass. "
            "Fix: point sampler= at an adaptive reset sampler such as SuccessGuidedSampling."
        )
    return term
