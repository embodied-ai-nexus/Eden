"""Event manager for applying randomization and other event-driven changes."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterator, Sequence

import torch
from eden.utils.format import format_table

from eden.constants import EventMode
from eden.managers.base import ConfigurableTermBase, ConfigurableTermFuncWrapper, ManagerBase, ManagerTermBase
from eden.options.managers.events import EventManagerOptions, EventTermOptions
from eden.utils.common import ConfigurableMixin
from eden.utils.misc import assign_over_envs, sanitize_envs_idx
from eden.utils.registry import TermRegistry


# Define proxies for fast lookup
_STARTUP, _RESET, _INTERVAL = EventMode


class _EventTermFields(ConfigurableMixin):
    """The configurable fields every event term has, class or function."""

    mode: EventMode = _RESET
    interval_range_s: tuple[float, float] | None = None
    is_global_time: bool = False
    min_step_count_between_reset: int = 0
    priority: int = 0


class EventTerm(ConfigurableTermBase[EventTermOptions], _EventTermFields):
    """Base class for event terms."""

    @classmethod
    def compute_batch(cls, terms: Sequence[EventTerm], envs_idx: slice | torch.Tensor | None) -> None:
        """Fire ``terms``, RESET terms of this kind adjacent in priority order, on the same ``envs_idx``.

        The default fires them one by one. A kind whose terms each write simulator state overrides it
        to write once for all of them (the reference-state-initialization terms: one Genesis write and
        one forward-kinematics pass per reset, not one per entity). The event manager hands a run
        of adjacent RESET terms to one call when each term's class takes both ``compute`` and
        ``compute_batch`` from the same class (a subclass that overrides ``compute`` alone runs alone)
        and none sets ``min_step_count_between_reset`` (so every term fires on exactly ``envs_idx``). A
        throttle set on a batched term after the build splits its batch back into single terms.
        """
        for term in terms:
            term.compute(envs_idx=envs_idx)


def _batch_owner(term: EventTerm | EventTermFuncWrapper) -> type[EventTerm] | None:
    """The class whose :meth:`EventTerm.compute_batch` fires ``term`` together with its neighbours, or None.

    None for a function term, a throttled term, a kind that keeps the default ``compute_batch``, and a
    subclass whose ``compute`` overrides the one its ``compute_batch`` was written for.
    """
    if not isinstance(term, EventTerm) or term.min_step_count_between_reset != 0:
        return None
    # The rule: the class that defines the compute_batch this term inherits must also be (or derive
    # from) the class that defines its compute — so a subclass overriding compute alone is left out.
    mro = type(term).__mro__
    batch_owner = next(klass for klass in mro if "compute_batch" in vars(klass))
    compute_owner = next(klass for klass in mro if "compute" in vars(klass))
    if batch_owner is EventTerm or not issubclass(batch_owner, compute_owner):
        return None
    return batch_owner


class EventTermFuncWrapper(ConfigurableTermFuncWrapper[EventTermOptions], _EventTermFields):
    """Base class for event terms defined as function."""


EVENT_TERM_REGISTRY = TermRegistry(
    "EVENT_TERM", EventTerm, EventTermFuncWrapper, populated_by=("eden.managers.terms.events",)
)


@dataclass
class _IntervalTermState:
    """An interval-mode term bundled with its countdown buffer.

    ``time_left`` is shape ``(1,)`` for global-time terms and ``(num_envs,)``
    otherwise. Bundling the term with its buffer makes the coupling structural:
    the term and its countdown live in one object, so no sort can misalign them
    the way two lists paired by index could be.

    The two shapes deliberately live on different devices, and this is the one place
    that says why. The per-env buffer is consumed as a mask (``time_left < 1e-6``
    indexes the envs to fire) so it must sit on ``self.device`` alongside everything
    it indexes. The global-time buffer is consumed by a host-side ``if``, which on a
    CUDA tensor forces a device-to-host sync on *every* env step — so it stays on the
    CPU, and is sampled there too, leaving no cross-device copy in the assignment.
    Do not "unify" them.
    """

    term: EventTerm | EventTermFuncWrapper
    time_left: torch.Tensor


@dataclass
class _ResetTermState:
    """A reset-mode term bundled with its throttle bookkeeping.

    ``last_triggered_step`` (per-env int32) and ``triggered_once`` (per-env
    bool) enforce ``min_step_count_between_reset``. Bundled with the term for
    the same reason as :class:`_IntervalTermState`.
    """

    term: EventTerm | EventTermFuncWrapper
    last_triggered_step: torch.Tensor
    triggered_once: torch.Tensor


@dataclass(eq=False)
class ResetTermBatch:
    """RESET terms adjacent in priority order that one :meth:`EventTerm.compute_batch` fires together.

    Every batchable term belongs to one, alone or with its neighbours. The event manager fires it
    through :meth:`compute`, never through the terms' own ``compute``, so a per-term timer has to
    wrap this instead (``RLStepProfiler`` does; see :meth:`EventManager.iter_reset_batches`).
    """

    owner: type[EventTerm]
    names: list[str]
    states: list[_ResetTermState]

    @property
    def terms(self) -> list[EventTerm]:
        return [state.term for state in self.states]

    def compute(self, envs_idx: slice | torch.Tensor) -> None:
        self.owner.compute_batch(self.terms, envs_idx)


class EventManager(ManagerBase[EventManagerOptions]):
    _registry = EVENT_TERM_REGISTRY

    def summary(self) -> str:
        msg = f"<EventManager> contains {len(self._mode_term_names)} active terms.\n"
        for mode in self._mode_term_names.keys():
            terms = enumerate(zip(self._mode_term_names[mode], self._mode_terms[mode], strict=True))
            if mode == _INTERVAL:
                field_names = ["Index", "Name", "Priority", "Interval time range (s)"]
                rows = ([index, name, term.priority, term.interval_range_s] for index, (name, term) in terms)
            else:
                field_names = ["Index", "Name", "Priority"]
                rows = ([index, name, term.priority] for index, (name, term) in terms)
            msg += format_table(f"Active Event Terms in Mode: '{mode}'", field_names, rows, align={"Name": "l"})
            msg += "\n"
        if self._domain_randomization_fields:
            msg += format_table(
                "Domain Randomization Fields",
                ["Index", "Field Name"],
                enumerate(self._domain_randomization_fields),
                align={"Field Name": "l"},
            )
            msg += "\n"
        return msg

    @property
    def mode_term_names(self) -> dict[EventMode, list[str]]:
        """The active event term names keyed by mode, in priority order."""
        return self._mode_term_names

    @property
    def active_terms(self) -> list[str]:
        """The flat, de-duplicated union of all active term names across modes."""
        return self._flatten_unique(self._mode_term_names)

    def iter_terms(self) -> Iterator[tuple[EventMode, str, ManagerTermBase]]:
        """Yield ``(mode, name, term)`` for every active event term, in mode/priority order.

        Public accessor mirroring :meth:`~eden.managers.reward_manager.RewardManager.iter_terms`
        so callers can scan term objects without touching the private per-mode storage.
        """
        for mode in self._mode_term_names:
            for name, term in zip(self._mode_term_names[mode], self._mode_terms[mode], strict=True):
                yield mode, name, term

    def get_term(self, name: str) -> ManagerTermBase:
        """Return the active event term registered under config key ``name``, whatever its mode.

        For a term another one reads from, e.g. ``ResetFromStateBuffer.current_column`` serving a goal stored
        with the reset states.

        Raises
        ------
        KeyError
            If ``name`` is not an active term. The message lists the active ones, worded as the flat
            managers' ``get_term`` words it.
        """
        for _, term_name, term in self.iter_terms():
            if term_name == name:
                return term
        available = ", ".join(self.active_terms) or "none"
        raise KeyError(f"No active term named '{name}' in {type(self).__name__}. Available options: {available}")

    def iter_reset_batches(self) -> Iterator[ResetTermBatch]:
        """Yield every batch of RESET terms (see :class:`ResetTermBatch`), in priority order.

        Their terms' own ``compute`` is never called on a reset, so time the batch instead.
        """
        for run in self._reset_runs:
            if isinstance(run, ResetTermBatch):
                yield run

    @property
    def available_modes(self) -> list[EventMode]:
        return list(self._mode_term_names.keys())

    @property
    def domain_randomization_fields(self) -> list[str]:
        return self._domain_randomization_fields

    def reset(self, envs_idx: (slice | torch.Tensor) | None = None):
        for mode_terms in self._mode_reset_terms.values():
            for term in mode_terms:
                term.reset(envs_idx=envs_idx)
        envs_idx = sanitize_envs_idx(envs_idx, self._env.num_envs)
        for state in self._interval_states:
            if not isinstance(state.term, EventTerm):
                continue
            if not state.term.is_global_time:
                assert state.term.interval_range_s is not None
                lower, upper = state.term.interval_range_s
                sampled_interval = torch.rand(self._env.num_envs, device=self.device) * (upper - lower) + lower
                assign_over_envs(state.time_left, envs_idx, sampled_interval)
        return {}

    def iter_post_reset_terms(self) -> Iterator[ManagerTermBase]:
        """Every event term, in mode/priority order (a term shared across modes repeats)."""
        return (term for _, _, term in self.iter_terms())

    def compute(
        self,
        mode: EventMode,
        envs_idx: (slice | torch.Tensor) | None = None,
        dt: float | None = None,
        global_env_step_count: int | None = None,
        n_envs: int | None = None,
    ):
        # ``n_envs``: how many envs ``envs_idx`` selects, when the caller has already counted them (the
        # auto-reset has — it is the gate in ``EnvBase._reset_scene_and_events``). Counting a bool mask
        # is a GPU->CPU sync, so a known count is not paid for twice.
        # Dispatch per mode rather than re-checking ``mode`` for every term. The
        # per-term bodies live in ``_step_{interval,reset}_term`` so each mode's
        # trigger logic reads independently.
        if mode == _INTERVAL:
            if dt is None:
                raise ValueError(f"Event mode '{mode}' requires the time-step of the environment.")
            if envs_idx is not None:
                raise ValueError(
                    f"Event mode '{mode}' does not require environment indices. This is an undefined behavior"
                    " as the environment indices are computed based on the time left for each environment."
                )
            for state in self._interval_states:
                self._step_interval_term(state, dt)
        elif mode == _RESET:
            if global_env_step_count is None:
                raise ValueError(f"Event mode '{mode}' requires the total number of environment steps to be provided.")
            if n_envs is None:
                envs_idx, n_envs = sanitize_envs_idx(envs_idx, self._env.num_envs, return_n_envs=True)
            else:
                envs_idx = sanitize_envs_idx(envs_idx, self._env.num_envs)
            if n_envs == 0:
                return
            for run in self._reset_runs:
                if isinstance(run, ResetTermBatch):
                    self._step_reset_batch(run, envs_idx, global_env_step_count)
                else:
                    self._step_reset_term(run, envs_idx, global_env_step_count)
        elif mode == _STARTUP:
            for term in self._mode_terms[_STARTUP]:
                term.compute(envs_idx=envs_idx)
        else:
            raise ValueError(f"Invalid event mode: {mode}")

    def _step_interval_term(self, state: _IntervalTermState, dt: float) -> None:
        """Tick one interval term's countdown and fire it (and resample its interval) on expiry."""
        term = state.term
        state.time_left -= dt  # in-place countdown
        assert term.interval_range_s is not None
        lower, upper = term.interval_range_s
        if term.is_global_time:
            # CPU countdown, sampled on the CPU — see _IntervalTermState for why.
            if state.time_left < 1e-6:
                sampled_interval = torch.rand(1, device="cpu") * (upper - lower) + lower
                state.time_left[:] = sampled_interval
                term.compute()
        else:
            valid_envs_mask = state.time_left < 1e-6
            # Always runs: no `.any()` gate, and the resample is a masked write, not a gather — both
            # would sync every step, and with thousands of envs some env is almost always due.
            sampled_time = torch.rand(self._env.num_envs, device=self.device) * (upper - lower) + lower
            assign_over_envs(state.time_left, valid_envs_mask, sampled_time)
            term.compute(envs_idx=valid_envs_mask)

    def _step_reset_term(
        self, state: _ResetTermState, envs_idx: slice | torch.Tensor, global_env_step_count: int
    ) -> None:
        """Fire one reset term on the resetting envs, honoring its ``min_step_count_between_reset``."""
        term = state.term
        min_step_count = term.min_step_count_between_reset
        if min_step_count == 0:
            self._mark_fired(state, envs_idx, global_env_step_count)
            term.compute(envs_idx=envs_idx)
        else:
            last_triggered_step = state.last_triggered_step[envs_idx]
            triggered_at_least_once = state.triggered_once[envs_idx]
            steps_since_triggered = global_env_step_count - last_triggered_step
            valid_trigger = steps_since_triggered >= min_step_count
            valid_trigger |= (last_triggered_step == 0) & ~triggered_at_least_once

            valid_envs_mask = self._expand_to_full_mask(envs_idx, valid_trigger)

            # Always execute — bool-mask ops are no-ops for all-False
            self._mark_fired(state, valid_envs_mask, global_env_step_count)
            term.compute(envs_idx=valid_envs_mask)

    def _step_reset_batch(
        self, batch: ResetTermBatch, envs_idx: slice | torch.Tensor, global_env_step_count: int
    ) -> None:
        """Fire a batch of RESET terms through one ``compute_batch`` (see :meth:`EventTerm.compute_batch`).

        Only terms unthrottled at build join a batch, so each fires on exactly ``envs_idx``, as
        :meth:`_step_reset_term` would fire it. A throttle set on one of them since is read here, as
        :meth:`_step_reset_term` reads it every reset: the batch then fires term by term through it.
        """
        if any(state.term.min_step_count_between_reset != 0 for state in batch.states):
            for state in batch.states:
                self._step_reset_term(state, envs_idx, global_env_step_count)
            return
        for state in batch.states:
            self._mark_fired(state, envs_idx, global_env_step_count)
        batch.compute(envs_idx)

    @staticmethod
    def _mark_fired(state: _ResetTermState, envs_idx: slice | torch.Tensor, global_env_step_count: int) -> None:
        """Record that ``state``'s term fired in ``envs_idx`` at this step (its throttle bookkeeping)."""
        state.last_triggered_step[envs_idx] = global_env_step_count
        state.triggered_once[envs_idx] = True

    def _expand_to_full_mask(self, envs_idx: slice | torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """Scatter ``valid`` (defined over ``envs_idx``) into the shared (num_envs,) scratch
        mask and return it.

        The returned tensor is reused on every call — consume it before the next call.
        Returning a full bool mask (vs ``envs_idx[valid]``) avoids the GPU->CPU sync of
        variable-size boolean indexing; bool-mask writes are no-ops when all-False, so no
        ``.any()`` guard is needed.
        """
        self._env_mask_buf.zero_()
        self._env_mask_buf[envs_idx] = valid
        return self._env_mask_buf

    def _prepare_terms(self) -> None:
        # Temporary storage for terms with their priorities before sorting
        temp_mode_terms: dict[EventMode, list[tuple[int, EventTerm | EventTermFuncWrapper, str]]] = defaultdict(list)
        temp_mode_reset_terms: dict[EventMode, list[tuple[int, EventTerm, str]]] = defaultdict(list)

        # Per-term state, each bundling a term with its own buffer(s) and the
        # priority used to order it (see _IntervalTermState / _ResetTermState).
        temp_interval_states: list[tuple[int, _IntervalTermState]] = []
        temp_reset_states: list[tuple[int, _ResetTermState, str]] = []

        self._domain_randomization_fields: list[str] = list()

        for term_name in self._options.term_keys():
            term_option: EventTermOptions = getattr(self._options, term_name)
            term: EventTerm | EventTermFuncWrapper = self._build_term(term_name)

            temp_mode_terms[term_option.mode].append((term_option.priority, term, term_name))
            if isinstance(term, EventTerm):
                temp_mode_reset_terms[term_option.mode].append((term_option.priority, term, term_name))

            if term_option.mode == _INTERVAL:
                if term_option.interval_range_s is None:
                    raise ValueError(
                        f"Event term '{term_name}' has mode 'interval' but 'interval_range_s' is not specified."
                    )
                lower, upper = term_option.interval_range_s
                if term_option.is_global_time:
                    # CPU on purpose — read by a host-side `if`, see _IntervalTermState.
                    time_left = torch.rand(1, device="cpu") * (upper - lower) + lower
                else:
                    time_left = torch.rand(self.num_envs, device=self.device) * (upper - lower) + lower
                temp_interval_states.append((term_option.priority, _IntervalTermState(term, time_left)))
            elif term_option.mode == _RESET:
                step_count = torch.zeros(self.num_envs, device=self.device, dtype=torch.int32)
                no_trigger = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
                temp_reset_states.append(
                    (term_option.priority, _ResetTermState(term, step_count, no_trigger), term_name)
                )

        # Sort terms by priority (lower number = higher priority = runs first) and
        # create final structures. The state lists are sorted by the same key as
        # _mode_terms, so the term/buffer pairing is preserved without any
        # positional index shared across separate lists.
        self._mode_term_names: dict[EventMode, list[str]] = defaultdict(list)
        self._mode_terms: dict[EventMode, list[EventTerm | EventTermFuncWrapper]] = defaultdict(list)
        self._mode_reset_terms: dict[EventMode, list[EventTerm]] = defaultdict(list)

        for mode in temp_mode_terms:
            sorted_terms = sorted(temp_mode_terms[mode], key=lambda x: x[0])
            for priority, term, name in sorted_terms:
                self._mode_term_names[mode].append(name)
                self._mode_terms[mode].append(term)

        for mode in temp_mode_reset_terms:
            sorted_reset_terms = sorted(temp_mode_reset_terms[mode], key=lambda x: x[0])
            for priority, term, name in sorted_reset_terms:
                self._mode_reset_terms[mode].append(term)

        self._interval_states: list[_IntervalTermState] = [
            s for _, s in sorted(temp_interval_states, key=lambda x: x[0])
        ]
        # The reset states in priority order. A batchable term joins the batch before it when both
        # fire through the same compute_batch, and opens a new one otherwise (see EventTerm.compute_batch);
        # any other term stays a bare state and fires through its own compute.
        self._reset_runs: list[_ResetTermState | ResetTermBatch] = []
        for _, state, name in sorted(temp_reset_states, key=lambda x: x[0]):
            owner = _batch_owner(state.term)
            last = self._reset_runs[-1] if self._reset_runs else None
            if owner is None:
                self._reset_runs.append(state)
            elif isinstance(last, ResetTermBatch) and last.owner is owner:
                last.names.append(name)
                last.states.append(state)
            else:
                self._reset_runs.append(ResetTermBatch(owner, [name], [state]))

        # Reusable scratch: scatter a subset selector (slice/tensor) into a full (num_envs,)
        # bool mask in _expand_to_full_mask. Throwaway per-call scratch, NOT persistent state.
        self._env_mask_buf = torch.zeros(self._env.num_envs, dtype=torch.bool, device=self.device)
