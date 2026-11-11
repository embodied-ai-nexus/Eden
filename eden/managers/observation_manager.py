"""Observation manager for computing observations."""

from __future__ import annotations

from collections import OrderedDict, defaultdict
from types import FunctionType
from typing import TYPE_CHECKING, Iterator, TypeAlias

import numpy as np
import torch
from eden.utils.format import format_table

import eden as en
from eden.managers.base import ConfigurableTermBase, ConfigurableTermFuncWrapper, ManagerBase
from eden.managers.modifiers.base import NOISE_MODEL_REGISTRY, OBSERVATION_MODIFIER_REGISTRY
from eden.options.managers.observations import (
    ObservationGroupOptions,
    ObservationManagerOptions,
    ObservationTermOptions,
)
from eden.utils.buffers.circular_buffer import CircularBuffer
from eden.utils.common import ConfigurableMixin
from eden.utils.misc import sanitize_envs_idx
from eden.utils.torch import resolve_scalar_or_vector
from eden.utils.registry import TermRegistry

if TYPE_CHECKING:
    from eden.envs.base import EnvBase
    from eden.managers.modifiers.base import NoiseModel, ObservationModifier


def _resolve_term_scale(
    term_name: str,
    group_name: str,
    scale: float | list[float],
    obs_dims: tuple[int, ...],
    device: torch.device | str,
) -> float | torch.Tensor | None:
    """Resolve ``ObservationTermOptions.scale`` into its runtime form; ``None`` means identity.

    Delegates the length check and tensor construction to :func:`eden.utils.torch.resolve_scalar_or_vector`
    (shared with the task-space action terms); what is observation-specific and stays here is the
    identity short-circuit and the feature dim the vector is matched against.

    Anything that multiplies out to identity — the scalar ``1.0`` *or* an all-ones vector — resolves
    to ``None`` so both hot-path branches keep their existing short-circuit and pay nothing. A vector
    is matched against the feature dim: the last dimension of ``obs_dims``, the shape of the term's
    raw ``compute()`` output. Scale is applied before history stacking, so that is the raw feature
    dim, not the history-flattened one.

    ``obs_dims`` is passed whole rather than pre-indexed so an output with no feature axis can be
    rejected: a term returning ``(num_envs,)`` would otherwise have its *batch* dim taken as the
    feature dim, silently matching a vector scale against ``num_envs`` and scaling per-environment
    instead of per-component.

    Called once per term at build; the result is snapshotted onto ``term._resolved_scale``. See
    :class:`ObservationTerm` for what that means for post-build mutation of ``_options.scale``.
    """
    if isinstance(scale, (int, float)) and not isinstance(scale, bool):
        return None if scale == 1.0 else float(scale)
    # Gated on the value actually being a sequence, so a bad *type* falls through to
    # ``resolve_scalar_or_vector``'s type guard instead of being announced as a "vector scale".
    if isinstance(scale, (list, tuple)) and len(obs_dims) < 2:
        raise ValueError(
            f"Observation term '{term_name}' in group '{group_name}' has a vector scale, but the raw "
            f"compute() output has shape {obs_dims} — a single dim, which the manager reads as the batch "
            f"dim, so there is no feature dim to match the vector against. Use a scalar scale, or return "
            f"a (num_envs, ...) tensor."
        )
    resolved = resolve_scalar_or_vector(
        scale,
        obs_dims[-1],
        device,
        what=f"Observation term '{term_name}' in group '{group_name}' scale for feature dim {obs_dims[-1]}",
    )
    # An all-ones vector is identity: drop it so the hot path short-circuits like a scalar 1.0
    # instead of paying a broadcast multiply every step, forever.
    if bool(torch.all(resolved == 1.0)):
        return None
    return resolved


class _ObservationTermFields(ConfigurableMixin):
    """What every observation term has, class or function: the options-only post-compute fields and what
    the manager resolves from them at build.
    """

    _extra_option_params = ObservationTermOptions.POST_COMPUTE_FIELDS

    #: Constructed in :meth:`_init_post_compute_fields` from ``modifiers``, in order; empty when there are none.
    _modifiers: tuple[ObservationModifier, ...]
    #: Resolved once in :meth:`_init_post_compute_fields`; ``None`` when the term configures no noise.
    _noise: NoiseModel | None = None
    #: Resolved once in ``_instantiate_group_terms`` (see :func:`_resolve_term_scale`) against the term's
    #: warm-computed feature dim; ``None`` means identity — no multiply on the hot path — and is what it
    #: reads until then. Every term reaching ``compute_group`` has been through that call.
    _resolved_scale: float | torch.Tensor | None = None

    def _init_post_compute_fields(self, env: EnvBase) -> None:
        """Construct the term's modifiers and noise model; called from both term bases' ``__init__``.

        Constructed here, with the term, so a bad name or parameter fails as the term does. The modifiers are
        built in ``_build_term_modifiers`` once the term's output is known.
        """
        options = self._options
        if options.history_length == 1:
            en.logger.warning(
                f"History length is 1 for {type(self).__name__}, is this intended? (use history length 0 instead)"
            )
        modifiers = []
        for index, modifier_options in enumerate(options.modifiers):
            try:
                modifier_cls = OBSERVATION_MODIFIER_REGISTRY.get(modifier_options.name)
                modifiers.append(modifier_cls(env=env, options=modifier_options))
            except Exception as e:
                e.add_note(f"while constructing modifier {index} ('{modifier_options.name}')")
                raise
        self._modifiers = tuple(modifiers)
        if options.noise is not None:
            self._noise = NOISE_MODEL_REGISTRY.get(options.noise.name)(env=env, options=options.noise)

    @property
    def effective_scale(self) -> float | torch.Tensor | None:
        """The scale actually applied to this term's output, as resolved at build.

        Not the configured ``scale`` option: a vector is resolved to a device tensor, and anything that
        multiplies out to identity — the scalar ``1.0`` *or* an all-ones vector — resolves to ``None``,
        meaning the manager skips the multiply entirely. ``None`` before the term is built into a group.
        Mirrors :attr:`~eden.managers.terms.actions.joint_actions.JointAction.effective_scale`.
        """
        return self._resolved_scale


class ObservationTerm(ConfigurableTermBase[ObservationTermOptions], _ObservationTermFields):
    """Base class for observation terms.

    **``compute()`` is a read, not a tick.** The manager calls it more often than once per control
    step and never promises which call is "the" step: once per term at build to learn the output shape
    (before any ``reset()``), from ``EnvBase.reset()``, at phase 19 of ``RLEnvBase.step``, again at
    phase 12 for the final-observation snapshot when ``record_final_observations`` is on, from
    ``RslRlVecEnvWrapper.get_observations()`` outside the step, and from the deploy pre-flight probe.
    The cross-group raw cache goes the other way and may call ``compute()`` on only *one* of several
    identically configured instances. A term that advances state inside ``compute()`` — an integral,
    a counter, a latch — therefore double-counts under any of those, silently: the failure is a wrong
    number, not an exception.

    Per-step state lives on the term and moves in the lifecycle hooks instead: ``build()`` allocates
    the buffer, :meth:`post_physics_refresh` advances it exactly once per control step on the
    post-physics state (phase 7 — the same hook, epoch and reasoning as a command term's measurement
    state at 5, and after the metrics at 6 so a term may integrate one), ``reset(envs_idx)`` zeroes the
    rows of the envs that restart, ``post_reset(envs_idx)`` rebuilds what the post-reset scene
    determines, and ``compute()`` returns the buffer.
    :class:`~eden.managers.terms.observations.locomotion.VelocityTrackingErrorIntegral` is the shipped
    worked example. Function terms have none of these hooks — see :class:`ObservationTermFuncWrapper`.

    The post-compute fields (modifiers/noise/clip/scale/history) are options-only (see
    ``ObservationTermOptions.POST_COMPUTE_FIELDS``): ``configure()`` accepts them and stores them on the
    options, but they are NOT mirrored as term attributes — the manager applies them from outside so a
    ``compute()`` can't accidentally read (and be wrongly deduped on) a modifier value.

    **When each modifier is read.** ``clip`` and ``history_length`` are read off ``self._options`` on every
    step, so a post-build write to them takes effect immediately. ``noise`` and ``scale`` are *resolved once*
    at build — into ``self._noise`` and ``self._resolved_scale`` — because both need setup the hot path must
    not repeat (a noise model instance; a length check against the term's feature dim plus a device tensor).
    Writing ``term._options.scale`` after ``env.build()`` therefore has **no effect**: it is not the value the
    manager multiplies by, and nothing detects the divergence for you. Read what is actually applied off
    :attr:`effective_scale`; to retune it at runtime, call
    :meth:`~eden.managers.observation_manager.ObservationManager.set_term_scale`, which re-runs the feature-dim
    check. Modifiers stay the manager's business in both directions — that is what keeps them off this class.
    """

    def __init__(self, env: EnvBase, options: ObservationTermOptions):
        super().__init__(env=env, options=options)
        self._init_post_compute_fields(env)

    def post_physics_refresh(self) -> None:
        """Advance per-step state on the post-physics scene (phase 7 — the *grading* epoch).

        No-op by default. The one call per control step that a stateful term may count on: it runs
        after the physics sub-steps, after ``command_manager.post_physics_refresh`` (5) and
        ``metric_manager.compute`` (6) — so a term may integrate a refreshed command cache or a shared
        metric and see this step's value — and before termination/reward; and, unlike ``compute()``,
        exactly once, whatever else reads the observation this step. Accumulate here (``add_`` into a
        ``build()``-time buffer, in place), and return the buffer from ``compute()``.

        Running *above* the auto-reset (phase 13) is what gives a stateful observation the semantics
        one expects of it: the final-observation snapshot at phase 12 carries this step's increment,
        ``reset()`` then zeroes the envs that restart, and the fresh episode's first observation at
        phase 19 reads zero rather than one post-reset increment. ``EnvBase.reset()`` and the build-time
        shape probe run no phase 7 at all, so a term reads as zero there too.

        Invoked by :meth:`ObservationManager.post_physics_refresh` from ``RLEnvBase.step`` and
        ``EnvBase.step``, and, on hardware, once per control tick from
        ``DeploymentBase.state_to_observation`` after the robot state is synced (a command term's
        ``post_physics_refresh`` does **not** run there; this one does, because the policy's
        observations are the hardware path). Never from ``compute()``, and never from a reset.
        """


class ObservationTermFuncWrapper(ConfigurableTermFuncWrapper[ObservationTermOptions], _ObservationTermFields):
    """Base class for observation terms defined as function. See :class:`ObservationTerm` for the modifier
    fields (options-only, not mirrored as attributes), for when each one is read, and for
    :attr:`effective_scale`.

    **Stateless by construction.** A function term is a pure forward of env state: the manager never
    dispatches ``reset()``, ``post_reset()`` or ``post_physics_refresh()`` to it (the inherited ``reset``
    and ``post_reset`` are no-ops, and there is no refresh hook to call), so state kept in a closure or a
    module global is never zeroed at episode reset and advances on every one of the several ``compute()``
    calls a step may make. A term that needs per-episode or per-step state is a class term.
    """

    def __init__(self, func: FunctionType, env: EnvBase, options: ObservationTermOptions):
        super().__init__(func=func, env=env, options=options)
        self._init_post_compute_fields(env)


ObservationTermLike: TypeAlias = ObservationTerm | ObservationTermFuncWrapper
"""Union of a class-based observation term and its function-wrapper form."""


OBSERVATION_TERM_REGISTRY = TermRegistry(
    "OBSERVATION_TERM",
    ObservationTerm,
    ObservationTermFuncWrapper,
    populated_by=("eden.managers.terms.observations",),
)


def _with_history_dims(obs_dims: tuple[int, ...], history_length: int, flatten: bool) -> tuple[int, ...]:
    """Insert the history axis at dim 1, flattening dims >= 1 into one when ``flatten`` is set."""
    dims = list(obs_dims)
    dims.insert(1, history_length)
    if flatten:
        return (dims[0], int(np.prod(dims[1:])))
    return tuple(dims)


def _format_settings(settings: dict[str, int | bool]) -> str:
    """Render option settings as ``name=value`` pairs for a log line."""
    return ", ".join(f"{name}={value}" for name, value in settings.items())


class ObservationManager(ManagerBase[ObservationManagerOptions]):
    _registry = OBSERVATION_TERM_REGISTRY

    def __init__(self, env: EnvBase, options: ObservationManagerOptions):
        super().__init__(env=env, options=options)

        self._group_obs_dim: dict[str, tuple[int, ...] | list[tuple[int, ...]]] = dict()
        self._obs_buffer: dict[str, torch.Tensor | dict[str, torch.Tensor]] | None = None

        for group_name, group_term_dims in self._group_obs_term_dim.items():
            if self._group_obs_options[group_name].concatenate_terms:
                try:
                    term_dims = torch.stack(
                        [torch.tensor(dims, device="cpu") for dims in group_term_dims],
                        dim=0,
                    )
                    if len(term_dims.shape) > 1:
                        dim = self._group_obs_options[group_name].concatenate_dim
                        dim_sum = torch.sum(term_dims[:, dim], dim=0)
                        term_dims[0, dim] = dim_sum
                        term_dims = term_dims[0]
                    else:
                        term_dims = torch.sum(term_dims, dim=0)
                    self._group_obs_dim[group_name] = tuple(term_dims.tolist())
                except RuntimeError:
                    raise RuntimeError(f"Unable to concatenate observation terms in group `{group_name}`.") from None
            else:
                self._group_obs_dim[group_name] = group_term_dims

        # Cross-group raw-output cache. Many tasks reuse the same observation term in multiple groups (e.g. an
        # identical `policy` and `critic` group that differ only in noise/corruption); the raw `term.compute()` is then
        # computed once per group even though it's identical. Terms whose raw output is provably identical (same
        # registered term + same compute params, ignoring post-compute noise/clip/scale/history) compute once per step
        # and the result is reused. Safe because compute() is contractually a pure read of env state (see
        # ``ObservationTerm``): a stateful term advances in ``post_physics_refresh``, which reaches every instance,
        # so two identically configured instances hold identical state and either one's compute() stands for both.
        self._raw_cache: dict = {}
        self._build_dup_signatures()

        # ``(group, term)`` pairs that ``nan_policy="warn"`` has already warned about. Each pair is warned
        # about once and after that only sanitized, with no host read: a warning every step would flood a
        # training log.
        self._non_finite_warned: set[tuple[str, str]] = set()

    @staticmethod
    def _freeze(value):
        """Recursively convert ``value`` into a hashable form for signature keys."""
        if isinstance(value, dict):
            return tuple(sorted((k, ObservationManager._freeze(v)) for k, v in value.items()))
        if isinstance(value, (list, tuple)):
            return tuple(ObservationManager._freeze(v) for v in value)
        return value

    @staticmethod
    def _raw_term_signature(term: ObservationTermLike):
        """Signature identifying terms with identical raw ``compute()`` output.

        Built from the term class plus every config field that feeds ``compute()`` — declared fields and dynamic extras
        alike (registered ``name``, ``func``/``params`` for function terms, custom config fields for class terms) —
        minus the post-compute modifiers in :attr:`ObservationTermOptions.POST_COMPUTE_FIELDS` (noise/clip/scale/
        history, applied per-group after compute()). Excluding those is what lets a noisy ``policy`` term share with a
        clean ``critic`` term.

        Including everything-but-modifiers by default (rather than extras-only) is deliberate: a field added to
        ``ObservationTermOptions`` later participates in the signature unless explicitly listed as a modifier, so a new
        compute-affecting field can't silently merge distinct terms. Returns ``None`` if the signature isn't hashable
        (term won't cache).
        """
        options = term._options
        keys = set(type(options).model_fields) | set(getattr(options, "__pydantic_extra__", None) or {})
        items = []
        for key in sorted(keys):
            if key in ObservationTermOptions.POST_COMPUTE_FIELDS:
                continue
            if key.startswith("_option_"):  # serialization metadata, not config
                continue
            items.append((key, ObservationManager._freeze(getattr(options, key))))
        sig = (term.__class__.__qualname__, tuple(items))
        try:
            hash(sig)
        except TypeError:
            return None
        return sig

    def _build_dup_signatures(self) -> None:
        """Map ``id(term) -> signature`` for terms whose raw output is duplicated.

        Only terms whose signature appears in more than one place are recorded; unique terms are never cached so they
        pay no lookup/storage overhead.
        """
        sig_by_id: dict[int, object] = {}
        counts: dict[object, int] = {}
        for terms in self._group_obs_terms.values():
            for term in terms:
                sig = self._raw_term_signature(term)
                if sig is None:
                    continue
                sig_by_id[id(term)] = sig
                counts[sig] = counts.get(sig, 0) + 1
        self._dup_sig: dict[int, object] = {term_id: sig for term_id, sig in sig_by_id.items() if counts[sig] > 1}

    def _term_raw(self, term: ObservationTermLike) -> torch.Tensor:
        """Compute a term's raw output, reusing a cached result for duplicates."""
        sig = self._dup_sig.get(id(term))
        if sig is None:
            return term.compute()
        cached = self._raw_cache.get(sig)
        if cached is None:
            cached = term.compute()
            self._raw_cache[sig] = cached
        return cached

    def summary(self) -> str:
        msg = f"<ObservationManager> contains {len(self._group_obs_term_names)} groups.\n"
        for group_name, group_dim in self._group_obs_dim.items():
            title = f"Active Observation Terms in Group: '{group_name}'"
            if self._group_obs_options[group_name].concatenate_terms:
                title += f" (shape: {group_dim})"
            rows = []
            obs_terms = zip(
                self._group_obs_term_names[group_name],
                self._group_obs_term_dim[group_name],
                self._group_obs_terms[group_name],
                strict=False,
            )
            for index, (name, dims, term) in enumerate(obs_terms):
                if term._options.history_length > 0 and term._options.flatten_history_dim:
                    # Flattened history: show (9,) ← 3×(3,)
                    original_size = int(np.prod(dims)) // term._options.history_length
                    original_shape = (original_size,) if len(dims) == 1 else dims[1:]
                    shape_str = f"{dims}  ← {term._options.history_length}×{original_shape}"
                else:
                    shape_str = str(tuple(dims))
                rows.append([index, name, shape_str])
            msg += format_table(title, ["Index", "Name", "Shape"], rows, align={"Name": "l"})
            msg += "\n"
        return msg

    @property
    def group_term_names(self) -> dict[str, list[str]]:
        """The active observation term names keyed by group, in registration order (a copy, safe to mutate)."""
        return {group: list(names) for group, names in self._group_obs_term_names.items()}

    @property
    def active_terms(self) -> list[str]:
        """The flat, de-duplicated union of all active term names across groups."""
        return self._flatten_unique(self._group_obs_term_names)

    def iter_terms(self) -> Iterator[tuple[str, str, ObservationTermLike]]:
        """Yield ``(group, name, term)`` for every active observation term, in group order.

        Public accessor so callers (e.g. the per-term profiler) can scan term objects
        grouped, without reaching into the private per-group storage; mirrors
        :meth:`~eden.managers.reward_manager.RewardManager.iter_terms`.
        """
        for group_name in self._group_obs_term_names:
            for name, term in zip(
                self._group_obs_term_names[group_name], self._group_obs_terms[group_name], strict=True
            ):
                yield group_name, name, term

    def get_term(self, name: str) -> ObservationTermLike:
        # term_keys() (not keys()) so declared config fields aren't iterated as obs groups.
        for group_name in self._options.term_keys():
            for term_name, term in zip(
                self._group_obs_term_names[group_name],
                self._group_obs_terms[group_name],
                strict=True,
            ):
                if term_name == name:
                    return term
        # Available options listed for the same reason ``Registry.get`` and
        # ``FlatManagerBase.get_term`` list theirs: the caller has the wrong name and no other way
        # to see the right one. Terms are grouped here, so the groups are named too.
        available = ", ".join(
            f"{group}: {', '.join(names) or 'none'}" for group, names in self.group_term_names.items()
        )
        raise ValueError(f"Term '{name}' not found in observation manager. Available options: {available or 'none'}")

    def set_term_scale(self, group_name: str, term_name: str, scale: float | list[float]) -> None:
        """Retune one term's post-compute ``scale`` after the env is built.

        ``scale`` is resolved once during env construction (see :func:`_resolve_term_scale`), so assigning
        ``term._options.scale`` afterwards changes what the options report but not what the manager
        multiplies by. This is the supported way to change it: the value goes through exactly the check the
        build-time resolution ran — feature dim, length, identity short-circuit — and both halves are written
        together, so a term can never report a scale it is not applying. A rejected value leaves both
        untouched, because the resolution runs first.

        Addressed by ``(group_name, term_name)`` rather than a term handle because the same term name in two
        groups is two independent instances with their own scale, which :meth:`get_term` cannot distinguish.

        **Call this once, at setup.** Scale is applied per frame *before* the frame enters a history buffer,
        so retuning a term whose window already holds frames would leave the older ones at the previous scale
        and the newer ones at the new one — one feature carrying two scales for ``history_length`` steps,
        which nothing in the config can show. That is refused. The window counts as empty before the first
        ``compute()`` and again after a full ``reset()``, so the usual setup orders — retune before the first
        step, or after ``env.reset()``, which is where the training wrappers leave you — are all allowed.

        A term with no history holds no cross-step state in the *observation* pipeline, so it stays retunable
        at any point as far as this manager is concerned. That is not a licence to retune mid-training: the
        RL side holds state derived from the old scale (an observation normalizer's running statistics, most
        obviously) that nothing here can see or invalidate.

        Parameters
        ----------
        group_name: str
            The observation group holding the term.
        term_name: str
            The term to retune, as named in the group.
        scale: float or list[float]
            The new scale. A list is matched against the term's raw feature dim, as at build.
        """
        if group_name not in self._group_obs_term_names:
            available = ", ".join(self._group_obs_term_names) or "none"
            raise ValueError(f"Observation group '{group_name}' not found. Available options: {available}")
        names = self._group_obs_term_names[group_name]
        if term_name not in names:
            available = ", ".join(names) or "none"
            raise ValueError(
                f"Term '{term_name}' not found in observation group '{group_name}'. Available options: {available}"
            )
        self._assert_history_not_started(group_name, term_name)
        index = names.index(term_name)
        term = self._group_obs_terms[group_name][index]
        resolved = _resolve_term_scale(
            term_name, group_name, scale, self._group_obs_term_raw_dim[group_name][index], self._env.device
        )
        term._resolved_scale = resolved
        # Assignment on an options model is not validated (``validate_assignment`` is off), so normalize to
        # the declared ``float | list[float]`` rather than letting a tuple survive into the options.
        term._options.scale = float(scale) if isinstance(scale, (int, float)) else [float(v) for v in scale]

    def _assert_history_not_started(self, group_name: str, term_name: str) -> None:
        """Refuse a post-compute retune that would splice two scales into one history window.

        The hazard is specific to history: a group-level buffer holds the concatenated group, so any term in
        it counts, while a per-term buffer only exists for terms that configure ``history_length``. A term
        with neither is unaffected and this returns.

        Keyed on :attr:`~eden.utils.buffers.circular_buffer.CircularBuffer.has_frames` — "does the window
        still hold frames scaled the old way" — not on ``is_initialized``, which only asks whether the
        storage is allocated and therefore stays ``True`` across a reset. That distinction is what lets a
        retune through right after a full reset, the window then being empty; it also keeps this guard from
        silently depending on ``reset()`` happening to zero in place rather than drop the allocation.
        """
        history = self._group_obs_term_history_buffer[group_name]
        buffer = history if isinstance(history, CircularBuffer) else history.get(term_name)
        if buffer is not None and buffer.has_frames:
            scope = "its group's" if isinstance(history, CircularBuffer) else "its"
            raise RuntimeError(
                f"Cannot retune the scale of observation term '{term_name}' in group '{group_name}': "
                f"{scope} history buffer has already accumulated frames at the current scale. Retuning now "
                f"would leave the older frames scaled one way and newer ones another, for the length of the "
                f"window. Set `scale` in the config, or call set_term_scale() while the window is empty — "
                f"before the first compute(), or after a full reset()."
            )

    @property
    def group_obs_dim(self) -> dict[str, tuple[int, ...] | list[tuple[int, ...]]]:
        return self._group_obs_dim

    @property
    def group_obs_term_dim(self) -> dict[str, list[tuple[int, ...]]]:
        return self._group_obs_term_dim

    def reset(self, envs_idx: torch.Tensor | slice | None = None) -> dict[str, float]:
        history_batch_ids = (
            None if envs_idx is None else sanitize_envs_idx(envs_idx, self._env.num_envs, prefer_slice=False)
        )

        for group_name in self._group_obs_term_names:
            for _, term in self._group_obs_class_terms.get(group_name, ()):
                term.reset(envs_idx=envs_idx)

            for term in self._group_obs_terms[group_name]:
                if term._noise is not None:
                    term._noise.reset(envs_idx=envs_idx)
                # Modifier state is per-step state like the history window, so it restarts with it, on
                # the same indices.
                for modifier in term._modifiers:
                    modifier.reset(envs_idx=history_batch_ids)

            history = self._group_obs_term_history_buffer[group_name]
            if isinstance(history, dict):
                for buffer in history.values():
                    buffer.reset(batch_ids=history_batch_ids)
            else:
                # Group-level history: one CircularBuffer for the concatenated group.
                history.reset(batch_ids=history_batch_ids)
        return {}

    def iter_post_reset_terms(self) -> Iterator[ObservationTermLike]:
        """Every observation term, in group order (a term shared across groups repeats)."""
        return (term for _, _, term in self.iter_terms())

    def iter_post_physics_terms(self) -> Iterator[tuple[str, str, ObservationTerm]]:
        """Yield ``(group, name, term)`` for every term :meth:`post_physics_refresh` ticks, in tick order.

        The class terms (a term shared across groups repeats, as it is ticked once per group): what a CUDA graph
        of the graded phase (``EnvOptions.cuda_graph_options``) watches beside the graded managers' terms.
        """
        for group_name, terms in self._group_obs_class_terms.items():
            for name, term in terms:
                yield group_name, name, term

    def post_physics_refresh(self) -> None:
        """Advance every class term's per-step state on the post-physics scene — phase 7, once per control step.

        The once-per-step tick that ``compute()`` is not (see :class:`ObservationTerm`). Called by the
        env's step after ``command_manager.post_physics_refresh()`` (5) and ``metric_manager.compute()``
        (6), before the graded layer and above the auto-reset; and once per control tick from the deploy
        loop. This method owns that call-time contract; terms override the no-op
        :meth:`ObservationTerm.post_physics_refresh`.

        Reaches class terms only: a function term is stateless by construction
        (:class:`ObservationTermFuncWrapper`). Every instance is ticked, including the duplicates the
        cross-group raw cache computes only once — that is what keeps their state identical and the
        cache sound.
        """
        for terms in self._group_obs_class_terms.values():
            for _, term in terms:
                term.post_physics_refresh()

    def compute(self, update_history: bool = False) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        # Reset the per-step cross-group cache so reused raw outputs reflect the current state. Cleared (not just
        # overwritten) to avoid pinning tensors.
        self._raw_cache.clear()
        obs_buffer: dict[str, torch.Tensor | dict[str, torch.Tensor]] = dict()
        # One pass for `compute_once` (ManagerBase): terms share an intermediate within it, across groups too.
        with self.compute_pass():
            for group_name in self._group_obs_term_names:
                obs_buffer[group_name] = self.compute_group(group_name, update_history)
        self._obs_buffer = obs_buffer
        return obs_buffer

    def compute_group(self, group_name: str, update_history: bool = False) -> torch.Tensor | dict[str, torch.Tensor]:
        # The frames ``nan_policy`` "warn" / "error" still has to check, read back together once the
        # group's terms are computed: one host read per group rather than one per term.
        unchecked: list[tuple[str, torch.Tensor]] = []

        # Fast path: pre-allocated concatenation buffer (no history, 1-D obs, concatenate_terms).
        concat_buf = self._group_concat_buffer.get(group_name)
        if concat_buf is not None:
            slices = self._group_concat_slices[group_name]
            for i, (term_name, term) in enumerate(
                zip(self._group_obs_term_names[group_name], self._group_obs_terms[group_name], strict=True)
            ):
                start, end = slices[i]
                concat_buf[:, start:end].copy_(self._term_frame(group_name, term_name, term, update_history, unchecked))
            self._check_non_finite(group_name, unchecked)
            return concat_buf.clone()

        # General path
        group_obs: dict[str, torch.Tensor] = OrderedDict()

        for term_name, term in zip(
            self._group_obs_term_names[group_name],
            self._group_obs_terms[group_name],
            strict=True,
        ):
            obs = self._term_frame(group_name, term_name, term, update_history, unchecked)
            if self._group_obs_options[group_name].history_length is not None:
                group_obs[term_name] = obs
            elif term._options.history_length > 0:
                circular_buffer = self._group_obs_term_history_buffer[group_name][term_name]
                if update_history or not circular_buffer.is_initialized:
                    circular_buffer.append(obs)
                    buf = circular_buffer.buffer
                else:
                    # Snapshot path: produce the buffer that *would* result from
                    # appending ``obs`` now, without mutating the underlying state.
                    # The canonical once-per-step advance happens elsewhere
                    # (post-reset compute in ``RLEnvBase.step``); doing it here
                    # too would skip a frame for non-done envs, while skipping
                    # it entirely would drop the freshly-computed post-physics
                    # frame from the snapshot.
                    buf = circular_buffer.peek_buffer(obs)

                if term._options.flatten_history_dim:
                    group_obs[term_name] = buf.reshape(self._env.num_envs, -1)
                else:
                    group_obs[term_name] = buf
            else:
                group_obs[term_name] = obs

        self._check_non_finite(group_name, unchecked)

        if self._group_obs_options[group_name].history_length is not None:
            group_obs = torch.cat(
                list(group_obs.values()),
                dim=self._group_obs_options[group_name].concatenate_dim,
            )
            circular_buffer = self._group_obs_term_history_buffer[group_name]
            if update_history or not circular_buffer.is_initialized:
                circular_buffer.append(group_obs)
                buf = circular_buffer.buffer
            else:
                # Snapshot path: see term-level branch above.
                buf = circular_buffer.peek_buffer(group_obs)

            if self._group_obs_options[group_name].flatten_history_dim:
                return buf.reshape(self._env.num_envs, -1)
            else:
                return buf

        if self._group_obs_options[group_name].concatenate_terms:
            return torch.cat(
                list(group_obs.values()),
                dim=self._group_obs_options[group_name].concatenate_dim,
            )
        return group_obs

    def _term_frame(
        self,
        group_name: str,
        term_name: str,
        term: ObservationTermLike,
        update_history: bool,
        unchecked: list[tuple[str, torch.Tensor]],
    ) -> torch.Tensor:
        """One term's frame as its group receives it: raw, modifiers, noise, clip, scale, then ``nan_policy``.

        ``update_history`` marks the once-per-step pass: the modifiers advance here, and the caller pushes
        the history windows. A frame that ``nan_policy`` still has to check is added to ``unchecked``,
        for :meth:`_check_non_finite` once the group is done.
        """
        obs = self._term_raw(term)
        for modifier in term._modifiers:
            output = modifier.compute(obs)
            if update_history:
                modifier.advance(obs)
            obs = output
        # Corruption is training-time domain randomization; skip it during eval (env.training is False)
        # so the policy sees clean obs (mjlab play-mode parity). Default True for envs without a
        # training flag.
        if term._noise is not None and getattr(self._env, "training", True):
            obs = term._noise.compute(obs)
        if term._options.clip is not None:
            obs = obs.clamp(min=term._options.clip[0], max=term._options.clip[1])
        if term._resolved_scale is not None:
            obs = term._resolved_scale * obs
        return self._apply_nan_policy(obs, group_name, term_name, unchecked)

    def _apply_nan_policy(
        self, obs: torch.Tensor, group_name: str, term_name: str, unchecked: list[tuple[str, torch.Tensor]]
    ) -> torch.Tensor:
        """Apply the group's ``nan_policy`` to the frame a term delivers, after its clip and scale.

        Checked last so that a ``clip`` which bounds an infinity (a depth term's no-hit ``+inf``) is
        never reported. Nothing here reads a value back to the host: "sanitize" and "warn" replace
        non-finite entries on the device, and "warn" and "error" leave the check itself to
        :meth:`_check_non_finite`. Out of place, because ``obs`` may still be the cross-group raw
        cache, a term's own buffer, or a view of simulator state.
        """
        nan_policy = self._group_obs_options[group_name].nan_policy
        if nan_policy == "disabled" or not obs.is_floating_point():
            return obs  # an integer or bool frame cannot hold a non-finite value
        match nan_policy:
            case "sanitize":
                return torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
            case "warn":
                if (group_name, term_name) not in self._non_finite_warned:
                    unchecked.append((term_name, obs))
                return torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
            case "error":
                unchecked.append((term_name, obs))
                return obs
            case _:  # only reachable through an unvalidated assignment to the group options
                raise ValueError(
                    f"Observation group '{group_name}' has nan_policy={nan_policy!r}; expected one of "
                    f"'disabled', 'sanitize', 'warn', 'error'."
                )

    def _check_non_finite(self, group_name: str, unchecked: list[tuple[str, torch.Tensor]]) -> None:
        """Warn about, or raise on, the non-finite frames of one group computation (``nan_policy`` "warn"/"error").

        One host read covers the whole group; the per-term report reads more only when something was
        found. The frames are the ones the terms produced, before "warn" replaced anything.
        """
        if not unchecked:
            return
        flags = torch.stack([(~torch.isfinite(frame)).any() for _, frame in unchecked])
        if not bool(flags.any()):  # the one GPU sync of a clean group
            return
        found = [(name, frame) for (name, frame), bad in zip(unchecked, flags.tolist(), strict=True) if bad]
        reports = " ".join(self._describe_non_finite(frame, f"{group_name}.{name}") for name, frame in found)
        if self._group_obs_options[group_name].nan_policy == "error":
            raise FloatingPointError(
                f"{reports} Observation group '{group_name}' has nan_policy='error'. Find where the value comes "
                f"from (the term, one of its modifiers, or a simulation that diverged), or set "
                f"nan_policy='sanitize' to deliver 0.0 there instead."
            )
        self._non_finite_warned.update((group_name, name) for name, _ in found)
        en.logger.warning(
            f"{reports} Replaced with 0.0 (nan_policy='warn' on observation group '{group_name}'); later "
            f"occurrences in the same term are replaced without a warning."
        )

    @staticmethod
    def _describe_non_finite(frame: torch.Tensor, term_path: str) -> str:
        """Say which non-finite values ``term_path`` produced and in which envs, for a warning or an error."""
        kinds = [
            kind
            for kind, present in (
                ("NaN", torch.isnan(frame).any()),
                ("+Inf", torch.isposinf(frame).any()),
                ("-Inf", torch.isneginf(frame).any()),
            )
            if bool(present)
        ]
        non_finite = ~torch.isfinite(frame)
        envs = non_finite.reshape(frame.shape[0], -1).any(dim=1).nonzero().flatten().tolist()
        shown = ", ".join(str(env_id) for env_id in envs[:10]) + (", ..." if len(envs) > 10 else "")
        return (
            f"Observation term '{term_path}' produced {'/'.join(kinds)} in {len(envs)} of {frame.shape[0]} envs "
            f"(env ids: {shown})."
        )

    def _prepare_terms(self) -> None:
        self._group_obs_term_names: dict[str, list[str]] = defaultdict(list)
        self._group_obs_terms: dict[str, list[ObservationTermLike]] = defaultdict(list)
        # The class terms only — the ones with lifecycle hooks (``reset`` / ``post_physics_refresh``). A
        # function term's wrapper has nothing to dispatch to; it is stateless by construction.
        self._group_obs_class_terms: dict[str, list[tuple[str, ObservationTerm]]] = defaultdict(list)
        self._group_obs_term_dim: dict[str, list[tuple[int, ...]]] = defaultdict(list)
        # The *raw* compute() shape per term, batch dim included — unlike ``_group_obs_term_dim``, which is
        # history-expanded and batch-stripped. Kept because that is the shape a vector ``scale`` is matched
        # against, and ``set_term_scale`` has to re-run the same check the build-time resolution ran.
        self._group_obs_term_raw_dim: dict[str, list[tuple[int, ...]]] = defaultdict(list)
        self._group_obs_options: dict[str, ObservationGroupOptions] = {}

        self._group_obs_term_history_buffer: dict[str, dict[str, CircularBuffer] | CircularBuffer] = dict()

        for group_name in self._options.term_keys():
            group_options: ObservationGroupOptions = getattr(self._options, group_name)
            # The group's whole construction rule, asked again here because its declared fields are
            # written unchecked: a bare ``group.terms_order = [...]`` or ``group.concatenate_terms =
            # False`` is how a group reaches the build out of step with itself (a *term* is checked
            # where it is assigned). Asked before anything reads the entries, so the message
            # is the group's own, naming the entry and the fix, not a ``KeyError`` from the loops below.
            group_options._validate_group()
            self._group_obs_options[group_name] = group_options

            terms_data = self._build_group_term_options(group_name, group_options)
            # NOTE: the original (terms-less) ``group_options`` stays in
            # ``self._group_obs_options``; the resolved copy carrying ``terms`` /
            # ``terms_order`` is used only to instantiate this group's terms.
            resolved_options = self._resolve_terms_order(group_options, terms_data)
            self._instantiate_group_terms(group_name, resolved_options)

        self._prealloc_concat_buffers()

    def _build_group_term_options(
        self, group_name: str, group_options: ObservationGroupOptions
    ) -> dict[str, ObservationTermOptions]:
        """Parse a group's per-term options from its Pydantic extra fields.

        Applies the group-level overrides each term inherits: noise is dropped when
        the group disables corruption, and history settings are forced onto every
        term when the group enforces a fixed history length. This is the only place
        either happens — the group options keep what the author wrote, so one line
        here names every ignored setting however the config was assembled.

        Both group-level overrides are documented idioms, so applying one is an info
        line per group: ``history_length`` to give every term the same window, and
        ``enable_corruption=False`` to run a critic on the clean state with the same
        term dict the noisy policy uses. Replacing a history setting a term wrote
        *itself* is the exception and warns — a term's own ``flatten_history_dim`` /
        ``backfill``, and a ``history_length`` it set above the default 0 (a term at 0
        has no window of its own to lose) — because nothing else in the config says
        that window was asked for and refused.
        """
        group_path = f"observation_options.{group_name}"
        enforced_history: dict[str, int | bool] | None = None
        if group_options.history_length is not None:
            enforced_history = {
                "history_length": group_options.history_length,
                "flatten_history_dim": group_options.flatten_history_dim,
                "backfill": group_options.backfill,
            }
        noise_dropped: list[str] = []
        history_replaced: list[str] = []

        terms_data: dict[str, ObservationTermOptions] = {}
        # The group answers which of its extras are terms (the rest is the ``_option_*`` type
        # metadata and the resolved ``terms`` dict), and every one of them is an
        # ``ObservationTermOptions`` it owns: a term is adopted however it arrives — constructor,
        # ``model_copy(update=...)``, assignment — and an entry that is not a term is refused at
        # the line that wrote it. So there is nothing to re-filter or re-build here.
        for key, term_options in group_options._term_entries().items():
            if not group_options.enable_corruption and term_options.noise is not None:
                noise_dropped.append(key)
                term_options = term_options.model_copy(update={"noise": None})

            if enforced_history is not None:
                # Only what the term actually wrote and the group then discards. A term at the
                # default ``history_length=0`` has no window of its own for the group to replace,
                # so its zero is not a discarded setting — but a ``flatten_history_dim`` /
                # ``backfill`` it wrote is overwritten whether or not it asked for a length, so it
                # is reported too.
                replaced: dict[str, int | bool] = {}
                for field, enforced in enforced_history.items():
                    own = getattr(term_options, field)
                    if own != enforced and not (field == "history_length" and own == 0):
                        replaced[field] = own
                if replaced:
                    history_replaced.append(f"{key} ({_format_settings(replaced)})")
                term_options = term_options.model_copy(update=enforced_history)

            terms_data[key] = term_options

        if noise_dropped:
            en.logger.info(
                f"{group_path}: enable_corruption is False, so the noise set on {', '.join(noise_dropped)} is not "
                f"applied. Set {group_path}.enable_corruption = True if it was meant to be."
            )
        if enforced_history is not None and terms_data:
            settings = _format_settings(enforced_history)
            en.logger.info(f"{group_path}: {settings} applied to {len(terms_data)} terms.")
            if history_replaced:
                en.logger.warning(
                    f"{group_path}: the group's {settings} replaces the history settings these terms wrote: "
                    f"{'; '.join(history_replaced)}. A group history is one buffer over the concatenated group, so "
                    f"no per-term window or flag survives it — drop these settings, or set "
                    f"{group_path}.history_length = None and give each term its own history_length (a term left at "
                    f"0 then gets no history at all)."
                )
        return terms_data

    def _resolve_terms_order(
        self, group_options: ObservationGroupOptions, terms_data: dict[str, ObservationTermOptions]
    ) -> ObservationGroupOptions:
        """Default the group's ``terms_order`` to declaration order and stash ``terms`` on a copy.

        Returns a copy with both ``terms_order`` and ``terms`` populated.
        """
        if group_options.terms_order is None:
            # No explicit order given: default to term declaration order. An explicit one was
            # checked against the group's terms in ``_prepare_terms``.
            group_options = group_options.model_copy(update={"terms_order": list(terms_data.keys())})

        # Store terms on group options for downstream usage
        return group_options.model_copy(update={"terms": terms_data})

    def _instantiate_group_terms(self, group_name: str, group_options: ObservationGroupOptions) -> None:
        """Build every term in ``group_options.terms_order`` and record its shape / history buffer.

        Each term is warm-computed once (``term.compute()``) purely to discover its output
        shape, since term dims aren't declared up front. That call happens before any ``reset()``
        and with no ``post_physics_refresh()`` before it, which is harmless under the compute-is-a-read
        contract (:class:`ObservationTerm`): a stateful term returns its freshly allocated buffer.
        """
        if group_options.history_length is None:
            # NOTE: prepare buffer for each term history separately if history length is not enforced for the group
            group_entry_history_buffer: dict[str, CircularBuffer] = dict()

        for term_name in group_options.terms_order:
            term_options = group_options.terms[term_name]
            # Labelled with the group too: the term knows only its key.
            term: ObservationTermLike = self._build_term(f"{group_name}.{term_name}", term_options)
            self._group_obs_term_names[group_name].append(term_name)
            self._group_obs_terms[group_name].append(term)
            if isinstance(term, ObservationTerm):
                self._group_obs_class_terms[group_name].append((term_name, term))

            probe = term.compute()
            obs_dims = tuple(probe.shape)
            self._build_term_modifiers(group_name, term_name, term, probe)
            self._group_obs_term_raw_dim[group_name].append(obs_dims)
            term._resolved_scale = _resolve_term_scale(
                term_name, group_name, term._options.scale, obs_dims, self._env.device
            )

            if group_options.history_length is not None:
                obs_dims = _with_history_dims(obs_dims, term._options.history_length, term._options.flatten_history_dim)
            elif term._options.history_length > 0:
                group_entry_history_buffer[term_name] = CircularBuffer(
                    max_len=term._options.history_length,
                    batch_size=self._env.num_envs,
                    device=self._env.device,
                    backfill=term._options.backfill,
                )
                obs_dims = _with_history_dims(obs_dims, term._options.history_length, term._options.flatten_history_dim)

            self._group_obs_term_dim[group_name].append(obs_dims[1:])

        if group_options.history_length is None:
            self._group_obs_term_history_buffer[group_name] = group_entry_history_buffer
        else:
            self._group_obs_term_history_buffer[group_name] = CircularBuffer(
                max_len=group_options.history_length,
                batch_size=self._env.num_envs,
                device=self._env.device,
                backfill=group_options.backfill,
            )

    @staticmethod
    def _build_term_modifiers(group_name: str, term_name: str, term: ObservationTermLike, probe: torch.Tensor) -> None:
        """Build a term's modifiers in order, each from the frame the previous one returns for the shape probe.

        Checks that each keeps the frame's shape, dtype and device: ``scale``, the history window and
        concatenation are all sized from the term's raw output, so a modifier that changed them would
        break those further down, with an error that does not mention the modifier (or, in the
        concatenation buffer, a silent cast). ``compute()`` is a read, so
        running it on the probe leaves every modifier as ``build()`` left it.
        """
        frame = probe
        for index, modifier in enumerate(term._modifiers):
            which = f"{index} ('{modifier._options.name}') of observation term '{group_name}.{term_name}'"
            try:
                modifier.build(frame)
                output = modifier.compute(frame)
            except Exception as e:
                e.add_note(f"while building modifier {which}")
                raise
            if (output.shape, output.dtype, output.device) != (frame.shape, frame.dtype, frame.device):
                raise ValueError(
                    f"Modifier {which} returned a {tuple(output.shape)} {output.dtype} tensor on {output.device} "
                    f"for a {tuple(frame.shape)} {frame.dtype} frame on {frame.device}. A modifier must return "
                    f"the shape, dtype and device it receives."
                )
            frame = output

    def _prealloc_concat_buffers(self) -> None:
        """Pre-allocate concatenation buffers for groups that qualify for the fast path:
        concatenate_terms=True, no group-level history, all 1-D terms with no per-term history.
        """
        self._group_concat_buffer: dict[str, torch.Tensor] = {}
        self._group_concat_slices: dict[str, list[tuple[int, int]]] = {}
        for group_name, group_options in self._group_obs_options.items():
            if not group_options.concatenate_terms or group_options.history_length is not None:
                continue
            term_dims = self._group_obs_term_dim[group_name]
            has_history = any(t._options.history_length > 0 for t in self._group_obs_terms[group_name])
            if has_history or not all(len(d) == 1 for d in term_dims):
                continue
            total_dim = sum(d[0] for d in term_dims)
            slices: list[tuple[int, int]] = []
            offset = 0
            for d in term_dims:
                slices.append((offset, offset + d[0]))
                offset += d[0]
            self._group_concat_buffer[group_name] = torch.zeros(
                self._env.num_envs,
                total_dim,
                device=self._env.device,
            )
            self._group_concat_slices[group_name] = slices
