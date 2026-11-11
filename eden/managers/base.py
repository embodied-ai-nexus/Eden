"""Base classes and shared logic for managers and their terms.

A manager owns an ordered set of *terms* resolved from its options through a
:class:`~eden.utils.registry.Registry`. Terms come in two flavors:

- function-based (stateless): a plain function registered with ``@REGISTRY.register()``
  and wrapped in a ``FuncWrapper``.
- class-based (stateful): subclass the kind's term base (a :class:`ConfigurableTermBase`) and
  implement ``compute()``, optionally ``build()`` and ``reset()``.

Concrete managers (action, observation, reward, termination, ...) live in sibling
modules and are imported lazily inside the env's ``_load_managers()`` to break the
manager <-> env circular import.
"""

from __future__ import annotations

from abc import ABCMeta, abstractmethod
from contextlib import AbstractContextManager
from types import FunctionType
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Generic, Hashable, Iterable, TypeVar

import torch
from genesis.repr_base import RBC
from eden.utils.format import format_table

from eden.options.envs import EnvOptions
from eden.options.options import Options
from eden.utils.common import ConfigurableFuncWrapperMixin, ConfigurableMixin
from eden.utils.misc import resolve_named_entity, sanitize_envs_idx
from eden.utils.registry import Registry

if TYPE_CHECKING:
    from eden.entities.base import Entity
    from eden.envs.base import EnvBase


T = TypeVar("T", bound=Options)
_TermOptionsT = TypeVar("_TermOptionsT", bound=Options)


class ManagerTermBase(RBC):
    """Base class for all manager terms."""

    #: For a class term that was once a *function* term under the same registry key: the keys it
    #: took in ``params={...}`` back then. A config saved before the conversion still carries them
    #: there, and ``ManagerTermOptions`` moves exactly these onto the class's fields at construction
    #: so that config keeps loading. Empty — the default — for a term that was never a
    #: function term, on which ``params`` stays an error. Every key must be a parameter of the class.
    _function_era_params: ClassVar[frozenset[str]] = frozenset()

    #: Boolean :class:`~eden.options.envs.EnvOptions` flags this term needs set to ``True`` — a
    #: *scene-wide* precondition no parameter of the term can satisfy (the per-env batching
    #: switches, today). :meth:`_check_env_options_requirements` enforces them at construction, so
    #: a config that forgets one raises while the env is being built, naming the term and the flag,
    #: rather than failing on the first write. The catalog renders them as its ``Requires`` column
    #: (:class:`~eden.managers.catalog.TermEntry`), so the requirement is readable before the first
    #: build. A *function* term declares one through its registration —
    #: ``@REGISTRY.register(_requires_env_options=("batch_links_info",))`` — which synthesizes a
    #: wrapper subclass carrying it.
    #:
    #: The leading underscore is load-bearing: ``ConfigurableMixin.get_parameter_names`` collects
    #: every *public* class attribute, so a name without it would become a ``configure()``
    #: parameter on the term and a row in the catalog's parameter column.
    #:
    #: Only for env-options flags. A requirement on something the term *can* inspect (a sibling
    #: modifier, a named link, a command term) stays a check in the term's own ``build()``.
    _requires_env_options: ClassVar[tuple[str, ...]] = ()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._validate_env_options_declaration()

    @classmethod
    def _validate_env_options_declaration(cls) -> None:
        """Reject a malformed :attr:`_requires_env_options` where it is written.

        At class-definition time, not at construction: the catalog reads the declaration without
        instantiating anything, so a typo would otherwise be *rendered into the shipped page*
        (``EnvOptions(batch_dof_info=True)``) and raise only for whoever first builds a scene with
        the term in it. Checked on the declaring class only (``__dict__``, not ``getattr``), so
        each declaration is validated exactly once — including the one ``TermRegistry._with_defaults``
        synthesizes for a function term registered with the flag.
        """
        declared = cls.__dict__.get("_requires_env_options")
        if declared is None:
            return
        if isinstance(declared, str):
            # A string is a perfectly good collection of *characters*: left alone this iterates
            # 'b', 'a', 't', ... and reports the first letter as the offending field name.
            raise TypeError(
                f"{cls.__name__}._requires_env_options must be a tuple of EnvOptions field names, got the "
                f"bare string {declared!r} — pass ({declared!r},)."
            )
        unknown = [flag for flag in declared if flag not in EnvOptions.model_fields]
        if unknown:
            raise ValueError(
                f"{cls.__name__}._requires_env_options names {unknown}, which are not EnvOptions fields. "
                f"Fix the term's declaration."
            )

    def __init__(self, env: EnvBase):
        self._env = env
        self._check_env_options_requirements()

    def _check_env_options_requirements(self) -> None:
        """Raise unless every flag in :attr:`_requires_env_options` is set on this env.

        Runs at construction — the earliest point both the term class and the env options are
        known, and well before the first write the flag governs, which is where the failure would
        otherwise surface (as a Genesis rejection on the first reset, or as a silently unbatched
        write). A ``ValueError``, not an ``assert``: ``python -O`` strips asserts, and a stripped
        precondition is a training run whose domain randomization quietly does nothing.

        Returns before touching ``env.env_options`` when the term declares nothing — which is
        nearly every term — so constructing one still needs only the env surface it actually uses
        (the lightweight env doubles in the test suite carry no ``env_options`` at all).
        """
        if not self._requires_env_options:
            return
        env_options = self._env.env_options
        for flag in self._requires_env_options:
            # The name is already known to be an EnvOptions field, so a missing attribute means an
            # env object that does not expose it — unset either way, and the same fix applies.
            if not getattr(env_options, flag, False):
                raise ValueError(
                    f"{type(self).__name__} requires `EnvOptions({flag}=True)`, which this env does not "
                    f"set. Set it in the task's `env_options` — `eden terms` lists it on the term's "
                    f"`requires:` line."
                )

    @property
    def num_envs(self) -> int:
        """The number of parallel environments."""
        return self._env.num_envs

    @property
    def device(self) -> torch.device:
        """The torch device the term's tensors live on."""
        return self._env.device

    def resolve_entity(self, name: str) -> Entity:
        """Resolve a named entity from the env — the declared way for a term to reach an entity.

        Resolve through this in ``compute()`` rather than caching the handle in ``build()``:
        the deployment hot-swaps ``env.entities[name]`` (sim entity → ``RobotStateEntity``) so
        observations read real robot state, and a cached handle would keep pointing at the stale
        sim entity. Cache only *derived* data (resolved link/DOF indices stay valid across the
        swap), not the entity itself. (``DeploymentBase`` repoints any straggler cached handles
        by object identity as a safety net; resolving lazily means a term never needs it.)

        An unknown name raises here, listing the names the scene does have.
        """
        return resolve_named_entity(self._env.entities, name)

    def reset(self, envs_idx: torch.Tensor | None = None) -> Any:
        """Reset the manager term."""
        del envs_idx  # Unused.

    def post_reset(self, envs_idx: slice | torch.Tensor) -> None:
        """Hook invoked after ALL managers have reset ``envs_idx``.

        Runs after every manager's ``reset()``, the reset-mode events, and the zeroing of
        ``episode_length_buf`` — the guaranteed point to rebuild per-step caches from the
        *final* post-reset scene state, without depending on the order managers reset in.
        (A term's own ``reset()`` runs mid-sequence: state it derives from other entities
        or terms may still change before observations are computed.) No-op by default.

        Managers dispatch source caches before their consumers (for example, command
        terms before metric terms). Runs under ``torch.inference_mode``: mutate
        ``build()``-allocated buffers in place; never rebind attributes to tensors
        created inside this hook.

        Parameters
        ----------
        envs_idx : slice | torch.Tensor
            The envs that were reset — never ``None``, but *index with it, never measure
            it*: the auto-reset path in ``RLEnvBase.step`` passes the full-length boolean
            ``reset_buf``, while ``EnvBase.reset`` passes a slice or an index tensor. So
            ``buf[envs_idx] = ...`` and ``src[envs_idx]`` are always right, and
            ``len(envs_idx)`` / ``envs_idx.numel()`` silently mean ``num_envs`` rather
            than the reset count on the step path.
        """
        del envs_idx  # Unused.

    @abstractmethod
    def compute(self, *args, **kwargs) -> Any:
        """Step the manager term."""
        raise NotImplementedError

    def build(self) -> None:
        """Deferred build hook for terms that resolve entities after env setup."""
        return None


class ManagerTermFuncWrapperBase(ManagerTermBase):
    """Runtime contract for manager terms defined as a function: hold ``func`` and call it.

    Like :class:`ManagerTermBase`, this carries no options machinery. Concrete wrappers pair
    it with ``ConfigurableFuncWrapperMixin[KindOptions]``, which contributes ``configure()``
    and the ``params`` attribute that :meth:`compute` forwards to the wrapped function.
    """

    def __init__(self, func: FunctionType, env: EnvBase):
        self._func = func
        ManagerTermBase.__init__(self, env=env)

    def compute(self, *args, **kwargs) -> Any:
        """Invoke the wrapped function with the environment and configured params."""
        return self._func(env=self._env, **self.params, **kwargs)


class ConfigurableTermBase(ManagerTermBase, ConfigurableMixin[_TermOptionsT]):
    """A class term bound to its options: the base every kind's class-term base derives from.

    Parameterize it with the kind's options class (``ConfigurableTermBase[RewardTermOptions]``);
    ``ConfigurableMixin`` reads the type argument through this intermediate generic base. The
    configurable fields every term of a kind shares go on a ``ConfigurableMixin`` subclass the
    kind's class base and its :class:`ConfigurableTermFuncWrapper` both inherit, so they are
    declared once; such a mixin must not define ``__init__``.
    """

    def __init__(self, env: EnvBase, options: _TermOptionsT):
        ManagerTermBase.__init__(self, env=env)
        ConfigurableMixin.__init__(self, options=options)


class ConfigurableTermFuncWrapper(ManagerTermFuncWrapperBase, ConfigurableFuncWrapperMixin[_TermOptionsT]):
    """The function-term counterpart of :class:`ConfigurableTermBase`: each kind's ``*FuncWrapper`` base.

    ``_func`` is set before ``ConfigurableFuncWrapperMixin.__init__`` runs, because that is where
    the ``params`` keys are checked against the wrapped function's signature.
    """

    def __init__(self, func: FunctionType, env: EnvBase, options: _TermOptionsT):
        ManagerTermFuncWrapperBase.__init__(self, func=func, env=env)
        ConfigurableFuncWrapperMixin.__init__(self, options=options)


def build_term(registry: Registry, env: EnvBase, options: Options, *, label: str) -> ManagerTermBase:
    """Construct the term ``options`` names from ``registry`` and ``build()`` it.

    The one place a term is built from its options — by every manager (:meth:`ManagerBase._build_term`,
    the observation manager's groups) and by the composite terms that build children of their own.

    Parameters
    ----------
    registry : Registry
        The term registry ``options.name`` is a key of.
    env : EnvBase
        The env the term is built for.
    options : Options
        The term's options; ``options.name`` is its registry key.
    label : str
        Names the config entry being built, starting with the class that holds it
        (``"RewardManager term 'lift_milk'"``, ``"MetricTermAND.terms[0]"``): the term itself only knows its
        key, and a task can hold a dozen entries of one term (``metric_reward`` ×12 in robosuite
        PickPlace), so the message alone may not say which to fix. It opens the error for a nameless term
        and is added as a note to any exception constructing or building the term raises, whose type and
        message are left unchanged.

    Raises
    ------
    ValueError
        If ``options.name`` is ``None`` — a legal placeholder the registry would answer with
        ``KeyError: None``, naming neither the entry nor the fix.
    """
    if options.name is None:
        raise ValueError(
            f"{label} names no registered term (name=None), so there is nothing to build. "
            f"Write it as <TermClass>.configure(...), which sets name, or give it name='<registry key>' — "
            f"`eden terms --grep <text>` lists them."
        )
    try:
        term = registry.get(options.name)(env=env, options=options)
        term.build()
    except Exception as e:
        e.add_note(f"while building {label} (name='{options.name}')")
        raise
    return term


_R = TypeVar("_R")


class ManagerBase(RBC, Generic[T], metaclass=ABCMeta):
    """Base class for all managers."""

    #: The registry every term of this manager is resolved from (:meth:`_build_term`).
    _registry: ClassVar[Registry]

    #: Configured term fields this manager reads from the device rather than as Python values, kept current by
    #: :meth:`sync_device_fields`. A CUDA graph of the graded phase (``EnvOptions.cuda_graph_options``) calls
    #: that before each replay instead of re-capturing when one of them moves. Empty unless a manager mirrors any.
    device_synced_fields: ClassVar[frozenset[str]] = frozenset()

    def __init__(self, env: EnvBase, options: T):
        self._env = env
        self._options: T = options
        self._prepare_terms()

    def compute_once(self, key: Hashable, producer: Callable[[], _R]) -> _R:
        """:meth:`EnvBase.compute_once <eden.envs.base.EnvBase.compute_once>` of this manager's env.

        The memo belongs to the env's pass, not to a manager: a key a term of one manager produced is served
        to a term of another within one pass.
        """
        return self._env.compute_once(key, producer)

    def compute_pass(self) -> AbstractContextManager[None]:
        """:meth:`EnvBase.compute_pass <eden.envs.base.EnvBase.compute_pass>` of this manager's env.

        A manager wraps its term loop in this; inside a pass the env already opened, it joins that one.
        """
        return self._env.compute_pass()

    def sync_device_fields(self) -> None:
        """Bring the device copies of :attr:`device_synced_fields` up to date with the terms' values (a no-op here)."""

    @property
    def num_envs(self) -> int:
        """The number of parallel environments."""
        return self._env.num_envs

    @property
    def device(self) -> str:
        """The torch device the manager's tensors live on."""
        return self._env.device

    def reset(self, envs_idx: torch.Tensor | None = None) -> dict[str, Any]:
        """Reset the manager and return logging info for the current step."""
        del envs_idx  # Unused.
        return {}

    @abstractmethod
    def compute(self, *args, **kwargs) -> Any:
        """Compute the manager."""
        raise NotImplementedError

    @abstractmethod
    def _prepare_terms(self):
        raise NotImplementedError

    def _build_term(self, term_name: str, term_option: Options | None = None) -> ManagerTermBase:
        """Resolve and build a term from this manager's options.

        Centralizes the ``getattr(options, term_name)`` → ``_registry.get(term_option.name)``
        → ``term.build()`` sequence shared by every manager's ``_prepare_terms``. Pass
        ``term_option`` explicitly when the option object is not a direct attribute of the
        manager options (e.g. a reward term nested inside a reward group, an observation term
        inside a group); ``term_name`` is then only the label below.

        An error raised while constructing or building the term gets a note naming the entry
        (``term_name``) and its registry key — see :func:`build_term`.
        """
        if term_option is None:
            term_option = getattr(self._options, term_name)
        return build_term(self._registry, self._env, term_option, label=f"{type(self).__name__} term '{term_name}'")

    @staticmethod
    def _flatten_unique(grouped: dict[Any, list[str]]) -> list[str]:
        """Flatten grouped term names into a de-duplicated list, in first-seen order."""
        return list(dict.fromkeys(name for names in grouped.values() for name in names))

    @abstractmethod
    def iter_post_reset_terms(self) -> Iterable[ManagerTermBase]:
        """Yield every term this manager owns, for the :meth:`post_reset` fan-out.

        Abstract rather than defaulted so a new manager cannot silently drop the hook:
        :meth:`post_reset` is the only generic thing the env can call on an arbitrary
        :class:`ManagerBase`, and a no-op default would make a missed term look like a
        term that simply had nothing to rebuild. May yield the same term more than once
        (a term shared across observation groups or event modes); the fan-out dedupes.
        """
        raise NotImplementedError

    @torch.inference_mode()
    def post_reset(self, envs_idx: slice | torch.Tensor | None = None) -> None:
        """Fan :meth:`ManagerTermBase.post_reset` out once per unique term.

        The env calls this from its public entry points (:meth:`~eden.envs.base.EnvBase.reset`,
        :meth:`~eden.envs.base.RLEnvBase.step`) right *after* ``_reset_idx`` returns — not from
        inside it, so a subclass's extra reset work finishes first (the mimic env refills the
        motion buffer after ``super()._reset_idx``). By then every manager's ``reset()``, the
        reset-mode events and the ``episode_length_buf`` zeroing have all run, so terms can
        rebuild per-step caches from the final post-reset state. The env owns that call-time
        contract, like ``CommandManager.post_physics_refresh``.

        Decorated ``@torch.inference_mode()`` so terms whose buffers were created under an
        inference-moded manager path (metric and reward terms) can mutate them here; the flip
        side is the in-place-only contract in :meth:`ManagerTermBase.post_reset`.

        Parameters
        ----------
        envs_idx : slice | torch.Tensor, optional
            The envs that were reset; ``None`` means all of them. Sanitized before it reaches
            the terms.
        """
        envs_idx = sanitize_envs_idx(envs_idx, self._env.num_envs)
        seen: set[int] = set()
        for term in self.iter_post_reset_terms():
            term_id = id(term)
            if term_id in seen:
                continue
            seen.add(term_id)
            term.post_reset(envs_idx)


class FlatManagerBase(ManagerBase[T]):
    """Base for managers whose terms form a single flat, ordered set.

    Owns the flat-term API (``terms``, ``active_terms``, ``get_term``,
    ``get_term_options``) backed by ``_terms`` / ``_term_names``, which
    :meth:`_prepare_terms` populates. The hierarchical managers (observations by
    group, events by mode) inherit :class:`ManagerBase` directly and expose their
    own named views instead.
    """

    def __init__(self, env: EnvBase, options: T):
        self._term_names: list[str] = list()
        self._terms: dict[str, ManagerTermBase] = dict()
        super().__init__(env, options)
        #: The terms a manager's ``reset()`` dispatches to: the class terms. A function term's ``reset()``
        #: is the inherited no-op, and the auto-reset calls ``reset()`` every step. Fixed at construction
        #: (``CommandManager.swap_term`` replaces terms after it, and resets every term itself).
        self._reset_terms: list[ManagerTermBase] = [
            term for term in self._terms.values() if not isinstance(term, ManagerTermFuncWrapperBase)
        ]

    def _prepare_terms(self) -> None:
        """Build every term of the options, in declaration order.

        A subclass with more to set up per term (counters, a second resolution pass, a reserved
        name) extends this rather than repeating the loop.
        """
        for term_name in self._options.term_keys():
            term = self._build_term(term_name)
            self._term_names.append(term_name)
            self._terms[term_name] = term

    @property
    def terms(self) -> dict[str, ManagerTermBase]:
        """Mapping from term name to the built :class:`ManagerTermBase` instance."""
        return self._terms

    @property
    def active_terms(self) -> list[str]:
        """The names of the active terms, in registration order (the live internal list; do not mutate)."""
        return self._term_names

    def get_term(self, name: str) -> ManagerTermBase:
        """Return the active term registered under ``name``.

        Raises
        ------
        KeyError
            If ``name`` is not an active term. The message lists the active ones, in the wording
            :meth:`eden.utils.registry.Registry.get` uses for the same miss: a caller reaching here
            with the wrong name (a term renamed in the config, a reward term naming an action term
            that was never added) has no other way to see what *is* there.
        """
        if name not in self._terms:
            available = ", ".join(self._term_names) or "none"
            raise KeyError(f"No active term named '{name}' in {type(self).__name__}. Available options: {available}")
        return self._terms[name]

    def get_term_options(self, term_name: str) -> Options:
        """Return the options object for the active term ``term_name``.

        Raises
        ------
        ValueError
            If ``term_name`` is not an active term.
        """
        if term_name not in self._term_names:
            raise ValueError(f"Term '{term_name}' not found in active terms.")
        return self._terms[term_name].options

    def iter_post_reset_terms(self) -> Iterable[ManagerTermBase]:
        """The flat term set, in registration order."""
        return self._terms.values()

    def _format_summary_table(
        self,
        *,
        title: str,
        field_names: list[str],
        rows: Iterable[Iterable[Any]],
        align: dict[str, str] | None = None,
    ) -> str:
        r"""Build the canonical ``<{Manager}> contains N active terms.\\n<table>\\n`` summary string."""
        msg = f"<{type(self).__name__}> contains {len(self._term_names)} active terms.\n"
        msg += format_table(title, field_names, rows, align=align)
        msg += "\n"
        return msg
