"""ConfigurableMixin: bind options classes to their runtime objects."""

from __future__ import annotations
from collections.abc import Hashable, Iterable, Mapping
from dataclasses import fields as dataclass_fields, is_dataclass
from enum import Enum
from typing import Any, ClassVar, TYPE_CHECKING, Generic, TypeVar, get_args
from types import FunctionType, MappingProxyType
from functools import cached_property
import copy
import inspect

import genesis as gs
import numpy as np
import torch
from genesis.options.surfaces import Surface
from pydantic import BaseModel

from eden.utils.misc import to_snake_case
from eden.utils.params import (
    ParamInfo,
    function_param_specs,
    invalid_function_param_keys,
    invalid_parameters_message,
    invalid_params_message,
    render_annotation,
    render_default,
    render_params,
)

# NOTE: nothing from ``eden.options`` may be imported at module scope here. Importing any
# ``eden.options`` submodule runs ``eden/options/__init__.py``, which reaches back into
# ``eden.utils`` (``eden.options.entities`` -> ``eden.utils.terrains.base``) and imports
# ``ConfigurableMixin`` from this module — a cycle whenever ``eden.utils.common`` is the first
# of the three to be imported (e.g. through ``eden.datasets.base``).

if TYPE_CHECKING:
    from eden.options.options import ConfigurableOptions

T = TypeVar("T", bound="ConfigurableOptions")


#: How deep :func:`frozen_value` follows nested containers, options models and dataclasses.
_FROZEN_DEPTH = 4


def frozen_value(value: Any, depth: int = 0) -> Hashable:
    """``value`` as a hashable constant that compares equal exactly when the value is the same configuration.

    For keying or fingerprinting what a term was configured with (:meth:`ConfigurableMixin.configured_fingerprint`).
    A scalar, string or enum is itself. A tensor is where it lives (its storage address, shape, strides and dtype),
    not its values: two terms reading one tensor are configured alike, and a tensor written in place is the same
    configuration while one rebound to another tensor is not. A numpy array is its values (a host array is read
    as a constant, so one edited in place is another configuration). A list, tuple, set or dict is its items,
    and an options model or a dataclass is its fields, so a value edited in place inside one is seen; past
    ``_FROZEN_DEPTH`` levels, and for any other object, a value is its identity.
    """
    if value is None or isinstance(value, (bool, int, float, str, Enum)):
        return value
    if isinstance(value, torch.Tensor):
        return ("tensor", value.data_ptr(), tuple(value.shape), value.stride(), value.dtype)
    if isinstance(value, np.ndarray):
        return ("ndarray", value.shape, value.dtype.str, value.tobytes())
    if isinstance(value, np.generic):
        return value.item()
    if depth >= _FROZEN_DEPTH:
        return ("object", id(value))
    if isinstance(value, (list, tuple)):
        return tuple(frozen_value(item, depth + 1) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(frozen_value(item, depth + 1) for item in value)
    if isinstance(value, dict):
        return tuple((key, frozen_value(item, depth + 1)) for key, item in value.items())
    if isinstance(value, BaseModel):
        items = [(name, getattr(value, name)) for name in type(value).model_fields]
        items += list((value.model_extra or {}).items())
        return (type(value), tuple((name, frozen_value(item, depth + 1)) for name, item in items))
    if is_dataclass(value) and not isinstance(value, type):
        return (
            type(value),
            tuple((f.name, frozen_value(getattr(value, f.name), depth + 1)) for f in dataclass_fields(value)),
        )
    return ("object", id(value))


def mirror_options_onto(obj: ConfigurableMixin, options: ConfigurableOptions) -> None:
    """Mirror every configurable parameter of ``obj`` from ``options`` onto ``obj`` itself.

    A parameter the options do not carry falls back to ``obj``'s own class attribute. Both
    sources are deep-copied, so a runtime object never shares a mutable value with

    * its **class** — otherwise every instance is handed the *same* list/dict, and mutating
      one instance leaks into all the others *and* permanently corrupts the class default
      for the rest of the process; or
    * its **options** — otherwise mutating the runtime attribute in place rewrites the
      config ``EnvBase.config`` serializes, and two objects built from one options instance
      (a config reused across envs) silently share state.

    :class:`Entity`, :class:`Sensor` and :class:`Camera` call this directly rather than
    delegating ``__init__`` to :class:`ConfigurableMixin`: their options also carry fields
    that are *consumed* instead of mirrored (morph construction, the nested Genesis
    sensor/camera options), which the mixin's unused-parameter check would reject.

    Two consequences of copying the options' side. A mirrored attribute and
    ``obj._options.<name>`` are independent from here on, so code that mutates one must not
    expect the other to follow — read a mirrored field off ``obj``, not off the options. And a
    built Genesis :class:`Surface` (the non-``SurfaceOptions`` half of ``SurfaceLike``) is
    *aliased*, not copied — see :func:`_copy_mirrored_value` for why.
    """
    # Field names only. ``keys()`` is a cached, value-free enumeration; ``model_dump()`` would
    # recursively serialize the entire options subtree (material, surface, every default_dofs_*
    # table) once per call, and emit serializer warnings for arbitrary-typed fields.
    option_fields = set(options.keys())
    # ``ManagerTermOptions.params`` defaults to ``None``, meaning "not a function term".
    # Mirroring that ``None`` would shadow the wrapper's ``{}`` class default and every function
    # term would die at its first ``compute()`` with "argument after ** must be a mapping", so an
    # empty ``params`` falls through to the class default. Same predicate the unused-parameter
    # check below uses, deliberately.
    if not getattr(options, "params", None):
        option_fields.discard("params")
    for name in obj.get_parameter_names():
        source = options if name in option_fields else obj
        setattr(obj, name, _copy_mirrored_value(getattr(source, name)))


def _copy_mirrored_value(value: Any) -> Any:
    """Deep-copy a value being mirrored onto a runtime object — *except* a built Genesis
    :class:`Surface`, which is aliased by reference.

    Deep-copying is what keeps a runtime object from sharing mutable state with its class default
    or its options. A ``gs.surfaces.*`` surface (the non-``SurfaceOptions`` half of ``SurfaceLike``)
    is the one value that must not be copied: it has no ``ConfigurableOptions.__deepcopy__``
    reconstruction path, so ``copy.deepcopy`` would clone any decoded texture it carries once per
    entity and raises outright on an uncopyable GPU/handle member. Nothing in-tree mutates a mirrored surface in
    place (it is read by ``resolve_surface`` or reassigned wholesale), so the alias is safe. An Eden
    ``SurfaceOptions`` is plain config and still deep-copies through the fall-through.
    """
    if isinstance(value, Surface):
        return value
    return copy.deepcopy(value)


class ConfigurableMixin(Generic[T]):
    """Bind a runtime object (term / entity) to its declarative ``Options`` class.

    The binding is reflection-driven, so subclasses just declare class attributes and get a
    matching options class "for free":

    * **Options class discovery** — :meth:`__init_subclass__` reads the ``ConfigurableMixin[T]``
      type argument off ``__orig_bases__`` and stores it as ``_options_class_``.
    * **Parameter discovery** — :meth:`get_parameter_names` walks the MRO and collects every
      plain (non-function, non-underscore, non-property/classmethod/staticmethod) class attribute,
      minus ``_non_parameter_names``; those are the configurable parameters.
    * **Options ↔ attribute mirroring** — :meth:`configure` packs the parameters into an
      ``_options_class_`` instance, and ``__init__`` mirrors them back onto the runtime object
      via :func:`mirror_options_onto` (deep-copying both the options' values and the class-level
      defaults, so an instance never aliases either).

    ``_extra_option_params`` (below) are the one documented exception to the mirroring.
    """

    _options_class_: type[T] | None = None

    #: Names of options fields that ``configure()`` accepts and stores in the options, but which are NOT
    #: mirrored as runtime instance attributes (so the runtime object can't read them). Default empty —
    #: a pure no-op for every configurable except those that opt in. Used by observation terms to keep the
    #: post-compute modifier fields (noise/clip/scale/history) off the term object while still configurable.
    _extra_option_params: ClassVar[frozenset[str]] = frozenset()

    #: Public class attributes that are class-level constants, not configurable parameters (e.g. a
    #: registry of strategy classes, or a capability flag the class hierarchy fixes). Default empty.
    #: Declarations accumulate over the MRO, so a subclass only lists the names it adds.
    _non_parameter_names: ClassVar[frozenset[str]] = frozenset()

    #: Parameters this class no longer accepts, each mapped to how to migrate a value written for it. They
    #: stay invalid — the note only joins the invalid-parameter error — so a config saved before a rename
    #: fails with directions instead of just the list of names accepted now. Declarations accumulate over the
    #: MRO (merged when the subclass is created), so a subclass lists only the names it adds, and a base's
    #: notes reach every subclass; a subclass's note for the same name wins.
    _retired_params: ClassVar[Mapping[str, str]] = MappingProxyType({})

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        retired: dict[str, str] = {}
        for base in reversed(cls.__mro__):
            retired.update(base.__dict__.get("_retired_params") or {})
        cls._retired_params = MappingProxyType(retired)
        if "_options_class_" not in cls.__dict__:
            for base in getattr(cls, "__orig_bases__", ()):
                origin = getattr(base, "__origin__", None)
                if origin is not None and issubclass(origin, ConfigurableMixin):
                    args = get_args(base)
                    if args and not isinstance(args[0], TypeVar):
                        cls._options_class_ = args[0]
                        break

    def __init__(self, options: T) -> None:
        self._uid = gs.UID()
        self._options: T = options

        mirror_options_onto(self, self._options)

        # Deferred, not module-scope — see the NOTE above this module's imports.
        from eden.options._metadata import OPTION_METADATA_KEYS

        # check if there are unused parameters in the options. ``_extra_option_params`` are legitimate
        # options-only fields (not mirrored as attrs), so they are expected to be "unused" here.
        unused_params = set(self._options.keys()) - set(self.get_parameter_names())
        tolerated = {"name", "class_name"} | set(OPTION_METADATA_KEYS) | self._extra_option_params
        # ``params`` is a declared framework field on function-capable term options. An *empty* one is
        # legitimately unused when the same options class configures a class-based term; a non-empty
        # ``params`` on a class-based term would be silently ignored, so it stays invalid.
        if not getattr(self._options, "params", None):
            tolerated.add("params")
        invalid_params = unused_params - tolerated
        if invalid_params:
            # Name the term, not only the class: for a function term the class is the shared
            # ``*FuncWrapper``, which cannot tell the author which of a dozen terms is wrong.
            # Same wording as the construction-time check (``ManagerTermOptions._validate_param_names``).
            term_name = getattr(self._options, "name", None)
            subject = f"term '{term_name}' ({type(self).__name__})" if term_name else type(self).__name__
            raise ValueError(
                invalid_parameters_message(subject, invalid_params, self.get_configure_specs(), self._retired_params)
            )

    @classmethod
    def get_parameter_names(cls) -> list[str]:
        params = set()
        excluded: set[str] = set()
        # Only collect attributes from classes in the ConfigurableMixin
        # hierarchy, skipping unrelated bases like nn.Module.
        for base_cls in cls.__mro__:
            excluded |= base_cls.__dict__.get("_non_parameter_names", frozenset())
            if base_cls is ConfigurableMixin or not issubclass(base_cls, ConfigurableMixin):
                continue
            for name, value in base_cls.__dict__.items():
                if name.startswith("__") or name.startswith("_"):
                    continue
                if isinstance(value, (property, cached_property, classmethod, staticmethod)):
                    continue
                if isinstance(value, FunctionType):
                    continue
                params.add(name)
        return list(params - excluded)

    @classmethod
    def fingerprint_names(cls, exclude: Iterable[str] = ()) -> tuple[str, ...]:
        """The parameters :meth:`configured_fingerprint` reads, name-sorted, leaving out ``exclude``.

        Resolving them walks the class's MRO (:meth:`get_parameter_names`). A caller that fingerprints one object
        on every step resolves them once and hands them to :meth:`configured_fingerprint` as ``names``.
        """
        excluded = frozenset(exclude)
        return tuple(name for name in sorted(cls.get_parameter_names()) if name not in excluded)

    def configured_fingerprint(
        self, exclude: Iterable[str] = (), names: tuple[str, ...] | None = None
    ) -> tuple[tuple[str, Hashable], ...]:
        """Every configured parameter's current value, by name, as :func:`frozen_value` makes it comparable.

        Two objects with equal fingerprints are configured alike right now; one fingerprint taken twice
        differs exactly when a parameter changed in between (a curriculum moving it, a nested options model
        edited in place, a tensor rebound). Name-sorted, so the pairs line up between two calls. ``exclude``
        leaves parameters out; ``names`` gives the parameters as :meth:`fingerprint_names` already resolved them,
        instead of ``exclude``.
        """
        excluded = frozenset(exclude)  # once: a generator is truthy even when it yields nothing
        if names is None:
            names = self.fingerprint_names(excluded)
        elif excluded:
            raise ValueError(
                f"configured_fingerprint of {type(self).__name__} takes exclude or names, not both: names are "
                f"already resolved without the excluded ones (fingerprint_names(exclude={sorted(excluded)!r}))."
            )
        return tuple((name, frozen_value(getattr(self, name, None))) for name in names)

    @classmethod
    def get_parameter_specs(cls) -> tuple[ParamInfo, ...]:
        """:meth:`get_parameter_names` with each parameter's declared type and default.

        Name-sorted, because ``get_parameter_names`` collects into a set and its order is
        therefore arbitrary — while an error message quoting the accepted set, and the generated
        catalogue page, both have to be reproducible.

        Nothing is ever reported ``required``: a class parameter *is* a class attribute, so it
        always has a default (an annotation-only declaration is not a parameter at all — see
        :meth:`get_parameter_names`). The required/optional split exists only for function terms,
        in :meth:`ConfigurableFuncWrapperMixin.get_function_parameter_specs`.
        """
        annotations: dict[str, Any] = {}
        # Reversed MRO so a subclass's annotation wins over the base's, and ``__dict__`` rather
        # than ``getattr`` because ``cls.__annotations__`` silently falls back to the nearest
        # annotated base — which would attribute a base's type to an unannotated override.
        for base_cls in reversed(cls.__mro__):
            annotations.update(base_cls.__dict__.get("__annotations__") or {})
        return tuple(
            ParamInfo(name, render_annotation(annotations.get(name)), render_default(getattr(cls, name)))
            for name in sorted(cls.get_parameter_names())
        )

    @classmethod
    def get_configure_specs(cls) -> tuple[ParamInfo, ...]:
        """Every parameter :meth:`configure` accepts — what an "invalid parameter" error must quote.

        :meth:`get_parameter_specs` plus the options-only :attr:`_extra_option_params`, whose type
        and default are read off the options class because they have no class attribute to read one
        from (that is exactly what makes them options-only).
        """
        fields = getattr(cls._options_class_, "model_fields", None) or {}
        extras = []
        for name in sorted(cls._extra_option_params):
            field = fields.get(name)
            if field is None:
                extras.append(ParamInfo(name))
                continue
            required = field.is_required()
            extras.append(
                ParamInfo(
                    name,
                    render_annotation(field.annotation),
                    # ``get_default`` (not ``field.default``) so a ``default_factory`` field renders
                    # its value, not the ``PydanticUndefined`` sentinel.
                    "" if required else render_default(field.get_default(call_default_factory=True)),
                    required=required,
                )
            )
        return cls.get_parameter_specs() + tuple(extras)

    @classmethod
    def _reject_invalid_parameters(cls, kwargs: dict[str, Any], subject: str | None = None) -> None:
        """Refuse ``configure()`` kwargs that are not parameters of ``cls``.

        Called by *both* ``_build_config_dict`` implementations. The func-wrapper one only ever reads
        the names it knows, so without this check a misspelled term-level field would be silently
        dropped: ``RewardTerm.configure(func=..., wieght=3.0)`` would return options carrying the
        default ``weight=1.0`` and train with it — the outcome ``extra="forbid"`` exists to prevent,
        and ``docs/guides/terms.md`` documents this exact call as raising.

        ``subject`` names the thing being configured in the message. For a class term (and the
        other configure targets — Camera, Sensor, Entity) the class *is* the subject, so it
        defaults to ``cls.__name__``. The func-wrapper passes the resolved term
        (``function term 'dofs_pos'``) because ``cls`` there is the shared ``*FuncWrapper``, whose
        repr tells the author nothing about which term they misconfigured.
        """
        invalid_params = set(kwargs) - set(cls.get_parameter_names()) - cls._extra_option_params
        if invalid_params:
            raise ValueError(
                invalid_parameters_message(
                    subject or cls.__name__, invalid_params, cls.get_configure_specs(), cls._retired_params
                )
            )

    @classmethod
    def _build_config_dict(cls, **kwargs) -> dict:
        if "name" in kwargs:
            raise ValueError("The 'name' parameter is reserved and cannot be used.")
        cls._reject_invalid_parameters(kwargs)
        config = {
            "name": to_snake_case(cls.__name__),
        }

        for name in cls.get_parameter_names():
            if name in kwargs:
                config[name] = kwargs[name]
            else:
                config[name] = copy.deepcopy(getattr(cls, name))
        # Options-only params: passed through to the options when provided, never read from a class attr.
        for name in cls._extra_option_params:
            if name in kwargs:
                config[name] = kwargs[name]
        return config

    @classmethod
    def configure(cls, **kwargs) -> T:
        """
        Configure the class with the given parameters.

        Parameters
        ----------
        **kwargs: dict
            The parameters to configure the class with.

        Returns
        -------
        T
            The options instance for the class.
        """
        config = cls._build_config_dict(**kwargs)
        if cls._options_class_ is None:
            raise TypeError(
                f"{cls.__name__} must parameterize ConfigurableMixin[T] with a concrete Options type "
                f"or set _options_class_ explicitly."
            )
        # ``name`` is framework-injected so managers can resolve terms by registry key
        # (``ManagerBase._build_term`` reads ``term_option.name``). Manager-term options always
        # subclass the extra="allow" ``*TermOptions`` bases and carry it as an extra. Every other
        # configure target (Camera, Sensor, Entity, extension options) never reads it, so for an
        # extra="forbid" options class that doesn't declare a ``name`` field we drop it instead of
        # tripping the forbid validation.
        options_class = cls._options_class_
        if options_class.model_config.get("extra") == "forbid" and "name" not in options_class.model_fields:
            config.pop("name", None)
        return options_class(**config)

    @property
    def name(self) -> str:
        return to_snake_case(self.__class__.__name__)

    @property
    def uid(self):
        return self._uid

    @property
    def options(self) -> T:
        return self._options


class ConfigurableFuncWrapperMixin(ConfigurableMixin[T]):
    #: Keyword arguments forwarded to the wrapped function on every ``compute()``. Declared here
    #: (not on ``ManagerTermFuncWrapperBase``) because ``get_parameter_names`` only collects
    #: attributes from ``ConfigurableMixin``-subclass MRO entries — this is what makes ``params``
    #: a configurable parameter of every function term.
    params: dict[str, Any] = {}

    #: The :class:`~eden.utils.registry.TermRegistry` this wrapper is registered into. Stamped by
    #: ``TermRegistry.__init__`` so ``configure(func=...)`` can verify the function is registered
    #: (build resolves the term by name, not by the callable object — see :meth:`_build_config_dict`).
    _term_registry: ClassVar[Any] = None

    #: Argument names the managers supply themselves — never expected in ``params``. ``env`` is passed
    #: by keyword by every func wrapper (:meth:`ManagerTermFuncWrapperBase.compute`); ``envs_idx`` is
    #: passed by the event manager. Excluded by name so a term declaring either as keyword-only (rather
    #: than positional, as the built-ins do) is not mis-flagged as missing a required param.
    _manager_injected_params: ClassVar[frozenset[str]] = frozenset({"env", "envs_idx"})

    def __init__(self, options: T) -> None:
        super().__init__(options=options)
        # Build-time backstop for ``params`` KEYS: construction and assignment validate them
        # (ManagerTermOptions), but IN-PLACE mutation of the raw params dict
        # (``options.params["typo"] = ...``) is invisible to ``__setattr__``. Catching it here
        # names the term at env build instead of a bare TypeError at the first ``compute()``.
        # ``_func`` is set by ManagerTermFuncWrapperBase.__init__, which every concrete wrapper
        # calls first; a missing func defers to the compute-time error.
        func = getattr(self, "_func", None)
        if func is None:
            return
        invalid_keys = invalid_function_param_keys(func, self.params)
        if invalid_keys:
            raise ValueError(
                invalid_params_message(options.name, invalid_keys, self.get_function_parameter_specs(func))
            )

    @classmethod
    def get_function_parameter_specs(cls, func: FunctionType) -> tuple[ParamInfo, ...]:
        """The ``params={...}`` entries an author supplies for ``func``, typed and defaulted.

        The listable counterpart of :meth:`get_required_function_parameter_names`, which returns
        only the subset a *rejection* may be based on. This is wider on purpose — see
        :func:`~eden.utils.params.function_param_specs` — because describing a term means naming
        every key it accepts, not only the ones ``configure()`` will refuse to go without.
        """
        return function_param_specs(func, exclude=cls._manager_injected_params)

    @classmethod
    def get_required_function_parameter_names(cls, func: FunctionType) -> set[str]:
        """Parameters of ``func`` without a default that the author must supply via ``params``.

        Keyword-only and positional-or-keyword params both count (``compute`` calls
        ``func(env=..., **params)``, so either is supplied by keyword), matching what
        :meth:`get_function_parameter_specs` advertises as required. ``_manager_injected_params``
        (``env``, ``envs_idx``) are excluded.
        """
        return {
            name
            for name, param in inspect.signature(func).parameters.items()
            if param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
            and param.default is inspect.Parameter.empty
        } - cls._manager_injected_params

    @classmethod
    def _build_config_dict(cls, func: FunctionType = None, **kwargs) -> dict:
        if "name" in kwargs:
            raise ValueError("The 'name' parameter is reserved and cannot be used.")
        # Guarded rather than left to ``func.__name__``: a term *name* where the function belongs
        # (``func="dofs_vel_limits"``) is the natural mistake to make from a config that reads
        # ``name: dofs_vel_limits``, and would otherwise surface as a bare
        # ``AttributeError: 'str' object has no attribute '__name__'`` from inside this function.
        # ``FunctionType`` and not merely callable, because that is what a term registry accepts as
        # a function term (``TermRegistry._register_term``) — a partial or a callable instance could
        # not have been registered, so it could not resolve at build either.
        if func is not None and not isinstance(func, FunctionType):
            raise TypeError(
                f"func must be the term function itself for {cls.__name__}, got "
                f"{type(func).__name__}: {func!r} — pass the callable (e.g. func=rewards.dofs_vel_limits), "
                f"not its name."
            )
        term_name = to_snake_case(func.__name__) if func is not None else to_snake_case(cls.__name__)
        # A term-level typo (``configure(func=..., wieght=3.0)``) must name the *term*, not the
        # shared ``*FuncWrapper`` that ``cls`` is. Falls back to ``cls.__name__`` when ``func`` is
        # absent (an unusual call the missing-func path handles downstream).
        cls._reject_invalid_parameters(kwargs, subject=f"function term '{term_name}'" if func is not None else None)
        # Build resolves the term by name via the registry, so an unregistered function would fail
        # only at build time with a bare KeyError. Fail loudly here instead. A function registered
        # under a *different* key cannot reach this point: ``TermRegistry.register`` takes no
        # ``name=`` (a term's key is always ``to_snake_case(__name__)``), which is what makes the key
        # this looks up and the key the term was registered under the same thing.
        if func is not None and cls._term_registry is not None and term_name not in cls._term_registry:
            raise ValueError(
                f"function '{func.__name__}' is not registered — add @{cls._term_registry._name}_REGISTRY.register()"
            )
        # Defaults come off the class the term is *built* as. ``cls`` is the kind's shared wrapper,
        # which knows nothing of ``@REGISTRY.register(direction=...)`` on this function: those land
        # on the per-registration wrapper subclass.
        defaults_cls = cls
        if func is not None and cls._term_registry is not None:
            defaults_cls = cls._term_registry.registered_class(term_name)
        config = {"name": term_name}

        for name in cls.get_parameter_names():
            if name in kwargs:
                if name == "params":
                    # Not an ``assert``: ``python -O`` strips those, and the next line would then
                    # raise AttributeError on a list — or silently accept the wrong parameter set
                    # from anything else carrying ``.keys()``.
                    if not isinstance(kwargs[name], dict):
                        raise TypeError(
                            f"params must be a dict mapping parameter name to value for term "
                            f"'{term_name}', got {type(kwargs[name]).__name__}: {kwargs[name]!r}"
                        )
                    # Same check as the construction-time and build-time ones, through the same
                    # helper, so the three cannot disagree about which keys are legal or say so
                    # differently.
                    invalid_keys = invalid_function_param_keys(func, kwargs[name])
                    if invalid_keys:
                        raise ValueError(
                            invalid_params_message(term_name, invalid_keys, cls.get_function_parameter_specs(func))
                        )
                    # NOTE: update provided params for the missing ones
                    params = {**defaults_cls.params}
                    params.update(kwargs[name])
                    kwargs[name] = params
                config[name] = kwargs[name]
            else:
                config[name] = copy.deepcopy(getattr(defaults_cls, name))
        # Options-only params (e.g. obs post-compute modifiers): passed through to the options when provided.
        for name in cls._extra_option_params:
            if name in kwargs:
                config[name] = kwargs[name]

        if func is not None:
            missing_params = cls.get_required_function_parameter_names(func) - set(config.get("params", {}))
            if missing_params:
                raise ValueError(
                    f"missing required param(s) {sorted(missing_params)} for function '{func.__name__}' — "
                    f"supply them via params={{...}}. Accepted params: "
                    f"{render_params(cls.get_function_parameter_specs(func))}"
                )
        return config

    @classmethod
    def configure(cls, func: FunctionType, **kwargs) -> T:
        """
        Configure the class with the given parameters.

        Parameters
        ----------
        func: FunctionType
            The function to configure the class with.
        **kwargs: dict
            The parameters to configure the class with.

        Returns
        -------
        T
            The options instance for the class.
        """
        config = cls._build_config_dict(func=func, **kwargs)
        if cls._options_class_ is None:
            raise TypeError(
                f"{cls.__name__} must parameterize ConfigurableMixin[T] with a concrete Options type "
                f"or set _options_class_ explicitly."
            )
        return cls._options_class_(**config)

    @property
    def name(self) -> str:
        return to_snake_case(self._func.__name__)
