"""Base manager-options class with dynamic per-term fields."""

import copy
import difflib
import importlib
from functools import partial
from types import UnionType
from typing import Any, ClassVar, Generic, TypeVar, get_args

from pydantic import field_validator, model_serializer, model_validator

from eden.options.options import ConfigurableOptions, DynamicConfigurableOptions
from eden.utils.params import (
    invalid_function_param_keys,
    invalid_parameters_message,
    invalid_params_message,
    unknown_parameter_names,
)


T_TermOptions = TypeVar("T_TermOptions", bound=ConfigurableOptions)

#: Every module that constructs a :class:`~eden.utils.registry.TermRegistry`, cheapest first.
#: Constructing one stamps ``_term_registry_`` onto the kind's options class, and ``import eden`` does
#: not import these for that side effect — see :meth:`ManagerTermOptions._resolve_term_registry`.
#: This is the ``module`` column of :data:`eden.managers.catalog.KINDS`, which
#: ``tests/managers/test_catalog.py`` pins as covering every registry in the tree; spelled out rather
#: than read from the catalog so the walk can stop early instead of importing all three (the
#: motion manager is ~0.2 s and one options class in thirteen needs it).
_TERM_REGISTRY_MODULES = ("eden.managers", "eden.managers.modifiers.base", "eden.managers.motion_manager")

#: The package every kind's term-options class is declared in: ``eden/options/managers/<kind>.py``
#: for the manager kinds, ``modifiers.py`` for the three modifier ones. Membership is what
#: :meth:`ManagerTermOptions._stamped_by_a_registry` reads to tell "a registry has not got to this
#: class yet" from "no registry will ever claim it", and ``tests/options/managers/test_base.py`` pins it
#: against the stamps themselves so a kind filed anywhere else fails a test rather than silently
#: losing its construction-time parameter check. This module is excluded deliberately: the classes
#: *here* are the shared bases and :class:`FunctionTermOptions`, which no registry stamps.
_TERM_OPTIONS_PACKAGE = "eden.options.managers."


_UNSET = object()


class ManagerTermOptions(DynamicConfigurableOptions):
    """Base for all per-term options (``RewardTermOptions``, ``EventTermOptions``, ...).

    Term options stay ``extra="allow"`` because class-based terms carry their per-term
    parameters as extras (the options class is shared by every term of the kind). Typo
    protection comes from validating those extras against the registered term instead:

    * **At construction and on attribute assignment** (``model_post_init`` /
      ``__setattr__``): when ``name`` resolves in the kind's term registry — stamped onto
      this class as ``_term_registry_`` by ``TermRegistry.__init__`` — extras must be
      parameters of the registered term, and for function terms the ``params`` keys must
      be arguments of the registered function. ``RewardTermOptions(name="upright",
      wieght=2.0)`` and ``options.wieght = 2.0`` both raise in the config file that
      contains the typo (a rejected assignment is rolled back).
    * **On copy-with-changes** (``updated(...)`` / ``model_copy(update=...)``): the same check
      runs on the copy — pydantic's ``update=`` is otherwise unvalidated and would park
      ``{"wieght": 2.0}`` as an extra next to the untouched ``weight``.
    * **At build** (``ConfigurableMixin.__init__``): the pre-existing invalid-parameter
      check backstops options whose ``name`` was missing or not yet registered at
      construction time.
    """

    #: Registry key of the term these options configure; injected by ``configure()`` and
    #: read by ``ManagerBase._build_term`` to resolve the term class.
    name: str | None = None

    #: The kind's term registry, stamped by ``TermRegistry.__init__`` when the manager module
    #: is imported. Read through :meth:`_resolve_term_registry`, never directly: ``import eden``
    #: does not import the manager modules, so an unstamped class here means "not imported yet",
    #: not "no registry".
    _term_registry_: ClassVar[Any] = None

    @classmethod
    def _resolve_term_registry(cls) -> Any:
        """This kind's :class:`~eden.utils.registry.TermRegistry`, importing it if it is not loaded.

        ``TermRegistry.__init__`` stamps :attr:`_term_registry_` onto the options class as a side
        effect of the registry being *constructed*, and ``import eden`` imports no manager module:
        ``eden.options.managers.rewards`` does not reach ``eden.managers.reward_manager``, so without
        this a ``RewardTermOptions`` built from a saved config could find no registry and skip the
        construction-time parameter check in silence — the failure would resurface at env build,
        far from the config that caused it.

        The modules are tried in order and the walk stops as soon as *this* class is stamped, so
        the nine manager kinds pay one ~4 ms import (``eden.options`` is already loaded by the
        time this can run) and only the motion kind pays for the motion manager.

        One case is *not* "no registry claims this class": a module that is already mid-import on
        this thread comes back from ``import_module`` half-built, so its ``TermRegistry(...)`` may
        not have run yet and the stamp is legitimately not there *yet*. Falling through to
        ``None`` there would skip the check silently — the exact failure this method exists to
        prevent — so that case raises.

        It raises only for a class some registry *would* stamp
        (:meth:`_stamped_by_a_registry`). The unfinished module is shared by up to nine kinds,
        so keying the raise on "any of the three is initializing" would also fire for a class none
        of them can ever claim — :class:`FunctionTermOptions`, or anything out of tree — where
        ``None`` is the right answer and every sentence of the message below is false of the
        caller.

        Raising costs nothing today and is not reachable by accident: nothing in ``eden/``
        constructs term options at module scope, and out-of-tree code cannot reach the case at
        all, because running a module body inside Eden's own import is the only way in and
        ``eden/managers/`` imports no plugins or entry points. It is a tripwire for the change
        that would make it reachable, and a raise is what puts that change's own frames in front
        of whoever wrote it — a warning cannot, since the construction runs inside pydantic and
        every fixed ``stacklevel`` lands on this module instead.

        There is deliberately **no negative cache**: an unclaimed class re-walks the three module
        names on every construction (~12 µs, against ~0.1 µs once stamped, twice per
        construction via :meth:`model_post_init`). Caching the ``None`` is what would break it —
        "no registry claims this" and "its module is not imported yet" are the same observation
        here, and memoizing the first answer would make a class that *is* claimed skip its check
        forever once anything had constructed it early. Only :class:`FunctionTermOptions` is
        unclaimed in tree, so the cost is nil; if that changes, cache per *class* on the first
        **successful** resolution, never on ``None``.

        Returns
        -------
        TermRegistry | None
            The registry, or ``None`` for an options class no registry claims — an out-of-tree
            class whose registry has genuinely not been created. The build-time check still
            applies there.

        Raises
        ------
        RuntimeError
            If a module that would stamp this class is still importing on this thread, so the
            answer is "not yet" rather than "never".
        """
        registry = getattr(cls, "_term_registry_", None)
        if registry is not None:
            return registry
        # Deferred, like the imports below: nothing from ``eden.utils`` may become a module-scope
        # import here — see the NOTE in ``eden/utils/common.py`` on the options <-> utils cycle.
        from eden.utils.registry import module_is_initializing

        unfinished: list[str] = []
        for module_name in _TERM_REGISTRY_MODULES:
            module = importlib.import_module(module_name)
            registry = getattr(cls, "_term_registry_", None)
            if registry is not None:
                return registry
            if module_is_initializing(module):
                unfinished.append(module_name)
        if unfinished and cls._stamped_by_a_registry():
            raise RuntimeError(
                f"{cls.__name__} was constructed while {', '.join(unfinished)} "
                f"{'is' if len(unfinished) == 1 else 'are'} still being imported on this thread, so "
                f"TermRegistry.__init__ has not stamped _term_registry_ onto the class yet and the "
                f"construction-time parameter check has nothing to validate against — a misspelled "
                f"parameter on this options object would go unreported until env build. Cause: term "
                f"options are being constructed at module scope inside eden.managers.*; move that "
                f"construction into a function or a task config, which runs once eden.managers has "
                f"finished importing. Do not wrap this in try/except and do not make it fall back to "
                f"skipping the check — losing the check is the failure being reported here."
            )
        return None

    @classmethod
    def _stamped_by_a_registry(cls) -> bool:
        """Whether some ``TermRegistry`` claims this class, stamp present or not yet.

        Read by :meth:`_resolve_term_registry` to tell the two reasons :attr:`_term_registry_`
        can be missing apart while a manager module is still importing: "that registry has not
        been constructed yet" (raise — the check has nothing to validate against) from "no
        registry will ever claim this class" (return ``None`` — there is nothing to wait for).

        Answered from where the class is *declared*, because the thing that would answer it
        directly — the registry — is precisely what has not been constructed yet. Every kind's
        options class lives in a :data:`_TERM_OPTIONS_PACKAGE` module, so a class with one of
        those in its MRO is a kind's (or a subclass of one); the shared bases and
        :class:`FunctionTermOptions` live in *this* module and are claimed by nobody.
        ``tests/options/managers/test_base.py`` pins the answer against the actual stamps.

        Returns
        -------
        bool
            Whether a registry claims this class.
        """
        return any(
            base.__module__.startswith(_TERM_OPTIONS_PACKAGE) and base.__module__ != __name__ for base in cls.__mro__
        )

    @model_validator(mode="before")
    @classmethod
    def _default_from_registered_term(cls, data: Any) -> Any:
        """Fill every field the caller left out from the registered term, not from this class.

        These options are shared by every term of the kind, so their field defaults are the kind's
        (``MetricTermOptions.direction`` is ``HIB``). A term that says otherwise — a class
        attribute on a class term, or ``@METRIC_TERM_REGISTRY.register(direction=MetricDirection.LIB)``
        on a function term — would lose to them at build, where ``mirror_options_onto`` copies
        every field the options carry over the term's own class attribute. ``configure()`` never
        had the problem for a class term, since it reads the class it is called on, but
        ``*TermOptions(name=...)`` did, and so, for a function term, did ``configure(func=...)``
        (``entity_distance`` would score the far cube as the success). Resolving the default here
        makes the options record the value the term runs with, so the saved config says so too.

        A field the options declare *required* is left alone, so leaving it out still raises: a
        term's placeholder class attribute (an action term's ``entity_name = ""``) is not a value.
        ``params`` is left alone as well — a function term's wrapper defaults it to ``{}`` and the
        options' ``None`` means "not a function term", which :func:`mirror_options_onto` handles.

        Parameters
        ----------
        data : Any
            The construction input. Only a mapping whose ``name`` is a registered key is changed.

        Returns
        -------
        Any
            ``data``, with the missing fields added from the registered term's class.
        """
        if not isinstance(data, dict) or not data.get("name"):
            return data
        registry = cls._resolve_term_registry()
        if registry is None or data["name"] not in registry:
            return data
        term_cls = registry.registered_class(data["name"])
        missing = {
            field
            for field in set(term_cls.get_parameter_names()) - set(data) - {"name", "params"}
            if field in cls.model_fields and not cls.model_fields[field].is_required()
        }
        if not missing:
            return data
        return {**data, **{field: copy.deepcopy(getattr(term_cls, field)) for field in missing}}

    def model_post_init(self, context) -> None:
        super().model_post_init(context)
        self._hoist_params_onto_class_term()
        self._validate_param_names()

    def _hoist_params_onto_class_term(self) -> None:
        """Move a function-era ``params`` dict onto the top-level fields of a now class-based term.

        A function term converted to a class term keeps its registry key, but a config saved
        before the conversion still carries ``params={"metric_name": ...}`` — which a class term
        never reads, and which :meth:`_validate_param_names` therefore rejects. That would strand
        every checkpoint's sibling config on the day a built-in term gains a ``build()`` (as
        ``metric_reward`` / ``metric_observation`` / ``on_metric_success`` did). Hoisting applies the
        saved values instead of dropping them.

        Opt-in per term, not a general spelling: only a class that declares
        ``_function_era_params`` (see :class:`~eden.managers.base.ManagerTermBase`) is hoisted, and
        only when every key is one of those. Anything else — ``params`` on a term that was never a
        function term, or a key the function never took — is left in place for
        :meth:`_validate_param_names` to reject with its class-term message. Construction only:
        assigning ``options.params`` afterwards is an authoring mistake and stays an error.
        """
        params = getattr(self, "params", None)
        registry = type(self)._resolve_term_registry()
        if not params or registry is None or not self.name or self.name not in registry:
            return
        term_cls = registry.get(self.name)
        if isinstance(term_cls, partial):
            return
        function_era_params = getattr(term_cls, "_function_era_params", frozenset())
        if not function_era_params or not set(params) <= function_era_params:
            return
        extras = self.__pydantic_extra__
        clashing = sorted(key for key in params if key in extras or key in type(self).model_fields)
        if clashing:
            self._raise_invalid(
                f"term '{self.name}' ({term_cls.__name__}) is class-based, and {clashing} in 'params' collide with "
                f"top-level options fields — pass each parameter once, as a top-level field."
            )
        extras.update(params)
        self.__dict__["params"] = None

    def __setattr__(self, key, value):
        """Re-validate parameter names on post-construction assignment.

        ``model_post_init`` only guards ``__init__``; without this, ``options.wieght = 2.0``
        would park an unvalidated extra that surfaces only at term build. A rejected
        assignment is rolled back so the options object is left exactly as it was.
        """
        if key.startswith("_"):
            super().__setattr__(key, value)
            return
        # Declared fields and extras live in different dicts but share one restore rule.
        container = self.__dict__ if key in type(self).model_fields else self.__pydantic_extra__
        previous = container.get(key, _UNSET) if container is not None else _UNSET
        super().__setattr__(key, value)
        try:
            self._validate_param_names()
        except ValueError:
            if container is not None:
                if previous is _UNSET:
                    container.pop(key, None)
                else:
                    container[key] = previous
            raise

    def _validate_copy_update(self, update) -> None:
        """Hold ``model_copy(update=...)`` to the construction rule.

        Term options are ``extra="allow"``, so pydantic parks ``update={"weigth": 0.0}`` as an
        extra and leaves ``weight`` alone. Re-running the parameter-name check on the copy rejects
        it on the line that contains the typo, like construction and assignment do.
        """
        self._validate_param_names()

    @model_serializer(mode="wrap")
    def _serialize_without_empty_framework_fields(self, handler):
        """Drop ``name``/``params`` from dumps when they carry no information.

        Both are framework fields present on every term options; emitting ``name: null`` or
        ``params: {}`` (every class-based term) only adds noise to every saved config.
        """
        data = handler(self)
        if isinstance(data, dict):
            if data.get("name") is None:
                data.pop("name", None)
            if not data.get("params"):
                data.pop("params", None)
        return data

    def _validate_param_names(self) -> None:
        registry = type(self)._resolve_term_registry()
        if registry is None or not self.name or self.name not in registry:
            return
        registered = registry.get(self.name)
        if isinstance(registered, partial):
            # Function term: the registry stores ``partial(wrapper_cls, func=obj)``.
            term_cls, func = registered.func, registered.keywords.get("func")
        else:
            term_cls, func = registered, None

        # The term is registered (checked above), so the catalog has a row for it — and for a
        # function term the accepted set below is the *wrapper's* fields, which say nothing about
        # what belongs in ``params``. The command that shows both is worth the line.
        catalog_hint = f". {registry.catalog_hint(self.name)}"

        invalid = unknown_parameter_names(self.__pydantic_extra__ or {}, term_cls, tolerated=("class_name",))
        if invalid:
            self._raise_invalid(
                invalid_parameters_message(
                    f"term '{self.name}' ({term_cls.__name__})",
                    invalid,
                    term_cls.get_configure_specs(),
                    term_cls._retired_params,
                )
                + catalog_hint
            )

        if func is not None:
            invalid_keys = invalid_function_param_keys(func, getattr(self, "params", None))
            if invalid_keys:
                self._raise_invalid(
                    invalid_params_message(self.name, invalid_keys, term_cls.get_function_parameter_specs(func))
                    + catalog_hint
                )
        elif getattr(self, "params", None) and unknown_parameter_names(("params",), term_cls):
            # ``params`` is only consumed by function terms; on a class-based term it would be
            # silently ignored — the exact failure mode this validator exists to prevent.
            self._raise_invalid(
                f"'params' is only used by function-based terms, but '{self.name}' ({term_cls.__name__}) "
                f"is class-based — pass its parameters as top-level options fields instead."
            )

    @staticmethod
    def _raise_invalid(message: str) -> None:
        """Fail loud on an invalid term parameter — at construction and at load alike.

        The load path validates eden options (``_safe_model_construct`` -> ``model_validate``), so a
        stale/misspelled term parameter in a saved config surfaces here at load time with the same
        message as live construction — never silently deferred to term build.
        """
        raise ValueError(message)


class FunctionTermOptions(ManagerTermOptions):
    """Term options for the function-capable kinds — the ones with a ``*TermFuncWrapper``.

    Declares the ``params`` dict that ``configure(func=...)`` fills and the wrapper's
    ``compute()`` forwards to the wrapped function. Kinds whose terms are class-based only
    (action, command, recorder, motion and the three modifier kinds) subclass :class:`ManagerTermOptions`
    directly; a new kind that grows a function wrapper should switch its options base to
    this class.
    """

    #: Keyword arguments forwarded to the wrapped function; ``None`` on class-based terms
    #: (``configure(func=...)`` always sets a dict for function-based terms).
    params: dict[str, Any] | None = None

    @field_validator("params", mode="plain")
    @classmethod
    def _keep_params_raw(cls, value):
        """Keep ``params`` exactly as passed (same pattern as the ``morph`` escape hatch).

        Schema validation would rebuild the dict; the caller's object is kept untouched. ``None``
        means "not a function term". ``_validate_param_names`` checks the *keys* against
        the registered function; the values are the function's business.
        """
        if value is None or isinstance(value, dict):
            return value
        raise ValueError(f"params must be a dict of keyword arguments, got {type(value).__name__}")


class ManagerOptions(DynamicConfigurableOptions, Generic[T_TermOptions]):
    """
    Generic base class for manager options.

    Stores term options as extra fields, validates their types via ``model_post_init``,
    and provides ``disable_term`` for runtime removal.

    Usage::

        class EventManagerOptions(ManagerOptions[EventTermOptions]):
            pass

        # Entries forming a union parametrize with `|` — accepted types stay visible
        # in the class signature (e.g. reward managers accept flat terms or groups):
        class RewardManagerOptions(ManagerOptions[RewardTermOptions | RewardGroupOptions]):
            pass

    The term options class (or union) is automatically extracted from the generic
    parameter; ``isinstance`` accepts union types directly.
    """

    _term_options_class_: type | UnionType | None = None
    #: What an entry is called in messages; ``ObservationManagerOptions`` holds groups, not terms.
    _entry_noun: ClassVar[str] = "term"

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Pydantic rewrites __orig_bases__, so extract the type arg from __bases__
        # via __pydantic_generic_metadata__ on the parametrized base class.
        for base in cls.__bases__:
            meta = getattr(base, "__pydantic_generic_metadata__", None)
            if meta and meta.get("origin") is ManagerOptions and meta.get("args"):
                cls._term_options_class_ = meta["args"][0]
                break

    def _entry_type_message(self, key: str, value: Any) -> str:
        """Error message for a term entry that failed the type check.

        Presentation hook only — the check itself lives in :meth:`_validate_entry_types`, and
        this is only invoked when the generic term-options parameter is set
        (``_term_options_class_`` is not ``None``). Subclasses override to add context
        (e.g. the reward manager names the group a mistyped flat-name override actually
        lives in).
        """
        term_cls = self._term_options_class_
        allowed = " or ".join(t.__name__ for t in (get_args(term_cls) or (term_cls,)))
        message = f"Term '{key}' must be an instance of {allowed}, got {type(value).__name__}"
        # A declared field misspelled (``dataset_filname``) lands in the extras beside the terms and
        # is reported as a term of the wrong type — name the field it most likely meant.
        close = difflib.get_close_matches(key, type(self).model_fields, n=1)
        return f"{message}. Did you mean the field '{close[0]}'?" if close else message

    def model_post_init(self, context) -> None:
        super().model_post_init(context)
        self._validate_entry_types()

    def _validate_copy_update(self, update) -> None:
        """Type-check entries added or replaced through ``model_copy(update=...)``."""
        self._validate_entry_types()

    def __setattr__(self, name: str, value: Any) -> None:
        """Type-check an entry *assigned* onto constructed manager options, as construction does.

        ``reward_options.<term> = ...`` is the natural way to add a term to an inherited config, and
        it is the one entry into the extras that skips both ``model_post_init`` and
        :meth:`_validate_copy_update`, so without this ``reward_options.bogus = "x"`` would be
        accepted and die at build, far from the line that wrote it. A declared field and a private attribute are left to
        pydantic; a refused entry is rolled back, so the options are left exactly as they were.
        """
        if name.startswith("_") or name in type(self).model_fields:
            super().__setattr__(name, value)
            return
        if not self._locked:  # a locked config raises its own RuntimeError below, ahead of any mutation
            self._refuse_shadowing_child(name, noun=self._entry_noun, assigned=True)
        extra = self.__pydantic_extra__
        previous = _UNSET if extra is None else extra.get(name, _UNSET)
        super().__setattr__(name, value)
        try:
            self._validate_entry_types()
        except Exception:
            extra = self.__pydantic_extra__
            if previous is _UNSET:
                extra.pop(name, None)
                # pydantic records an extras write as a set field; a name left behind there is a
                # field the options do not have.
                self.__pydantic_fields_set__.discard(name)
            else:
                extra[name] = previous
            raise

    def _validate_entry_types(self) -> None:
        extra = self.__pydantic_extra__ or {}
        for key in extra:
            if not key.startswith("_option_"):
                self._refuse_shadowing_child(key, noun=self._entry_noun)
        term_cls = self._term_options_class_
        if term_cls is None:
            return
        for key, value in extra.items():
            if key.startswith("_option_"):
                continue
            if not isinstance(value, term_cls):
                # Unconditionally hard, including on the load path: an entry that failed type
                # recovery can't be safely kept (the manager would crash on it later, far from
                # the cause), and `with_overrides_from_dict` — a live, user-authored operation —
                # routes through here too and needs a loud failure. The serialization round-trip
                # test pins the dump shape this relies on.
                raise TypeError(self._entry_type_message(key, value))

    def term_keys(self) -> tuple[str, ...]:
        """Return only the registered ``ManagerTerm`` entry names.

        These are the extras validated by :meth:`model_post_init` to be
        ``_term_options_class_`` instances.

        This is intentionally distinct from :meth:`ConfigurableOptions.keys`,
        which returns every named child of the options object — including
        declared model fields that are config knobs (e.g.
        ``RecorderManagerOptions.dataset_filename``), not terms.
        """
        extra = self.__pydantic_extra__ or {}
        return tuple(key for key in extra if not key.startswith("_option_"))

    def disable_term(self, name: str) -> None:
        """Remove a term from this manager options by name.

        Raises ``KeyError`` (listing the terms that exist) for an unknown name and
        ``RuntimeError`` on a locked config.
        """
        self._remove_named_child(name, noun="term")
