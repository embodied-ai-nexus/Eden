"""Generic Registry plus term/variant registries for pluggable components.

A :class:`Registry` maps name -> object so user code can register custom terms, tasks,
robots, etc. Keys default to ``to_snake_case(obj.__name__)``; pass ``name=`` to choose a
different one. Two naming caveats: keys starting with ``_`` emit a warning, and class names
with a digit suffix (e.g. ``Foo1``) do **not** gain an underscore before the digit.

Configurables (terms, file handlers, dataset modifiers, terrain generators — anything whose
``.configure()`` stamps ``options.name`` and is resolved back out of its registry by that name)
are the exception: their key *is* ``to_snake_case(__name__)``, so :class:`TermRegistry` takes no
``name=`` at all, and :class:`Registry` refuses one that would alias a configurable. See
:func:`_key_is_derived_elsewhere`.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Any, Iterable, Iterator, Sequence, Type

# Modified from fvcore
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import difflib
import importlib
import threading
import warnings
from types import FunctionType, TracebackType
from functools import partial
import inspect

from eden.utils.misc import to_snake_case

if TYPE_CHECKING:
    from eden.managers.base import ManagerTermFuncWrapperBase, ManagerTermBase


__all__ = ["Registry", "TermRegistry", "VariantRegistry"]


# The reason both refusals in this module give, worded once.
#
# One rule — a configurable's key is its own name — reached from two entry points, because the
# entry points can refuse different things at different times: :meth:`TermRegistry.register` runs
# as the decorator *expression*, before the decorated object exists, so it can name neither the
# class nor the derived key and refuses any ``name=`` at all; :func:`_derive_name` runs with the
# object in hand for the registries that must keep accepting ``name=`` for their other entries
# (assets, robots, tasks), so it refuses only a name that *diverges*. Collapsing them would mean
# one of the two giving up its semantics. What they must never disagree on is the cause, so the
# cause lives here and neither site restates it.
_WHY_THE_KEY_IS_THE_NAME = (
    "to_snake_case(__name__) is what .configure() stamps into options.name, and what the manager "
    "resolves it back out of the registry by"
)


def module_is_initializing(module: Any) -> bool:
    """Whether ``module`` is still executing its own body — a partial import, not a finished one.

    ``importlib.import_module`` returns immediately, with the half-built module, when the calling
    thread is already importing it; only a *different* thread blocks until the body is done. This
    is the flag the import system itself reads to tell those apart
    (``importlib._bootstrap._find_and_load``), and two callers need it for the same reason — to
    know whether what they just got back is complete enough to draw a conclusion from:
    :meth:`Registry._populate` before it memoizes ``_populated``, and
    ``ManagerTermOptions._resolve_term_registry`` before it reports "no registry claims this
    class".

    ``ModuleSpec._initializing`` is private, hence the defensive ``getattr`` chain — though not
    obscure: it has been there since the spec protocol landed in 3.4. Absent (or an object with
    no ``__spec__``, such as a C extension loaded the old way) reads as finished.

    Parameters
    ----------
    module : ModuleType
        The module :func:`importlib.import_module` returned.

    Returns
    -------
    bool
        ``True`` while the module's body is still running.
    """
    return bool(getattr(getattr(module, "__spec__", None), "_initializing", False))


def _describe(obj: Any) -> str:
    """``module.name`` for ``obj``, for use in an error message.

    Two collisions are only ever fixable by finding the two definitions, so the import path is the
    useful part — a bare name is ambiguous exactly when it matters (both sides of a duplicate share
    it). Function terms are stored as ``partial(wrapper, func=obj)``, whose repr is two class paths
    and a memory address, so the wrapped function is unwrapped first.
    """
    if isinstance(obj, partial):
        obj = obj.keywords.get("func", obj.func)
    name = getattr(obj, "__name__", None)
    if name is None:
        return repr(obj)
    module = getattr(obj, "__module__", None)
    return f"{module}.{name}" if module else name


def _key_is_derived_elsewhere(obj: Any) -> bool:
    """Whether ``obj``'s registry key is *also* derived from its ``__name__`` somewhere else.

    :meth:`~eden.utils.common.ConfigurableMixin.configure` stamps
    ``options.name = to_snake_case(__name__)``, and every manager resolves the object back out
    of its registry by that stamped name (``registry.get(term_option.name)``). So for a
    configurable the key is not free: registering it under anything else makes the config
    ``configure()`` builds name a key that is either absent from the registry (``KeyError`` at
    env build) or owned by a *different* object (silently the wrong term).

    Everything else a registry holds — assets, robots, tasks — is only ever looked up by its
    key, so those may be named freely.

    Being a ``ConfigurableMixin`` is the *proxy*, not the property: it deliberately over-applies to
    the five configurable registries that are looked up by a caller-chosen key rather than a stamped
    name (``POLICY``, ``DEPLOYMENT``, ``SYSID``, ``GRASP_SAMPLER``, ``MOTION_PLANNER`` — e.g.
    ``_resolve_extension_policy`` resolves a ``--policy`` argument against ``POLICY``), where an
    alias would in fact be harmless. That direction is the safe one — refusing an alias costs
    expressiveness, never correctness — and it keeps the rule automatic rather than a per-registry
    declaration to forget.

    ``ConfigurableMixin`` is imported here rather than at module scope: ``eden.utils.common``
    reaches this module transitively, and this runs once per registration.
    """
    from eden.utils.common import ConfigurableMixin

    return inspect.isclass(obj) and issubclass(obj, ConfigurableMixin)


def _derive_name(obj: Any, explicit_name: str | None, registry_name: str) -> str:
    """Return the registry key for ``obj``.

    Without an ``explicit_name`` the object's ``__name__`` is converted to ``snake_case`` and a
    warning is emitted when the result starts with an underscore — likely a private class that
    should not be registered at all.

    An ``explicit_name`` wins verbatim unless the key is derived elsewhere too — see
    :func:`_key_is_derived_elsewhere` — in which case it may only restate the derived key.
    (:class:`TermRegistry` does not reach that branch: it refuses ``name=`` outright.)
    """
    if explicit_name is None:
        derived = to_snake_case(obj.__name__)
        if derived.startswith("_"):
            warnings.warn(
                f"Registering '{obj.__name__}' into '{registry_name}' produces the "
                f"leading-underscore key '{derived}'. Leading underscores usually mark internals "
                f"— rename it, or (on a registry that takes one) pass an explicit name=.",
                stacklevel=3,
            )
        return derived

    if not _key_is_derived_elsewhere(obj):
        # Freely-named: ``__name__`` is never consulted, and a registry holds arbitrary objects
        # (partials, instances, module-level constants) that do not have one.
        return explicit_name

    derived = to_snake_case(obj.__name__)
    if explicit_name != derived:
        raise ValueError(
            f"cannot register '{obj.__name__}' into '{registry_name}' under the name "
            f"'{explicit_name}': its key is always '{derived}', because {_WHY_THE_KEY_IS_THE_NAME}. "
            f"Registering it as '{explicit_name}' would build a config naming a key this registry "
            f"does not hold. Drop the name= (or rename '{obj.__name__}' to match) so the two agree."
        )
    return explicit_name


class Registry(Iterable[tuple[str, Any]]):
    """Registry providing a name -> object mapping to support third-party custom modules.

    Keys default to ``to_snake_case(obj.__name__)``; pass ``name=`` to choose a different one.
    That override is for objects looked up **only** by their key — assets, robots, tasks. A
    :class:`~eden.utils.common.ConfigurableMixin` is not one of them: its key is also what
    ``configure()`` stamps into ``options.name``, so ``name=`` on one is refused rather than
    honoured (see :func:`_key_is_derived_elsewhere`), and :class:`TermRegistry` — whose entries
    are all configurable — takes no ``name=`` at all.

    To create a registry (e.g. a robot registry):

    ```python

    ROBOT_REGISTRY = Registry('ROBOT')
    ```

    To register an object:

    ```python

    @ROBOT_REGISTRY.register()
    class Go1():
        ...
    ```

    Or:

    ```python

    ROBOT_REGISTRY.register(Go1)
    ```

    Parameters
    ----------
    name : str
        The name of this registry, quoted in every message it raises.
    populated_by : Sequence[str]
        Modules whose import-time ``@register`` decorators fill this registry, imported on the
        first read (:meth:`_populate`) rather than by whoever imported the registry. A registry
        whose entries are defined in the module that declares it needs none.
    """

    def __init__(self, name: str, populated_by: Sequence[str] = ()) -> None:
        self._name: str = name
        self._obj_map: dict[str, Any] = {}
        self._populated_by: tuple[str, ...] = tuple(populated_by)
        self._populated: bool = False
        #: Thread ids currently inside :meth:`_populate`. A set, not a flag, because the only
        #: thread that may skip the import is one already doing it — see :meth:`_populate`.
        self._populating: set[int] = set()
        #: The exception a failed :meth:`_populate` raised, replayed to every later reader
        #: rather than retried — see :meth:`_populate`.
        self._populate_error: BaseException | None = None
        #: That exception's traceback as first caught, kept separately because re-raising the
        #: exception rebinds its ``__traceback__`` to a longer chain each time — see
        #: :meth:`_populate`.
        self._populate_error_tb: TracebackType | None = None

    def _populate(self) -> None:
        """Import the modules whose ``@register`` decorators fill this registry, once.

        A registry is filled by an import side effect, so every entry point that *reads* one has
        to guarantee those imports have happened. Importing the term and options packages from
        ``eden/__init__.py`` would add ~2.4 s to ``import eden`` over ``import genesis``, paid by
        ``eden --help`` and by any script that only reads a config. Declaring the modules here
        moves the cost to the first lookup, so a process that never resolves a term never pays it.

        Re-entrant by design: a populating module that itself looks something up in this registry
        (or a cycle between two of them) sees the registry as it stands rather than recursing.
        That escape is keyed on the *calling thread*. A shared flag also catches a thread that is
        merely waiting: a second thread reading a cold registry while the first is mid-import
        would skip the import it is not doing, find an empty map, and raise ``No object named 'x'
        found ... Available options: none`` — an empty registry reported as a missing term.

        No lock of this class's own, deliberately. ``importlib.import_module`` already serializes
        on the import system's per-module lock, so a concurrent caller blocks *there* and returns
        once the module is in ``sys.modules`` and its registrations have run. A lock held across
        those imports would sit outside the import locks and invert against a thread importing a
        populating module directly, which is the shape a deadlock needs.

        ``_populated`` is set only after every import returned *finished*, so a broken term module
        never leaves a silently empty registry behind — and a partial read is never memoized as a
        complete one. The second half matters because ``import_module`` returns early rather than
        blocking when the *calling* thread is already importing that module (CPython swallows its
        own ``_DeadlockError``), so "the import returned" and "the import finished" are not the
        same thing. A module imported directly registers its first term through
        :meth:`_do_register`, which populates; claiming ``_populated`` off that partial module
        would hand a concurrent reader a registry holding that one term and marked complete — the
        empty-registry-reported-as-a-missing-term failure above wearing a different hat.

        A failure is **replayed, not retried**. Re-importing looks like the obvious way to give
        the second reader the same error, and it is wrong: a module that registered some terms
        *before* it raised left them in ``_obj_map``, while the import system dropped the module
        itself from ``sys.modules``. The retry therefore re-executes the module body, and its
        first ``@register()`` collides with the entry the aborted run already made — so the
        second reader would get ``cannot register X ... that key is taken by X``, one object named
        as both the registration that cannot be made and the incumbent holding the key, advising a
        rename of "one of the two" when there is one, and the real cause nowhere in sight.
        Rolling the registrations back instead is not available: a populate that
        imports several modules would also roll back the ones that *succeeded*, and those stay in
        ``sys.modules`` and so never re-register. Holding the exception gives every later reader
        the first one's type, message and frames, which is what this contract promised.

        The replay restores the traceback as first caught, rather than re-raising the exception
        as it stands. Raising an exception object *rebinds* its ``__traceback__`` to a chain one
        call site longer, and those frames land in front of the original ones — so re-raising it
        bare would grow the traceback by three frames a read and bury the term module's own
        failure under a lengthening stack of ``get`` -> ``_populate`` -> ``raise``.
        """
        if self._populated:
            return
        if self._populate_error is not None:
            raise self._populate_error.with_traceback(self._populate_error_tb)
        ident = threading.get_ident()
        if ident in self._populating:
            return
        self._populating.add(ident)
        try:
            # A list, not a generator argument to ``all()``: every module must be imported, and
            # ``all`` would stop at the first one that came back unfinished.
            finished = [not module_is_initializing(importlib.import_module(module)) for module in self._populated_by]
            self._populated = all(finished)
        except Exception as exc:
            # ``Exception``, not ``BaseException``: a KeyboardInterrupt is not a property of the
            # module and must not be memoized as this registry's permanent state.
            #
            # Named here because the bare exception says nothing about why this import was even
            # attempted: a mistyped ``populated_by`` entry surfaces as ``ModuleNotFoundError: No
            # module named 'eden.managers.terms.rewardz'`` several frames inside an unrelated
            # ``registry.get(...)``, naming neither the registry nor the field that asked for it.
            # A note rather than a wrapper exception, so the type and message the reader matches
            # on are untouched -- and notes are rendered with the traceback, which is where this
            # is read.
            exc.add_note(
                f"Raised while importing the modules that fill the '{self._name}' registry "
                f"(populated_by={list(self._populated_by)}), which happens on the first read of "
                f"it -- so the frames above may belong to an unrelated lookup. Every later read "
                f"re-raises this same error rather than retrying the import, because the retry "
                f"would re-run the registrations the failed import already made and report a "
                f"duplicate key instead of this. Fix the module, or the populated_by entry "
                f"naming it on the registry."
            )
            self._populate_error = exc
            # Captured before the re-raise below extends what ``exc.__traceback__`` points at.
            self._populate_error_tb = exc.__traceback__
            raise
        finally:
            self._populating.discard(ident)

    def _duplicate_hint(self, obj: Any) -> str:
        """How to resolve a key collision on ``obj``, appended to the duplicate-registration error.

        Which fix is available depends on where the key came from, and that is a property of the
        **object**, not of the registry: one whose key is derived from its own name
        (:func:`_key_is_derived_elsewhere`) cannot be given a different one, so advising ``name=``
        there would cost the caller a second error to reach the real answer. Overridden where a
        registry knows something the object cannot tell us.
        """
        if _key_is_derived_elsewhere(obj):
            return "Rename one of the two: its key is its own name, so there is no name= to separate them."
        return "Rename one of the two, or give one of them a different name=."

    def _do_register(self, name: str, obj: Any) -> None:
        # A write is a read of the key space, so it populates like one. Without this, an
        # out-of-tree term registered into a still-cold registry — the normal case, since
        # nothing imports the term modules eagerly — is checked against an empty map, so a
        # key it shares with a built-in is not caught at the ``@register()`` that caused it. It
        # surfaces from inside :meth:`_populate` during some later, unrelated lookup, and with
        # the blame inverted: the message names the *built-in* as the registration that cannot
        # be made and the caller's own term as the incumbent holding the key, so
        # :meth:`_duplicate_hint` reads as advice to rename an Eden built-in.
        #
        # Proportionate: the cold entry points (``eden --help``, ``eden list``) register
        # nothing and stay cold, while a process that
        # registers into this registry resolves out of it too, so it was going to pay this
        # import anyway.
        self._populate()
        # An exception, not an ``assert``: registrations run at import time, and ``python -O``
        # strips asserts — under which the second registration would silently win and every
        # lookup of that key would resolve to the wrong object.
        if name in self._obj_map:
            raise ValueError(
                f"cannot register {_describe(obj)} into the '{self._name}' registry as '{name}': "
                f"that key is taken by {_describe(self._obj_map[name])}. {self._duplicate_hint(obj)}"
            )
        self._obj_map[name] = obj

    def register(self, obj: Any = None, name: str | None = None) -> Any:
        """Register the given object, usable as either a decorator or a direct call.

        Both forms return the registered object, so the direct call can be chained and the
        decorator leaves the decorated name bound to it. See the class docstring for usage.
        """
        if obj is None:
            # used as a decorator
            def deco(func_or_class: Any) -> Any:
                self._do_register(_derive_name(func_or_class, name, self._name), func_or_class)
                return func_or_class

            return deco

        # used as a function call
        self._do_register(_derive_name(obj, name, self._name), obj)
        return obj

    def _lookup_hint(self, name: str) -> str:
        """What to append to the not-found message, beyond the roster — nothing, by default.

        Overridden by :class:`TermRegistry`, where the roster is ~200 names long and the command
        that searches it is the useful next step. Single-sentence and single-*line*: the caller
        raises ``KeyError``, whose ``__str__`` is ``repr(args[0])``, so a newline here would reach
        the reader escaped.

        Parameters
        ----------
        name : str
            The key that was not found, so an override can quote it into the command it suggests.
        """
        return ""

    def get(self, name: str) -> Any:
        self._populate()
        ret = self._obj_map.get(name)
        if ret is None:
            available = ", ".join(sorted(self._obj_map.keys())) or "none"
            # The closest key first: with ~200 registered terms the roster alone makes the reader
            # (or an agent) diff two long lists to find the one character that differs.
            #
            # Guarded on ``str``, because ``name`` is annotated as one but is not checked to be:
            # ``ManagerTermOptions.name`` is ``str | None`` and ``ManagerBase._build_term`` passes
            # it straight through, so an options entry that never got a name reaches here as
            # ``None``. ``difflib.get_close_matches`` iterates it and raises ``TypeError:
            # 'NoneType' object is not iterable`` from inside the message — losing the readable
            # KeyError that names the missing key and lists the roster, which for a config
            # missing a ``name=`` is the whole diagnosis. Message-building code must not raise
            # (docs/guides/error-messages.md); a non-string key simply gets no suggestion, and
            # the ``'{name}'`` below renders it fine whatever it is.
            close = difflib.get_close_matches(name, self._obj_map, n=1) if isinstance(name, str) else []
            suggestion = f" Did you mean '{close[0]}'?" if close else ""
            raise KeyError(
                f"No object named '{name}' found in '{self._name}' registry.{suggestion}"
                f"{self._lookup_hint(name)} Available options: {available}"
            )
        return ret

    def keys(self):
        self._populate()
        return self._obj_map.keys()

    def __contains__(self, name: str) -> bool:
        self._populate()
        return name in self._obj_map

    def __repr__(self) -> str:
        """The roster, populating first — but never raising, unlike the other read paths.

        Population imports arbitrary term modules, so it can fail for reasons that have nothing
        to do with whoever is looking. Every other read path is a real lookup and must propagate
        that (see :meth:`_populate`); a repr is a diagnostic, and it runs in a debugger, in ``%r``
        logging and in pytest's assertion rewriting — places where raising replaces the object
        under inspection with a failure about the inspection itself, in the middle of diagnosing
        something else entirely. So the failure is *reported inline* rather than swallowed: it
        still shows up here, and still raises from ``get`` / ``keys`` / ``in`` / iteration.
        """
        note = ""
        try:
            self._populate()
        except Exception as exc:  # noqa: BLE001 — a repr must not raise; it says so instead
            note = f" [incomplete: {type(exc).__name__}: {exc}]"
        if not self._obj_map:
            return f"Registry of {self._name}: (empty){note}"
        max_key_length = max(len(key) for key in self._obj_map.keys())
        return f"Registry of {self._name}:{note}\n" + "\n".join(
            [f"{key.ljust(max_key_length)} : {value}" for key, value in self._obj_map.items()]
        )

    def __iter__(self) -> Iterator[tuple[str, Any]]:
        self._populate()
        return iter(self._obj_map.items())

    # pyre-fixme[4]: Attribute must be annotated.
    __str__ = __repr__


class TermRegistry(Registry):
    """
    Registry for manager terms.

    Every entry is keyed by its own ``to_snake_case(__name__)`` and :meth:`register` takes no
    ``name=``, so two terms can only collide by being named the same thing.

    Parameters
    ----------
    name: str
        The name of the registry.
    term_class: Type[ManagerTermBase]
        The class of the manager term.
    term_fn_wrapper_class: Type[ManagerTermFuncWrapperBase] | None = None
        The class of the manager term function wrapper.
    populated_by: Sequence[str]
        Modules whose import-time registrations fill this registry — see
        :meth:`Registry._populate`. For a manager kind this is its ``eden.managers.terms.*``
        module, which ``import eden`` does not import.
    """

    @property
    def cli_kind(self) -> str:
        """What ``eden terms --kind`` calls this registry: ``REWARD_TERM`` -> ``reward``.

        Derived rather than declared so a new registry cannot ship with the wrong one, and pinned
        against :data:`eden.managers.catalog.KIND_NAMES` by ``tests/utils/test_registry.py``, beside
        ``tests/managers/test_catalog.py`` asserting the catalog covers every registry in the tree.
        """
        return self._name.lower().removesuffix("_term")

    def catalog_hint(self, term: str | None = None) -> str:
        """One line pointing at the ``eden terms`` command that answers what the error left open.

        The roster :meth:`Registry.get` prints answers "which terms are there?" but not "which one
        does what I need?" or "what does this one take?", and a reader who cannot answer those
        reaches for the 92 kB generated catalog or greps the installed source. Both
        wordings live here so the kind is derived once and the two cannot disagree. One line, no
        newline: the lookup caller raises ``KeyError``, whose ``__str__`` is ``repr(args[0])``.

        Parameters
        ----------
        term : str | None
            A *registered* term to point at, when there is one. ``None`` — the failed-lookup case
            — gets the search form with a ``<text>`` placeholder instead: the key that just failed
            is by definition not in the catalog, so grepping for it would print nothing and read
            as a dead end. A typo is covered by the ``Did you mean`` :meth:`Registry.get` adds.

        Returns
        -------
        str
            A sentence beginning with ``Run`` and no trailing space.
        """
        if term is None:
            return (
                f"Run `eden terms --kind {self.cli_kind} --grep <text>` to search these by name, parameter "
                f"and summary, or `eden terms --kind {self.cli_kind}` to list them all with their parameters."
            )
        return f"Run `eden terms --kind {self.cli_kind} --grep {term}` for its parameters, defaults and summary."

    def _lookup_hint(self, name: str) -> str:
        """The search form of :meth:`catalog_hint`, space-prefixed for :meth:`Registry.get`.

        Parameters
        ----------
        name : str
            The key that was not found. Unused — the hint is about the *kind*, not the miss.
        """
        return f" {self.catalog_hint()}"

    def _duplicate_hint(self, obj: Any) -> str:
        """Always the derived-key answer — see :meth:`Registry._duplicate_hint`.

        Every entry here is a term, but the object-level test cannot see that for a *function*
        term: it reaches :meth:`_do_register` as ``partial(wrapper, func=...)``, which is not a
        ``ConfigurableMixin`` subclass.
        """
        return "Rename one of the two: a term's key is its own name, so there is no name= to separate them."

    def __init__(
        self,
        name: str,
        term_class: Type[ManagerTermBase],
        term_fn_wrapper_class: Type[ManagerTermFuncWrapperBase] | None = None,
        populated_by: Sequence[str] = (),
    ) -> None:
        super().__init__(name, populated_by)
        self._term_class = term_class
        self._term_fn_wrapper_class = term_fn_wrapper_class
        if term_fn_wrapper_class is not None:
            # Let ``configure(func=...)`` verify the function is registered here (see
            # ConfigurableFuncWrapperMixin._build_config_dict). Stamped on the wrapper *class*, so a
            # wrapper shared by two registries would make one registry's terms validate against the
            # other's — refuse it (each registry needs its own func-wrapper subclass). Re-stamping the
            # same registry (e.g. a re-import) is a harmless no-op.
            existing = getattr(term_fn_wrapper_class, "_term_registry", None)
            if existing is not None and existing is not self:
                raise ValueError(
                    f"func-wrapper class {term_fn_wrapper_class.__name__} is already bound to registry "
                    f"'{existing._name}'; a wrapper class must belong to a single registry (give "
                    f"registry '{name}' its own {term_fn_wrapper_class.__name__} subclass)."
                )
            term_fn_wrapper_class._term_registry = self
        # Stamp this registry onto the kind's term-options class so ManagerTermOptions can
        # validate parameter names at construction time (see
        # ManagerTermOptions._validate_param_names). First stamp wins: a second registry built
        # over the same options class (e.g. a user registry whose terms subclass a built-in
        # term) leaves the original binding intact — names it can't resolve simply skip the
        # construction-time check and fall back to the build-time one. ``__dict__`` (not
        # getattr) so a subclassed options class can still be bound to its own registry.
        for term_or_wrapper in filter(None, (term_class, term_fn_wrapper_class)):
            options_class = getattr(term_or_wrapper, "_options_class_", None)
            if options_class is not None and options_class.__dict__.get("_term_registry_") is None:
                options_class._term_registry_ = self

    @property
    def term_class(self) -> Type[ManagerTermBase]:
        """The base class every class term in this registry subclasses."""
        return self._term_class

    @property
    def func_wrapper_class(self) -> Type[ManagerTermFuncWrapperBase] | None:
        """The wrapper a function term of this kind is configured through, or ``None``.

        ``None`` means the kind takes class terms only — :meth:`_register_term` refuses a plain
        function there. Exposed (with :attr:`term_class`) so :mod:`eden.managers.catalog` can read
        the per-kind fields every term carries off the base, rather than reaching into ``_``-private
        attributes to describe a registry it does not own.
        """
        return self._term_fn_wrapper_class

    def registered_class(self, key: str) -> type:
        """The class the term registered under ``key`` is built as, which is where its defaults live.

        A class term is stored as its class — the defaults-carrying subclass when ``register`` was
        given keyword defaults (:meth:`_with_defaults`). A function term is stored as
        ``partial(wrapper, func=...)``, and ``wrapper`` is the per-registration subclass carrying
        *its* keyword defaults, not the kind's shared ``*FuncWrapper``: that subclass is the only
        place ``@METRIC_TERM_REGISTRY.register(direction=MetricDirection.LIB)`` on a function is
        recorded, so a default read off the shared wrapper is the kind's, not the term's.

        Parameters
        ----------
        key : str
            A registered key. An unknown one raises the :meth:`Registry.get` ``KeyError``.

        Returns
        -------
        type
            The registered class term, or the wrapper class of a function term.
        """
        registered = self.get(key)
        return registered.func if isinstance(registered, partial) else registered

    @staticmethod
    def _with_defaults(cls: type, term_defaults: dict[str, Any]) -> type:
        """Return ``cls`` carrying ``term_defaults`` as class attributes.

        Synthesizes an isolated subclass rather than mutating ``cls`` in place: the class object
        is shared across every import site, so ``setattr``-ing defaults onto it would leak the
        defaults across tasks. The subclass inherits ``cls``'s ``__name__``/``__qualname__``/
        ``__module__`` so it is indistinguishable from the original for the ``name`` property,
        ``repr``, and any introspection — only the defaults differ. ``register`` also hands this
        subclass back to the caller, so ``.configure()`` on the decorated class reads the defaults.
        """
        if not term_defaults:
            return cls
        # Identity fields last so they always win — a stray ``__qualname__``/``__module__`` in
        # ``term_defaults`` can't clobber the preserved identity.
        namespace = {**term_defaults, "__qualname__": cls.__qualname__, "__module__": cls.__module__}
        return type(cls.__name__, (cls,), namespace)

    def _register_term(self, obj: Any, key: str, term_defaults: dict[str, Any]) -> Any:
        """Register ``obj`` under ``key`` and return the object the caller should bind to.

        For class terms (and func-wrapper subclasses) with defaults that is the synthesized
        defaults-carrying subclass, so ``ClassTerm.configure()`` on the returned class sees the
        registered defaults. For function terms it is the original function: they are configured
        through the kind's shared ``*Term`` wrapper alias, which does not carry this registration's
        defaults, so both configuration paths read them back through :meth:`registered_class` —
        ``configure(func=...)`` and ``*TermOptions(name=...)`` alike.
        """
        if inspect.isclass(obj) and issubclass(obj, self._term_class):
            registered = self._with_defaults(obj, term_defaults)
            self._do_register(key, registered)
            return registered
        elif isinstance(obj, FunctionType):
            if self._term_fn_wrapper_class is None:
                raise ValueError("function wrapping is not supported for this registry")
            wrapper_cls = self._with_defaults(self._term_fn_wrapper_class, term_defaults)
            self._do_register(key, partial(wrapper_cls, func=obj))
            return obj
        elif (
            self._term_fn_wrapper_class is not None
            and inspect.isclass(obj)
            and issubclass(obj, self._term_fn_wrapper_class)
        ):
            registered = self._with_defaults(obj, term_defaults)
            self._do_register(key, registered)
            return registered
        else:
            raise ValueError(f"Invalid object type: {type(obj)}")

    def register(self, obj: Any = None, *, name: str | None = None, **term_defaults: Any) -> Any:
        """Register a class term, a function term, or a func-wrapper subclass.

        Takes no ``name``: a term's key is always its own ``to_snake_case(__name__)`` — see
        :data:`_WHY_THE_KEY_IS_THE_NAME`, the cause this refusal shares with :func:`_derive_name`.
        ``name`` is declared only so it can be *refused* — ``**term_defaults`` would otherwise
        swallow it and quietly set a class attribute called ``name`` on the term instead.
        """
        if name is not None:
            raise ValueError(
                f"{self._name}_REGISTRY.register() takes no name=: a term's key is always its own "
                f"name, because {_WHY_THE_KEY_IS_THE_NAME}. Passing '{name}' cannot change that. "
                f"Drop it, and rename the term itself if you want a different key."
            )
        if obj is None:
            # used as a decorator
            def deco(func_or_class: Any) -> Any:
                key = _derive_name(func_or_class, None, self._name)
                return self._register_term(func_or_class, key, term_defaults)

            return deco

        # used as a function call
        return self._register_term(obj, _derive_name(obj, None, self._name), term_defaults)


class VariantRegistry(Registry):
    """Registry for asset families with multiple mesh-file variants.

    A single decorated class declares its source directory name and the list of
    available variant ids. The decorator stamps ``_base_name`` and
    ``_default_variant`` (the first id in ``variant_list``) onto the class,
    merges family metadata into ``cls.metadata``, and registers one
    ``partial(cls, variant=N)`` callable per variant under the key
    ``f"{base_name}_{N}"``. The decorated class itself is also returned so it
    stays importable for type-level use.

    Asset families (objaverse, ...) follow this contract by declaring
    a base options class that defines ``_base_name``/``_default_variant`` as
    ``ClassVar`` and an ``__init__`` (or ``model_post_init``) that materializes
    ``file`` from those + the requested variant.

    Calling ``register()`` with neither ``base_name`` nor ``variant_list`` falls
    through to :class:`Registry`'s plain-name registration, so procedural
    primitives (boxes, spheres) can share the same registry as variant assets.
    """

    def _duplicate_hint(self, obj: Any) -> str:
        """Name every argument that can produce a key here — see :meth:`Registry._duplicate_hint`.

        Which one applies depends on the path the caller took: a variant key comes from
        ``base_name``, and ``name=`` is *ignored* on that path (it only reaches
        :class:`Registry` when neither ``base_name`` nor ``variant_list`` was given). Naming just
        ``name=`` would be inert advice for the registry's main path.
        """
        return "Rename one of the two, or change the name= / base_name that produced the key."

    def register(
        self,
        obj: Any = None,
        *,
        name: str | None = None,
        base_name: str = "",
        variant_list: Sequence[int] | None = None,
    ) -> Any:
        if base_name == "" and variant_list is None:
            return super().register(obj, name=name)

        if not variant_list:
            raise ValueError(f"variant_list is required when base_name is set (base_name={base_name!r})")

        def _register(options):
            options._base_name = base_name
            options._default_variant = variant_list[0]
            merged_metadata = dict(getattr(options, "metadata", None) or {})
            merged_metadata.update(
                {
                    "base_name": base_name,
                    "variant_list": list(variant_list),
                    "num_variants": len(variant_list),
                    "variants_name": [f"{base_name}_{variant}" for variant in variant_list],
                }
            )
            options.metadata = merged_metadata
            for variant in variant_list:
                self._do_register(f"{base_name}_{variant}", partial(options, variant=variant))
            return options

        if obj is None:
            return _register
        return _register(obj)
