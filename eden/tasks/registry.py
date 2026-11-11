"""Task registry: lazy task lookup and CLI field derivation.

:data:`TASK_REGISTRY` resolves task config classes by name, importing the owning module
lazily on first access. Built-in tasks self-register at their definition site — there is
no central index in ``eden/tasks/__init__.py``; the registry AST-scans ``eden/tasks/**/*.py``
for ``@TASK_REGISTRY.register()`` decorators without importing the modules. Register your
own the same way; ``register(name=..., override=True)`` called at run time replaces a registration,
but two scanned classes registering one name clash, ``override=`` or not. An installed
distribution makes its own task package scanned too by naming it under the
:data:`TASK_ENTRY_POINT_GROUP` entry-point group. This module also derives argparse CLI fields
from a task config's keyword-only fields.
"""

from __future__ import annotations

import ast
import copy
import difflib
import importlib
import inspect
import textwrap
import types
import warnings
from importlib.metadata import EntryPoint, entry_points
from importlib.util import find_spec
from pathlib import Path
from typing import Annotated, Any, Literal, NamedTuple, Union, get_args, get_origin, get_type_hints

from eden.utils.misc import to_snake_case

#: The entry-point group an installed distribution names its task packages under. Each entry
#: point's value is a package, AST-scanned like ``eden.tasks`` (see :class:`TaskRegistry`)::
#:
#:     [project.entry-points."eden.tasks"]
#:     my_project = "my_project.tasks"
TASK_ENTRY_POINT_GROUP = "eden.tasks"

# How a "no such task" message tells the reader where an out-of-tree task has to be declared.
OUT_OF_TREE_TASK_HINT = (
    f'a task outside eden.tasks is found only once its package is installed and named under [project.entry-points."'
    f'{TASK_ENTRY_POINT_GROUP}"] in its pyproject.toml (docs/guides/task-config.md)'
)


def _default_task_name(class_name: str) -> str:
    """Canonical task name from a config class name.

    Applies ``to_snake_case`` and strips a trailing ``_config`` suffix so that
    ``ReacherConfig`` → ``reacher`` (not ``reacher_config``), matching the
    convention used across ``eden/tasks/**/config.py``.
    """
    snake = to_snake_case(class_name)
    return snake.removesuffix("_config") or snake


class TaskMod:
    """Tool that modifies an EdenConfig instance.

    TaskMod is a plain Python class (not a Pydantic model) because it is
    *logic* with a few parameters, not user-facing data. ``EdenConfig`` is the
    Pydantic data model; ``TaskMod`` operates on it.

    To define one: subclass, declare CLI-tunable parameters as
    **positional-or-keyword** ``__init__`` arguments with type annotations,
    and implement :meth:`apply`. Non-CLI constructor state goes in
    **keyword-only** parameters (after ``*``) — the CLI introspection skips
    those.

    Quick reference (this is the convention agents miss most often)::

        def __init__(
            self,
            difficulty: Literal["easy", "hard"] = "easy",  # ← CLI flag (--difficulty)
            *,
            entity_name: str = "robot",                    # ← non-CLI ctor state
        ):
            self.difficulty = difficulty
            self.entity_name = entity_name

    Calling ``super().__init__()`` is **optional** — ``TaskMod`` keeps no
    instance state of its own; ``_prefix`` is a class-attribute default that
    :meth:`with_prefix` shadows on assignment. So your subclass ``__init__``
    can just store its parameters and return.

    For per-field CLI ``--help`` text, wrap the annotation in
    ``typing.Annotated[T, "help text"]``. The string metadata is forwarded
    to ``argparse``; the inner type is what drives validation::

        difficulty: Annotated[Literal["easy", "hard"], "Task difficulty preset."] = "easy"

    Implement :meth:`apply` to return the modified config; in-place mutation
    is fine because :meth:`TaskRegistry.build` calls ``cls()`` fresh on every
    build (and Pydantic deep-copies the config's BaseModel defaults per
    instance), so mutations do not leak across builds.

    Modifiers attach to a task by passing **instances** to
    ``@TASK_REGISTRY.register(modifiers=(...))`` (a bare subclass is accepted
    and instantiated with no args). Modifiers are stored on the registry,
    not on the config class. A config registered without ``modifiers=``
    inherits them from its nearest registered base class; ``modifiers=()``
    opts out (see :meth:`TaskRegistry.register`).

    Per-instance scoping: chain :meth:`with_prefix` on a constructed instance
    so its CLI flags rename from ``<field>`` to ``<prefix>_<field>``. The same
    subclass can be registered multiple times with distinct prefixes.

    Example
    -------
    ::

        from typing import Annotated, Literal

        class RobotMod(TaskMod):
            def __init__(
                self,
                robot: Annotated[
                    Literal["allegro", "shadow", "xhand1"],
                    "Hand model to load into the scene.",
                ] = "xhand1",
                *,
                entity_name: str = "robot",  # keyword-only → not CLI-exposed
            ):
                self.robot = robot
                self.entity_name = entity_name

            def apply(self, config):
                setattr(config.scene_options, self.entity_name, ROBOT_REGISTRY.get(self.robot)())
                return config

        @TASK_REGISTRY.register(modifiers=(
            RobotMod(robot="shadow", entity_name="robot_left").with_prefix("left"),
            RobotMod(entity_name="robot_right").with_prefix("right"),
        ))
        class TwoArmTask(EdenRLConfig):
            ...

        # CLI: --left_robot shadow --right_robot xhand1
        TASK_REGISTRY.build("two_arm_task", left_robot="allegro", right_robot="shadow")
    """

    # Per-instance CLI/kwarg namespace; class-attribute default lets subclasses
    # skip `super().__init__()`. with_prefix() shadows this with an instance attr.
    _prefix: str = ""

    @property
    def prefix(self) -> str:
        """Per-instance CLI/kwarg namespace; empty string means unprefixed."""
        return self._prefix

    def with_prefix(self, prefix: str) -> "TaskMod":
        """Set this instance's CLI/kwarg prefix and return self for chaining.

        Chain at registration time::

            RobotMod(entity_name="robot_left").with_prefix("left")
        """
        self._prefix = prefix
        return self

    def flag_for(self, field_name: str) -> str:
        """Return the CLI/build-kwarg name for a field on this instance.

        With a non-empty :attr:`prefix`, returns ``f"{prefix}_{field_name}"``;
        otherwise returns ``field_name`` unchanged.
        """
        return f"{self._prefix}_{field_name}" if self._prefix else field_name

    def apply(self, config: Any) -> Any:
        raise NotImplementedError(f"{type(self).__name__} must implement apply().")


def _unwrap_annotated(annotation: Any) -> tuple[Any, str | None]:
    """Strip ``typing.Annotated[T, ...]`` and return ``(T, first_string_meta)``.

    Annotation pass-through: non-Annotated annotations come back unchanged
    with ``help=None``. The first string in the metadata tuple is treated as
    argparse help text; non-string metadata is ignored so other tools can
    layer their own markers without conflict.
    """
    if get_origin(annotation) is Annotated:
        inner, *metadata = get_args(annotation)
        help_text = next((m for m in metadata if isinstance(m, str)), None)
        return inner, help_text
    return annotation, None


def cli_fields(mod_cls: type[TaskMod]) -> list[tuple[str, Any, Any, str | None]]:
    """Return ``[(name, annotation, default, help), ...]`` for a TaskMod subclass's CLI fields.

    CLI fields are the **positional-or-keyword** parameters of ``mod_cls.__init__``
    (excluding ``self``). Keyword-only parameters (declared after ``*`` in the
    signature) are non-CLI constructor state and skipped here. Annotations are
    resolved with :func:`typing.get_type_hints` (``include_extras=True``) so
    postponed annotations (``from __future__ import annotations``) and
    ``Annotated[...]`` wrappers are returned as real types. ``help`` is the
    first string inside ``Annotated[T, ...]`` metadata, or ``None``.
    """
    if mod_cls is TaskMod:
        return []
    sig = inspect.signature(mod_cls.__init__)
    # Narrow the resolution failure to the cases we expect — typo'd
    # forward references (NameError) and malformed Annotated args
    # (TypeError). Other exceptions are real bugs and should propagate.
    # We still warn on the survivable cases so the user sees what failed
    # instead of getting a misleading "unsupported annotation" downstream.
    try:
        hints = get_type_hints(mod_cls.__init__, include_extras=True)
    except (NameError, TypeError) as exc:
        warnings.warn(
            f"cli_fields: could not resolve type hints for "
            f"{mod_cls.__module__}.{mod_cls.__name__}.__init__ "
            f"({type(exc).__name__}: {exc}). Falling back to raw annotations; "
            f"CLI flag generation may misclassify fields on this mod.",
            stacklevel=2,
        )
        hints = {}
    out: list[tuple[str, Any, Any, str | None]] = []
    for name, param in sig.parameters.items():
        if name == "self":
            continue
        if param.kind != inspect.Parameter.POSITIONAL_OR_KEYWORD:
            continue
        raw = hints.get(name, param.annotation)
        annotation, help_text = _unwrap_annotated(raw)
        default = param.default if param.default is not inspect.Parameter.empty else None
        out.append((name, annotation, default, help_text))
    return out


def _keyword_only_fields(mod_cls: type[TaskMod]) -> set[str]:
    """Return the names of keyword-only ``__init__`` parameters (the non-CLI bucket).

    Used by :meth:`TaskRegistry.build` to convert a confusing
    ``Unknown modifier kwargs`` error into a targeted hint when the unknown
    kwarg matches a parameter that the author placed after ``*`` (the
    convention for non-CLI constructor state).
    """
    if mod_cls is TaskMod:
        return set()
    sig = inspect.signature(mod_cls.__init__)
    return {
        name
        for name, param in sig.parameters.items()
        if name != "self" and param.kind is inspect.Parameter.KEYWORD_ONLY
    }


def unwrap_optional(annotation: Any) -> tuple[Any, bool]:
    """Strip a ``T | None`` / ``Optional[T]`` wrapper: returns ``(T, allows_none)``.

    Both readers of a CLI field's annotation need this narrowing and have to agree on it:
    :func:`eden.tasks.parser._add_mod_field_to_parser` needs a concrete ``type=`` callable for
    argparse, and :func:`_check_cli_value` needs it or it silently stops checking — a union is
    in neither its Literal branch nor its primitive tuple, so a ``str | None`` field would turn
    ``build("contrack", data=5)`` from a named ValueError into a ``TypeError`` raised later, from
    inside the mod.

    A union of two or more non-None members comes back unchanged — neither caller can narrow
    that to one type, and both already treat an unsupported annotation as none of their business.
    """
    if get_origin(annotation) not in (Union, types.UnionType):
        return annotation, False
    args = get_args(annotation)
    non_none = [a for a in args if a is not type(None)]
    return (non_none[0] if len(non_none) == 1 else annotation), len(non_none) != len(args)


def _check_cli_value(field_name: str, annotation: Any, value: Any) -> None:
    """Validate ``value`` against a CLI field's annotation. Raises ValueError on mismatch.

    Replaces what Pydantic's construction-time validation gave us: Literal
    membership and primitive type coercion. Called from
    :meth:`TaskRegistry.build` before applying overrides — argparse already
    enforces these for the CLI path, so this only fires for programmatic
    ``build("task", field=value)`` calls.
    """
    annotation, allows_none = unwrap_optional(annotation)
    if value is None and allows_none:
        return
    choices = literal_choices(annotation)
    if choices is not None:
        if value not in choices:
            raise ValueError(f"Invalid value for '{field_name}': {value!r} is not one of {list(choices)}.")
        return
    if annotation in (int, float, str, bool) and not isinstance(value, annotation):
        raise ValueError(
            f"Invalid value for '{field_name}': expected {annotation.__name__}, got {type(value).__name__}."
        )


# A modifier entry at registration time is one of:
#   * a TaskMod instance (preferred; supports prefix + non-CLI ctor state),
#   * a TaskMod subclass (auto-instantiated with no args — convenience).
ModifierEntry = TaskMod | type[TaskMod]


class _Inherited:
    """Sentinel stored for a registration made without ``modifiers=``.

    Standing in for the resolved tuple is what keeps inheritance *live*: the MRO walk runs on
    every read, so what a task inherits follows the class graph rather than the order the
    registrations happened to run in. Freezing the tuple at registration instead would leave a
    subclass on the mods of a parent that had since been re-registered, silently and differently
    depending on which of the two ran first.
    """

    __slots__ = ()

    def __repr__(self) -> str:
        return "<inherited>"


_INHERITED = _Inherited()


class TaskRegistry:
    """Task registry with two lookup paths, tried in order.

    1. Eager registry — populated by ``@TASK_REGISTRY.register()`` decorators
       or explicit ``register(cls, ...)`` calls.
    2. Auto-discovery — an AST scan of the configured search packages (default
       ``["eden.tasks"]``) finds ``@TASK_REGISTRY.register()`` decorators in
       every ``*.py`` file without importing them. On first :meth:`get` miss
       we import only the one file that matches the requested name; that
       triggers the decorator and populates the eager registry.

    Tasks self-register at the definition site — no central index in
    ``eden/tasks/__init__.py`` to keep in sync. Out-of-tree tasks join the
    scan through the :data:`TASK_ENTRY_POINT_GROUP` entry point of the
    distribution that ships them, so ``eden list`` / ``eden train --task``
    see them without importing anything; a script can also call
    :meth:`add_search_path` before its first lookup.

    Auto-discovery rules (files that are NOT scanned):

    * ``__init__.py`` is skipped — put decorated classes in a sibling module
      such as ``config.py``.
    * files whose name starts with ``_`` are skipped (e.g. ``_meta.py``) —
      private modules are treated as implementation details.

    Auto-discovery rules (decorator recognition):

    * The scan recognizes ``@<name>.register`` where ``<name>`` is bound, in
      the same file, to ``TASK_REGISTRY`` via an absolute import from
      ``eden.tasks`` or ``eden.tasks.registry`` (``from ... import
      TASK_REGISTRY`` or ``... as X``). Both forms work.
    * Using the literal name ``TASK_REGISTRY`` without importing it from a
      recognized location emits a warning — the scan will not silently miss
      the registration.

    Usage::

        # Preferred: decorator at definition site; modifiers passed to the
        # decorator so they live on the registry (not on the config class,
        # which is a Pydantic model and should only hold serializable schema).
        @TASK_REGISTRY.register(modifiers=(ReacherDifficultyMod,))
        class ReacherConfig(EdenRLConfig):
            ...

        cfg = TASK_REGISTRY.build("reacher", difficulty="hard")  # a configured instance, mods applied
        cls = TASK_REGISTRY.get("reacher")                       # the bare class; cls() runs no mod
    """

    def __init__(self) -> None:
        self._registry: dict[str, Any] = {}
        # Modifiers attached at registration time (override class attr). ``_INHERITED`` stands
        # for "take the nearest registered base's", resolved on read by _resolve_modifiers.
        self._modifiers: dict[str, tuple[TaskMod, ...] | _Inherited] = {}
        # Memo for that walk, dropped whenever a registration changes what it would answer.
        self._resolved_modifiers: dict[str, tuple[TaskMod, ...]] = {}
        # Packages add_search_path added to the scan (eden.tasks always), each mapped to the source
        # task_source() reports for its tasks.
        self._search_packages: dict[str, str] = {"eden.tasks": "eden"}
        # The registrations _do_register set aside because their name clashes, by "module:Class",
        # as (name, class, modifiers entry): the class answers to no name, but a subclass still
        # inherits its modifiers through it, and a rescan that ends the clash registers it.
        self._clashing_registrations: dict[str, tuple[str, Any, tuple[TaskMod, ...] | _Inherited]] = {}

        # What the scan found. _ensure_discovered recomputes all of it, entry points included, so a
        # rescan sees a plugin fixed or installed since; nothing else writes it.
        # name -> "module.path:ClassName". None means "not yet scanned".
        self._discovered: dict[str, str] | None = None
        # name -> modifier display-names, read from the decorators.
        self._discovered_modifiers: dict[str, list[str]] = {}
        # Every package scanned, eden.tasks first, mapped to its source label (task_source()).
        self._package_sources: dict[str, str] = {}
        # What the scan could not answer, kept so that only a lookup it could change raises it (one
        # broken plugin must not stop every built-in task): entry points that name no scannable
        # package, as (the package it names, message, cause), and names registered in several
        # scanned files, as name -> [(module:Class, file path)].
        self._broken_entry_points: list[tuple[str, str, BaseException | None]] = []
        self._collisions: dict[str, list[tuple[str, str]]] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        obj: Any = None,
        *,
        name: str | None = None,
        override: bool = False,
        modifiers: tuple[ModifierEntry, ...] | None = None,
    ) -> Any:
        """Register a config class eagerly. Can be used as a decorator or function call.

        Parameters
        ----------
        obj: Any
            The config class to register. When ``None``, returns a decorator.
        name: str | None
            Lookup name. Defaults to ``_default_task_name(cls.__name__)``
            (``to_snake_case`` with any trailing ``_config`` stripped).
        override: bool
            If ``True``, silently replace an existing registration with the
            same name.
        modifiers: tuple[ModifierEntry, ...] | None
            Sequence of TaskMod **instances** (preferred) or bare TaskMod
            subclasses (auto-instantiated with no args). Applied in order by
            :meth:`build`. ``None`` (the default) inherits the modifiers of the
            nearest registered class in ``obj``'s MRO, so a subclass of a task
            keeps the mods that bind it — its robot, its data — instead of
            building against the parent's unset sentinels. That inheritance is
            **live**: it re-resolves on every read, so re-registering the class
            the mods come from — or registering it after this one — updates this
            task too, whichever order the two ran in. The registration
            ``override=True`` replaces counts, so shadowing a built-in with a
            subclass keeps the built-in's mods; that one is a snapshot, since the
            registration it came from no longer exists. ``()`` opts out; a class
            with no registered ancestor gets none either way.

        Raises
        ------
        ValueError
            If ``modifiers`` is ``None`` and the nearest registered ancestor is
            registered under several names with different modifiers, none of them
            the ``name`` being replaced — pass ``modifiers=`` explicitly to pick.
            A later registration that makes an already-registered task ambiguous
            the same way raises from :meth:`get_modifiers` / :meth:`build` instead
            — :meth:`modifier_names`, a display path, falls back to the names the
            decorators declare.
        """

        def deco(cls: Any) -> Any:
            if not isinstance(cls, type):
                # Every lookup reads the class's module and name, so anything else would register
                # here and then fail far from this line.
                raise TypeError(
                    f"TASK_REGISTRY.register() takes a config class; got {cls!r} ({type(cls).__name__}). "
                    "Register the class itself, and apply what a factory would set through modifiers=(...)."
                )
            self._do_register(name or _default_task_name(cls.__name__), cls, override=override, modifiers=modifiers)
            return cls

        return deco if obj is None else deco(obj)

    def add_search_path(self, package: str) -> None:
        """Register an additional package root to AST-scan for decorated tasks.

        Only the process that calls it sees the package, so it suits a script that does its own
        lookups::

            from eden.tasks import TASK_REGISTRY
            TASK_REGISTRY.add_search_path("my_project.tasks")

        For the ``eden`` CLI to see an installed project's tasks, name the package under the
        :data:`TASK_ENTRY_POINT_GROUP` entry-point group in the project's ``pyproject.toml``
        instead. A package that cannot be found is skipped, unlike a broken entry point.
        """
        if package in self._search_packages:
            return
        self._search_packages[package] = package
        # Invalidate the discovery cache so the new package is scanned on the next miss.
        self._discovered = None

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get(self, name: str) -> Any:
        """Get a config CLASS by name.

        Resolution order: eager registry → auto-discovery. The returned object
        is the bare config class, so ``get(name)()`` runs **none** of the task's
        registered TaskMods. For a task whose robot binding lives in a mod,
        that yields a config that constructs and serializes without
        complaint and then fails in ``env.build()`` on an empty link name — an
        error that says nothing about modifiers.
        Use :meth:`build` to obtain a configured instance; reach for ``get``
        only when the class itself is what you want (``issubclass`` checks,
        subclassing, matching a loaded config's type).

        Raises
        ------
        ValueError
            If ``name`` is registered in several scanned files.
        RuntimeError
            If ``name`` is not found and an installed :data:`TASK_ENTRY_POINT_GROUP` entry point
            names no scannable package, which could hold it.
        KeyError
            If ``name`` is not found anywhere.
        """
        qualname = self._lookup(name)
        if qualname is None:
            raise KeyError(self.missing_task_message(name))
        if name not in self._registry:
            # Importing the module triggers @TASK_REGISTRY.register(), which
            # populates self._registry[name].
            importlib.import_module(qualname.rsplit(":", 1)[0])
        if name in self._registry:
            return self._registry[name]
        # Import succeeded but the decorator didn't fire (e.g. the class
        # is gated behind `if __name__ == "__main__":`). Report clearly.
        raise RuntimeError(
            f"Auto-discovered task '{name}' at '{qualname}' but importing its module "
            "did not register it. Check that @TASK_REGISTRY.register() runs at import time."
        )

    def get_modifiers(self, name: str) -> tuple[TaskMod, ...]:
        """Resolve and return the TaskMod **instances** associated with ``name``.

        Modifiers live on the registry itself (not on the config class),
        attached via ``register(modifiers=...)``. For decorator-registered
        tasks the decorator's ``modifiers=`` tuple is populated when the
        module is imported, so we call :meth:`get` first to ensure that
        import has happened. A registration that inherits its modifiers
        resolves them here, against the registry as it stands now.
        """
        if name not in self:
            raise KeyError(f"No task named '{name}' found in task registry.")
        # Ensure the module is loaded so the decorator has had a chance to
        # attach modifiers. No-op if already loaded.
        self.get(name)
        return self._resolve_modifiers(name)

    def modifier_names(self, name: str) -> list[str]:
        """Modifier display-names for ``name`` **without importing** its module.

        Unlike :meth:`get_modifiers` (which imports the task to return live
        instances), this is purely informational — used by ``eden list`` so a
        read-only listing never triggers a task-module import and the asset
        downloads that can ride along. Prefers modifier instances already
        attached (eager registrations / already-loaded tasks) for exactness,
        an inherited registration included — resolving one reads the other live
        registrations, which needs no import either. Otherwise falls back to the
        identifier names captured from the ``@register(modifiers=...)`` decorator
        during AST auto-discovery — for a decorator without ``modifiers=``, the
        ones its base class's decorator names, since that is what registering it
        inherits. A listing renders rather than raises, so an inherited
        registration left ambiguous by a later one falls back here too; the
        error still belongs to :meth:`get_modifiers` and :meth:`build`, which
        are what a caller acts on.
        """
        try:
            attached = self._resolve_modifiers(name)
        except ValueError:
            attached = ()
        if attached:
            return [type(m).__name__ for m in attached]
        self._ensure_discovered()  # populates self._discovered_modifiers
        return list(self._discovered_modifiers.get(name, []))

    def build(self, name: str, **modifier_kwargs: Any) -> Any:
        """Instantiate the config and apply registered TaskMods.

        Modifier kwargs are flat and use each mod's **prefixed** field names
        (see :meth:`TaskMod.flag_for`). For an unprefixed mod that exposes
        ``difficulty``, pass ``difficulty=...``; for an instance registered
        with ``prefix="left"`` that exposes ``robot``, pass ``left_robot=...``.

        Collisions — two mods that would claim the same flag — raise
        ``ValueError`` at build time with both owners listed. Unknown kwargs
        raise ``ValueError`` with the list of accepted flags.
        """
        cls = self.get(name)
        config = cls()
        mods = self._resolve_modifiers(name)

        # Map each prefixed flag to (mod index, field_name, annotation).
        flag_map: dict[str, tuple[int, str, Any]] = {}
        for i, mod in enumerate(mods):
            for field_name, annotation, _default, _help in cli_fields(type(mod)):
                flag = mod.flag_for(field_name)
                if flag in flag_map:
                    j, other_field, _ = flag_map[flag]
                    raise ValueError(
                        f"Modifier flag collision on task '{name}': '{flag}' is claimed by "
                        f"both mod[{j}] ({type(mods[j]).__name__}.{other_field}) and "
                        f"mod[{i}] ({type(mod).__name__}.{field_name}). "
                        "Give the TaskMod instances distinct prefix values via .with_prefix(...)."
                    )
                flag_map[flag] = (i, field_name, annotation)

        unknown = set(modifier_kwargs) - set(flag_map)
        if unknown:
            # Surface a targeted hint when the unknown kwarg matches a
            # keyword-only init param on a registered mod — that's the most
            # common adoption mistake (declaring a CLI knob after `*` so it
            # never reaches the CLI surface). Match on the prefixed flag form
            # so the hint also fires for prefixed mods.
            kwonly_index: dict[str, tuple[TaskMod, str]] = {}
            for mod in mods:
                for field in _keyword_only_fields(type(mod)):
                    kwonly_index[mod.flag_for(field)] = (mod, field)
            kwonly_hits: list[str] = []
            for u in sorted(unknown):
                if u in kwonly_index:
                    mod, field = kwonly_index[u]
                    kwonly_hits.append(
                        f"'{u}' is keyword-only on {type(mod).__name__} (as '{field}') — "
                        f"keyword-only params are non-CLI ctor state. Set it at registration "
                        f"with modifiers=({type(mod).__name__}({field}=...),); if you own "
                        f"{type(mod).__name__} and want a CLI flag, move '{field}' before "
                        f"`*` in __init__."
                    )
            hint = ("\n" + textwrap.indent("Hint:\n" + "\n".join(kwonly_hits), "  ")) if kwonly_hits else ""
            raise ValueError(
                f"Unknown modifier kwargs for task '{name}': {sorted(unknown)}. "
                f"Known: {sorted(flag_map) or 'none'}{hint}"
            )

        # Group kwargs per mod, stripping the prefix and validating each value
        # against its declared annotation (Literal membership + primitive type).
        updates_per_mod: list[dict[str, Any]] = [{} for _ in mods]
        for flag, value in modifier_kwargs.items():
            i, field_name, annotation = flag_map[flag]
            _check_cli_value(field_name, annotation, value)
            updates_per_mod[i][field_name] = value

        for mod, updates in zip(mods, updates_per_mod):
            if updates:
                # Deep copy so per-build overrides don't mutate the registered
                # instance; deep so mutable instance state (lists/dicts a
                # subclass might keep) is not aliased across builds.
                applied = copy.deepcopy(mod)
                for field_name, value in updates.items():
                    setattr(applied, field_name, value)
            else:
                applied = mod
            config = applied.apply(config)
        return config

    def list_tasks(self) -> list[str]:
        """List all registered task names (eager + discovered).

        Raises
        ------
        ValueError
            If a task name is registered in several scanned files.
        RuntimeError
            If an installed :data:`TASK_ENTRY_POINT_GROUP` entry point names no scannable package
            — the list would be missing that package's tasks.
        """
        discovered = self._ensure_discovered()
        if self._collisions:
            raise ValueError("\n".join(_collision_message(n, where) for n, where in sorted(self._collisions.items())))
        self._raise_if_scan_incomplete("Cannot list every task.")
        return sorted(set(self._registry) | set(discovered))

    def config_qualname_for_task(self, name: str) -> str | None:
        """The ``module.Class`` of the config ``name`` builds, or ``None`` if no such task.

        The forward direction of :meth:`task_for_config_qualname`, and import-free for the same
        reason — eager entries are already in memory, discovered-only entries carry the
        ``module:Class`` string the AST scan recorded — so a caller can ask "is this saved config
        the one ``--task`` builds?" without importing the task module. The replay path
        (``eden inference`` / ``eden deploy``) never needs that module: it loads the checkpoint's
        own config, and only wants the identity.

        Prefer this over reversing :meth:`task_for_config_qualname` whenever the question is about
        a *named* task. Nothing stops one config class being registered under two names (aliases
        are ordinary here — :meth:`register` keys on ``name`` alone), and the reverse lookup
        returns only the first of them, so an alias would read as a mismatch.

        Parameters
        ----------
        name: str
            Registered task name.

        Returns
        -------
        str | None
            The dotted qualname, or ``None`` when no registered or discoverable task has that name.
        """
        qualname = self._lookup(name)
        return None if qualname is None else qualname.replace(":", ".", 1)

    def task_for_config_qualname(self, qualname: str) -> str | None:
        """The task whose config class is ``module.Class``, or ``None`` if no task owns it.

        The reverse of :meth:`get`, and deliberately **import-free**: eager entries are already
        in memory, and discovered-only entries are matched against the ``module:Class`` string
        the AST scan recorded. Calling :meth:`get` in a loop instead would import every task
        module in the tree — and could raise out of an unrelated one — which is the wrong trade
        for a lookup whose caller is building an error message.

        Parameters
        ----------
        qualname: str
            Dotted ``module.Class`` name, as :func:`eden.utils.configs.saved_config_qualname`
            reads it out of a saved config.

        Returns
        -------
        str | None
            The task name, or ``None`` when no registered or discoverable task declares it.
        """
        discovered = self._ensure_discovered()
        module, _, cls_name = qualname.rpartition(".")
        key = f"{module}:{cls_name}"  # the "module:Class" form the scan and _class_key use
        for name, cls in self._registry.items():
            if _class_key(cls) == key:
                self._raise_if_clashing(name, cls)
                return name
        for name, found in discovered.items():
            if found == key:
                return name
        for name, where in self._collisions.items():
            if any(found == key for found, _ in where):
                raise ValueError(_collision_message(name, where))
        self._raise_if_scan_incomplete(
            f"No task declares '{qualname}', and its module lies under a task package the scan could not read.",
            module=module,
        )
        return None

    def task_source(self, name: str) -> str | None:
        """Where the config ``name`` builds comes from, as ``eden list`` shows it.

        ``"eden"`` for a built-in, the distribution's name for a task found through its
        :data:`TASK_ENTRY_POINT_GROUP` entry point, the package for one under a package passed to
        :meth:`add_search_path`, and the defining module for a class registered from anywhere else
        — so a built-in name shadowed by a ``register(..., override=True)`` reads as the module
        that shadowed it. Import-free, like :meth:`config_qualname_for_task`, which it reads.

        Parameters
        ----------
        name: str
            Registered task name.

        Returns
        -------
        str | None
            The source label, or ``None`` when no registered or discoverable task has that name.
        """
        qualname = self.config_qualname_for_task(name)
        if qualname is None:
            return None
        module = qualname.rpartition(".")[0]
        # The innermost search package holding the module, so a package scanned inside another's tree wins.
        owners = [package for package in self._package_sources if _is_under(module, package)]
        return self._package_sources[max(owners, key=len)] if owners else module

    def __contains__(self, name: str) -> bool:
        """Whether ``name`` resolves; raises, like :meth:`get`, when the scan cannot tell."""
        return self._lookup(name) is not None

    def missing_task_message(self, name: str) -> str:
        """The message for a miss on ``name``: a close match, every task there is, and where to declare one.

        The text of :meth:`get`'s ``KeyError`` (so one line), for a caller that reports a miss its own way,
        as ``eden``'s ``--task`` check does. Unlike :meth:`list_tasks`, it lists what the scan resolved
        even while another name clashes or a plugin is broken, since those are reported where they
        are looked up, and the reader here needs the name they mistyped. It raises only what the scan
        itself does (a search-path package whose parents fail to import), and nothing once a lookup
        has scanned.
        """
        available = sorted(set(self._registry) | set(self._ensure_discovered()))
        close = difflib.get_close_matches(name, available, n=1)
        did_you_mean = f" Did you mean '{close[0]}'?" if close else ""
        return (
            f"No task named '{name}' found in task registry.{did_you_mean} "
            f"Available: {', '.join(available) or 'none'}. Note: {OUT_OF_TREE_TASK_HINT}"
        )

    def __repr__(self) -> str:
        # Not list_tasks(): a repr must render, and that raises while a plugin is broken or a name clashes.
        names = sorted(set(self._registry) | set(self._ensure_discovered()))
        if not names:
            return "TaskRegistry: (empty)"
        return "TaskRegistry:\n" + "\n".join(f"  {n}" for n in names)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _do_register(
        self,
        name: str,
        obj: Any,
        *,
        override: bool = False,
        modifiers: tuple[ModifierEntry, ...] | None = None,
    ) -> None:
        # The scan found this name on several classes, this one among them. Registering whichever
        # module happens to be imported first would make the answer depend on import order, and
        # raising here would fail every other task in the module; so neither class registers, and
        # every lookup of the name raises the collision (_raise_if_clashing). The registration is
        # set aside instead, so a subclass still inherits through it.
        clashing = self._is_colliding_class(name, obj)
        if not clashing and not override and name in self._registry:
            raise ValueError(f"A task named '{name}' is already registered in the task registry")
        entry: tuple[TaskMod, ...] | _Inherited
        if modifiers is not None:
            entry = _normalize_modifier_entries(modifiers)
        else:
            # Walk before writing anything, for three reasons: the ambiguity check is then a
            # registration-time error rather than a first-build one; the walk must still see the
            # registration override=True is replacing, so a subclass that shadows a task keeps the
            # task's mods; and a raise here leaves the registry untouched, replacements included.
            inherited = self._inherited_modifiers(name, obj)
            # Normally store the marker rather than that tuple: the walk re-runs on read, so
            # re-registering the class this one inherits from updates it too, in either
            # registration order.
            entry = _INHERITED
            if not clashing and name in self._registry and self._snapshot_on_replace(name, obj, inherited):
                entry = inherited
        if clashing:
            self._clashing_registrations[_class_key(obj)] = (name, obj, entry)
            self._resolved_modifiers.clear()
        else:
            self._write_registration(name, obj, entry)

    def _snapshot_on_replace(self, name: str, obj: Any, inherited: tuple[TaskMod, ...]) -> bool:
        """Whether the mods ``obj`` inherits as ``name`` have to be frozen rather than left live.

        They do when they came from the registration this one is about to replace — directly, or
        through registered classes in between whose names all inherit (the shadowing class's own
        other names among them) — because nothing answers for them once that registration is gone.
        Asked by re-running the walk the way a *read* would run it, with ``name``'s own
        registration hidden: that is exactly the registry the reads after the write will see, so it
        can be asked before the write, and nothing is half-applied if it raises.
        """
        try:
            live = self._inherited_modifiers(name, obj, resolving=frozenset({name}))
        except ValueError:
            # Ambiguous only without the registration being replaced: one more answer the live walk
            # cannot give for this entry. Nothing is masked — the entry becomes a snapshot and stops
            # taking part in that walk, and every other inheriting task still raises on its own read.
            return True
        return _modifier_signature(live) != _modifier_signature(inherited)

    def _write_registration(self, name: str, obj: Any, entry: tuple[TaskMod, ...] | _Inherited) -> None:
        """Record one registration and drop the memo, which any registration can invalidate."""
        self._registry[name] = obj
        self._modifiers[name] = entry
        self._resolved_modifiers.clear()

    def _resolve_modifiers(self, name: str) -> tuple[TaskMod, ...]:
        """The modifiers :meth:`build` applies for ``name``, resolving an inherited registration.

        A registration made without ``modifiers=`` holds :data:`_INHERITED` instead of a resolved
        tuple, so the answer is a function of the class graph and the registry as it stands — not
        of the order the registrations ran in. Memoized per name, and every registration
        drops the memo, since any of them can change what a walk would answer.

        The memo is reachable from here and from nowhere else: the three readers call this, and a
        walk calls :meth:`_walk_modifiers`. What it holds is what resolving with nothing set aside
        answered, which is not the answer a walk with names set aside is owed — handing one to the
        other returns the wrong mods, and keeping the two apart by construction is why the walk has
        no route here to take.
        """
        cached = self._resolved_modifiers.get(name)
        if cached is None:
            cached = self._resolved_modifiers[name] = self._walk_modifiers(name, frozenset())
        return cached

    def _walk_modifiers(self, name: str, resolving: frozenset[str], *, subject_cls: Any = None) -> tuple[TaskMod, ...]:
        """The modifiers ``name``'s own registration holds, walking the MRO if it inherits them.

        The un-memoized reader, and so the one a walk may call: ``resolving`` carries the
        registrations that walk has set aside, which is exactly what the memo knows nothing about.
        ``subject_cls`` is passed on to :meth:`_inherited_modifiers`.
        """
        entry = self._modifiers.get(name, ())
        if isinstance(entry, _Inherited):
            return self._inherited_modifiers(
                name, self._registry[name], resolving=resolving | {name}, subject_cls=subject_cls
            )
        return entry

    def _inherited_modifiers(
        self,
        name: str,
        obj: Any,
        *,
        resolving: frozenset[str] = frozenset(),
        subject_cls: Any = None,
        skip_self: bool = False,
    ) -> tuple[TaskMod, ...]:
        """The modifiers ``obj`` inherits when registered as ``name`` without ``modifiers=``.

        Walks ``obj``'s MRO, itself included (a second name for one class gets that class's mods),
        and stops at the first class registered under a name outside ``resolving`` — the names set
        aside, which on every path but registration's include ``name`` itself, so a registration
        never answers for itself. The instances are shared, not copied: :meth:`build` treats a
        registered mod as read-only and deep-copies one before an override touches it.

        Called from registration, where nothing is set aside, the walk does see ``name``'s own
        entry: the registration ``override=True`` is replacing. So a subclass that shadows a task
        keeps the task's mods, even when the shadowed class answers to other names as well, since
        ``override=True`` names the one registration it replaces and there is nothing to
        disambiguate. That holds whether or not the shadowing class was registered under some other
        name first, and through any registered classes between it and the shadowed class whose names
        all inherit: ``name`` stays the task asked for all the way up the walk, also where the walk
        goes on from another class. A class in between with a name that pins its own mods is
        the nearest registered base, so those mods win.

        ``subject_cls`` is the class the ambiguity error names, when that is not ``obj``: the walk
        goes on from a base for a group of names that all inherit and for a class set aside for a
        name clash, and the entry a registration replaces is walked with its old class, but a raise
        on any of these paths is still a failure of the class the caller asked about.

        ``skip_self`` starts the walk at ``obj``'s first base: how a class set aside for a name clash
        (``_clashing_registrations``) that itself inherits resolves what it inherits.
        """
        # Fixed here, before any recursion below replaces obj with a base or an old class.
        subject_cls = obj if subject_cls is None else subject_cls
        mro = getattr(obj, "__mro__", ())
        for base in mro[1:] if skip_self else mro:
            owners = [other for other, cls in self._registry.items() if cls is base and other not in resolving]
            if not owners:
                set_aside = self._clashing_registrations.get(_class_key(base))
                if set_aside is None:
                    continue
                clashing = set_aside[2]
                # A class set aside for a name clash answers to no name, but its modifiers are as
                # well-defined as a registered class's: take them, or, if it inherits, what it inherits.
                if not isinstance(clashing, _Inherited):
                    return clashing
                return self._inherited_modifiers(
                    name, base, resolving=resolving, subject_cls=subject_cls, skip_self=True
                )
            if name in owners:
                # Registration time, the one path that does not set `name` aside: the entry
                # override=True is about to replace. Resolved through _walk_modifiers, never
                # _resolve_modifiers — no walk may answer from the memo, which is why this is the
                # only route it has.
                return self._walk_modifiers(name, resolving, subject_cls=subject_cls)
            pinned: dict[str, tuple[TaskMod, ...]] = {}
            for other in owners:
                mods = self._modifiers[other]
                if not isinstance(mods, _Inherited):
                    pinned[other] = mods
            if not pinned:
                # Every name this class answers to inherits as well, so they all inherit the same
                # thing — whatever the class itself does. Resolve that once, with the whole group
                # set aside, rather than once per name through the others: k names resolving
                # through each other cost k! walks, which is 10 s at k=11 and a hang soon after.
                # The walk goes on as `name`, not as one of the group: with the group set aside, a
                # stand-in could never match `name in owners` above, so the entry override=True is
                # replacing would turn into a pinned owner like any other and could conflict.
                return self._inherited_modifiers(name, base, resolving=resolving | set(owners), subject_cls=subject_cls)
            first, *rest = pinned.values()
            if any(_modifier_signature(mods) != _modifier_signature(first) for mods in rest):
                cls_label = getattr(subject_cls, "__name__", subject_cls)
                listed = {other: _describe_modifiers(mods) for other, mods in pinned.items()}
                raise ValueError(
                    f"Task '{name}' ({cls_label!r}) would inherit the modifiers of "
                    f"{base.__name__}, but that class is registered under several names with different ones: "
                    f"{listed}. Pass modifiers=(...) to @TASK_REGISTRY.register() to choose (() for none)."
                )
            # A name for this class that inherits takes those same mods, so it cannot disagree:
            # the only registrations that can are the ones that named their own.
            return first
        return ()

    def _ensure_discovered(self) -> dict[str, str]:
        if self._discovered is None:
            entry_point_packages, broken_entry_points = _load_entry_points()
            if self._discovered is not None:
                # Locating an entry point's package imports its parents, and one whose __init__ looks
                # up a task has just run a whole scan of its own, entry points included.
                return self._discovered
            self._broken_entry_points = broken_entry_points
            # eden.tasks first, so a clash lists the built-in first; a package both added with
            # add_search_path and named by an entry point keeps the former's label.
            self._package_sources = self._search_packages | {
                package: dist for package, dist in entry_point_packages.items() if package not in self._search_packages
            }
            scan = _scan_packages_for_tasks(list(self._package_sources))
            self._discovered, self._discovered_modifiers, self._collisions = scan
            self._register_resolved_clashes()
            for name, where in sorted(self._collisions.items()):
                # Every other task still resolves; say so now, since only a lookup of this name raises.
                # Attributed to this line, not to whichever caller happened to trigger the scan.
                warnings.warn(_collision_message(name, where), stacklevel=1)
        return self._discovered

    def _register_resolved_clashes(self) -> None:
        """Register each set-aside class the latest scan gives its name to, now that the name does not clash.

        A rescan that fixed an entry point, say, can end a clash; the class's module will not be
        imported again, so this registers it as that import would have. A class whose own package
        left the scan (its plugin uninstalled, say) stays set aside: the name belongs to the class
        the scan still finds, and a subclass still inherits through it.
        """
        for key, (name, obj, entry) in list(self._clashing_registrations.items()):
            if self._is_colliding_class(name, obj) or self._discovered.get(name, key) != key:
                continue
            del self._clashing_registrations[key]
            if name not in self._registry:
                self._write_registration(name, obj, entry)

    def _lookup(self, name: str) -> str | None:
        """The ``module:Class`` that ``name`` resolves to, registered or only discovered; ``None`` if neither.

        Scans first even for a registered name, since only the scan knows whether the name clashes.

        Raises
        ------
        ValueError
            If ``name`` is registered in several scanned files.
        RuntimeError
            If ``name`` is not found and a broken entry point names a package that could hold it.
        """
        discovered = self._ensure_discovered()
        registered = self._registry.get(name)
        self._raise_if_clashing(name, registered)
        if registered is not None:
            return _class_key(registered)
        if name in discovered:
            return discovered[name]
        if self._broken_entry_points:  # the message's close-match search would cost every miss ~60 µs
            self._raise_if_scan_incomplete(self.missing_task_message(name))
        return None

    def _is_colliding_class(self, name: str, obj: Any) -> bool:
        """Whether the scan found ``name`` on several classes and ``obj`` is one of them."""
        return any(found == _class_key(obj) for found, _ in self._collisions.get(name, ()))

    def _raise_if_clashing(self, name: str, registered: Any) -> None:
        """Raise if ``name`` is registered in several scanned files, so which one builds is undefined.

        ``registered`` is the class an eager registration holds for ``name``, or ``None``. One the
        scan did not find among the clashing classes — a ``register(..., override=True)`` made at
        run time — is a deliberate choice between them, and answers.
        """
        if name not in self._collisions:
            return
        if registered is not None and not self._is_colliding_class(name, registered):
            return
        raise ValueError(_collision_message(name, self._collisions[name]))

    def _raise_if_scan_incomplete(self, problem: str, *, module: str | None = None) -> None:
        """Raise ``problem`` if a broken entry point kept a task package out of the scan.

        Called where a lookup came up empty or a listing is asked for — the answers that package
        could change. A lookup the scan did answer goes ahead, so one broken plugin cannot stop
        every built-in task. A task name could be in any package; a ``module`` can only be in the
        package it is under, so passing one limits the check to the entry points naming that package.
        """
        broken = [
            (message, cause)
            for package, message, cause in self._broken_entry_points
            if module is None or _is_under(module, package)
        ]
        if not broken:
            return
        messages = "\n\n".join(message for message, _ in broken)
        raise RuntimeError(
            f"{problem}\n\nThe scan could not read every task package, and the answer may lie in one it "
            f"skipped:\n\n{messages}\n\nDo NOT catch this and treat the task as missing: until the "
            "entry point resolves, no lookup can tell whether its package holds the task."
        ) from broken[0][1]

    def _invalidate_discovery_cache(self) -> None:
        """For tests: force the next lookup to re-scan the filesystem."""
        self._discovered = None


def _load_entry_points() -> tuple[dict[str, str], list[tuple[str, str, BaseException | None]]]:
    """Each installed distribution's :data:`TASK_ENTRY_POINT_GROUP` packages, for the scan about to run.

    Returns the scannable ones as ``{package: distribution}`` and the broken ones as
    ``[(the package it names, message, cause)]``. A broken entry point is recorded, not scanned: it
    warns here, and :meth:`TaskRegistry._raise_if_scan_incomplete` raises it from every lookup it
    could change. The scan skips a missing package silently (:func:`_iter_package_py_files`), so
    without this a typo in a distribution's ``pyproject.toml`` would surface only as "no task named
    ..." for each of its tasks.
    """
    packages: dict[str, str] = {}
    broken: list[tuple[str, str, BaseException | None]] = []
    for entry_point in entry_points(group=TASK_ENTRY_POINT_GROUP):
        dist = getattr(entry_point.dist, "name", None) or "<unknown distribution>"
        error = _task_entry_point_error(entry_point, dist)
        if error is not None:
            warnings.warn(
                f"{error[0]}\nTasks in other packages still resolve; a lookup that could have found its "
                "task in this package raises this instead.",
                stacklevel=1,  # this line, not whichever caller happened to trigger the scan
            )
            broken.append((entry_point.module, *error))
            continue
        # A package declared twice keeps its first source.
        packages.setdefault(entry_point.value, dist)
    return packages, broken


def _is_under(module: str, package: str) -> bool:
    """Whether ``module`` is ``package`` or lies inside it."""
    return module == package or module.startswith(f"{package}.")


# ---------------------------------------------------------------------------
# AST-based auto-discovery
# ---------------------------------------------------------------------------


# Modules that legitimately export ``TASK_REGISTRY``. A decorator matches only
# when the identifier it uses is bound (directly or via ``as``) to one of these.
_TASK_REGISTRY_IMPORT_SOURCES = frozenset({"eden.tasks", "eden.tasks.registry"})


def _task_registry_aliases(tree: ast.AST) -> set[str]:
    """Return the set of local names in ``tree`` bound to ``TASK_REGISTRY``.

    Only ``from eden.tasks[.registry] import TASK_REGISTRY [as X]`` forms are
    recognized (absolute imports; matches AGENTS.md's import convention). Names
    introduced by other routes (``import eden.tasks``, relative imports,
    re-export through a third module) are not tracked — decorators relying on
    those will either (a) match the bare ``TASK_REGISTRY`` fallback in
    :func:`_register_decorator` and emit a warning, or (b) not match at
    all if the identifier is unrecognized.
    """
    aliases: set[str] = set()
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ImportFrom) and node.module in _TASK_REGISTRY_IMPORT_SOURCES:
            for alias in node.names:
                if alias.name == "TASK_REGISTRY":
                    aliases.add(alias.asname or alias.name)
    return aliases


def _modifier_display_name(node: ast.expr) -> str:
    """Best-effort identifier for a single ``modifiers=`` tuple element (display only).

    Handles the forms that actually occur in decorators — ``FooMod`` (Name),
    ``FooMod()`` (Call) and ``pkg.FooMod`` (Attribute) — and falls back to the
    source text for anything else. This is for the ``eden list`` column, so an
    approximate label beats importing the module to get the exact class name.
    """
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Call):
        return _modifier_display_name(node.func)
    if isinstance(node, ast.Attribute):
        return node.attr
    return ast.unparse(node)


def _modifier_names_from_decorator(cls_def: ast.ClassDef, aliases: set[str] | frozenset[str]) -> list[str] | None:
    """Extract modifier display-names from a class's ``@<alias>.register(modifiers=...)`` decorator.

    Non-importing counterpart to reading the live ``modifiers=`` tuple. Returns ``None`` when the
    decorator passes no ``modifiers=`` keyword — the task inherits its base class's, which
    :func:`_resolve_inherited_modifier_names` fills in once every file is scanned — and ``[]``
    when the value is not a literal tuple/list or the class has no such decorator.
    """
    dec = _register_decorator(cls_def, aliases=aliases)
    if dec is None:
        return []
    value = _decorator_keyword(dec, "modifiers")
    if value is None:
        return None
    if isinstance(value, (ast.Tuple, ast.List)):
        return [_modifier_display_name(elt) for elt in value.elts]
    return []


def _base_class_names(cls_def: ast.ClassDef) -> list[str]:
    """The simple names of a class's bases (``Foo`` for both ``Foo`` and ``pkg.Foo``), in order."""
    out = []
    for base in cls_def.bases:
        if isinstance(base, ast.Name):
            out.append(base.id)
        elif isinstance(base, ast.Attribute):
            out.append(base.attr)
    return out


def _resolve_inherited_modifier_names(
    modifiers: dict[str, list[str] | None],
    task_classes: dict[str, str],
    class_bases: dict[str, list[list[str]]],
) -> None:
    """Replace each ``None`` in ``modifiers`` (no ``modifiers=``: inherit) with its ancestor's names.

    The AST counterpart of :meth:`TaskRegistry._inherited_modifiers`, for the import-free
    ``eden list`` column. Classes are matched by simple name across every scanned file, walking
    bases depth-first (the MRO of any single-inheritance chain) through unregistered intermediate
    classes too. It stops at a name defined more than once, and a task whose ancestry leaves the
    scanned packages lists no modifiers — this is a display label, and a best-effort one beats
    importing the module to find out.

    Parameters
    ----------
    modifiers: dict[str, list[str] | None]
        Task name to modifier display-names; updated in place.
    task_classes: dict[str, str]
        Task name to the simple name of its config class.
    class_bases: dict[str, list[list[str]]]
        Simple class name to the base names of each scanned definition of it.
    """
    task_by_class: dict[str, str] = {}
    for task, cls_name in task_classes.items():
        task_by_class.setdefault(cls_name, task)

    def resolve(cls_name: str, seen: frozenset[str]) -> list[str] | None:
        """The modifiers ``cls_name`` holds or passes on, or None if neither it nor an ancestor is a task."""
        task = task_by_class.get(cls_name)
        if task is not None and modifiers[task] is not None:
            return modifiers[task]
        inherited = None
        definitions = class_bases.get(cls_name, [])
        if len(definitions) == 1:
            for base in definitions[0]:
                if base in seen:
                    continue
                inherited = resolve(base, seen | {base})
                if inherited is not None:
                    break
        if task is None:
            return inherited  # unregistered: pass on whatever an ancestor has, or None
        # A registered class ends the walk even when it has no modifiers, as it does in the live MRO walk.
        return inherited or []

    for task, cls_name in task_classes.items():
        if modifiers[task] is None:
            modifiers[task] = resolve(cls_name, frozenset({cls_name}))


class _TaskScan(NamedTuple):
    """What :func:`_scan_packages_for_tasks` found."""

    # name -> "module.path:ClassName", for every name exactly one scanned class registers.
    tasks: dict[str, str]
    # name -> the modifier display-names its decorator declares or inherits, for listing without imports.
    modifiers: dict[str, list[str]]
    # name -> [("module:Class", file path), ...], for every name several scanned classes register.
    collisions: dict[str, list[tuple[str, str]]]


def _scan_packages_for_tasks(packages: list[str]) -> _TaskScan:
    """Find every class in ``packages`` decorated with ``@TASK_REGISTRY.register`` (with or without args).

    The scan does NOT import the target modules — it parses their source with
    :mod:`ast` so auto-discovery preserves the lazy-import contract.

    Matching is alias-aware: a decorator is recognized whenever its attribute
    base is a local name bound to ``TASK_REGISTRY`` via ``from eden.tasks[.registry]
    import TASK_REGISTRY [as X]``. When a file uses the literal name
    ``TASK_REGISTRY`` but hasn't imported it from a recognized location, the
    scan emits a warning so the miss is loud rather than silent.

    A name registered by more than one class is left out of ``tasks`` and reported in
    ``collisions``, for :class:`TaskRegistry` to raise from the lookups it affects; collisions with
    explicitly-registered (eager/lazy) names are resolved later at
    :meth:`TaskRegistry.get` time (explicit wins).
    """
    # name -> every (module:Class, file path) registering it; one entry unless it collides.
    found: dict[str, list[tuple[str, str]]] = {}
    # For resolving inherited modifier names once every file is read (see
    # _resolve_inherited_modifier_names): each task's class name, and every scanned class's bases.
    found_modifiers: dict[str, list[str] | None] = {}
    task_classes: dict[str, str] = {}
    class_bases: dict[str, list[list[str]]] = {}
    for package in packages:
        for module_path, file_path in _iter_package_py_files(package):
            try:
                source = file_path.read_text(encoding="utf-8")
            except OSError as exc:
                warnings.warn(
                    f"TaskRegistry auto-discovery: could not read '{file_path}': {exc}. "
                    "Tasks defined in this file will not be discoverable.",
                    stacklevel=2,
                )
                continue
            try:
                tree = ast.parse(source, filename=str(file_path))
            except SyntaxError as exc:
                warnings.warn(
                    f"TaskRegistry auto-discovery: syntax error in '{file_path}': {exc}. "
                    "Tasks defined in this file will not be discoverable.",
                    stacklevel=2,
                )
                continue
            aliases = _task_registry_aliases(tree)
            for node in ast.walk(tree):
                if not isinstance(node, ast.ClassDef):
                    continue
                class_bases.setdefault(node.name, []).append(_base_class_names(node))
                registered_name = _task_name_from_decorators(node, aliases=aliases, file_path=file_path)
                if registered_name is None:
                    continue
                qualname = f"{module_path}:{node.name}"
                where = found.setdefault(registered_name, [])
                if all(qualname != seen for seen, _ in where):
                    where.append((qualname, str(file_path)))
                task_classes[registered_name] = node.name
                found_modifiers[registered_name] = _modifier_names_from_decorator(node, aliases)
    _resolve_inherited_modifier_names(found_modifiers, task_classes, class_bases)
    return _TaskScan(
        tasks={name: where[0][0] for name, where in found.items() if len(where) == 1},
        modifiers=found_modifiers,
        collisions={name: where for name, where in found.items() if len(where) > 1},
    )


def _class_key(cls: Any) -> str:
    """``"module:Class"`` for a class, as the AST scan records it."""
    return f"{cls.__module__}:{cls.__name__}"


def _collision_message(name: str, where: list[tuple[str, str]]) -> str:
    """The error for a task name registered by several scanned classes, listing each one."""
    listed = "".join(f"\n  - {path} ({qualname})" for qualname, path in where)
    return (
        f"Task name '{name}' is registered in {len(where)} locations:{listed}\n"
        "Fix: pass a distinct name='...' to @TASK_REGISTRY.register() on all but one; to start from a "
        "built-in task, subclass its config under a new name rather than reusing the built-in's.\n"
        "Do NOT add override=True to pick one: the scan reads files without importing them, so it has no "
        "import order to apply it in, and the clash stands either way."
    )


def _task_entry_point_error(entry_point: EntryPoint, dist: str) -> tuple[str, BaseException | None] | None:
    """Why ``entry_point`` cannot be scanned for tasks, and the exception behind it; ``None`` when it can.

    Looking the package up imports its parent packages (not the package itself), as the scan does,
    so a failure there is either a wrong entry point or a parent that fails to import on its own
    account. Those take different fixes, so they get different messages.
    """
    value = entry_point.value
    head = f"Distribution '{dist}' names '{value}' under the '{TASK_ENTRY_POINT_GROUP}' entry-point group, but"
    fix_entry_point = (
        f"Fix: in {dist}'s pyproject.toml, set the entry point to the dotted package that holds its task "
        f'configs ([project.entry-points."{TASK_ENTRY_POINT_GROUP}"] {entry_point.name} = "<package>"), '
        f"then reinstall it; or uninstall {dist}."
    )
    if entry_point.attr is not None:
        return f"{head} it names an object, not a package — drop the ':{entry_point.attr}'.\n{fix_entry_point}", None
    try:
        spec = find_spec(value)
    except Exception as exc:  # noqa: BLE001 — kept as the cause and raised from every lookup it affects
        # 'a.b.c' and every package above it: a ModuleNotFoundError naming one of them means the
        # entry point is wrong; naming anything else, a parent's own import failed.
        own_packages = {value.rsplit(".", depth)[0] for depth in range(value.count(".") + 1)}
        missing = exc.name if isinstance(exc, ImportError) else None
        if missing in own_packages:
            return f"{head} no package '{missing}' is importable.\n{fix_entry_point}", exc
        if "." not in value:  # no parent to blame: the lookup of the package itself failed
            return f"{head} looking it up raised {type(exc).__name__}: {exc}.\n{fix_entry_point}", exc
        parents = ", ".join(f"'{package}'" for package in sorted(own_packages - {value}, key=len))
        return (
            f"{head} importing the packages above it ({parents}) raised {type(exc).__name__}: {exc}.\n"
            f"Fix: the entry point is fine — those packages fail to import on their own. Install what they "
            f"import (a missing module usually means an extra of {dist} is not installed), or fix the failing "
            "import in their __init__.py; the traceback above this error shows which line.\n"
            "Do NOT point the entry point somewhere else to get past this: the tasks are where it says.",
            exc,
        )
    if spec is None:
        return f"{head} no package of that name is importable.\n{fix_entry_point}", None
    if not spec.submodule_search_locations:
        return (
            f"{head} it is a module ({spec.origin}), not a package — the scan reads a package's *.py files.\n"
            f"{fix_entry_point}",
            None,
        )
    return None


def _iter_package_py_files(package: str):
    """Yield ``(dotted.module.path, Path)`` for every scannable ``*.py`` file under ``package``.

    Two kinds of files are skipped:

    * ``__init__.py`` — typically re-exports; tasks live in siblings.
    * files whose name starts with ``_`` (e.g. ``_meta.py``) — private modules.

    Writing ``@TASK_REGISTRY.register()`` inside a skipped file will NOT be
    discovered; put the decorated class in a regular module (e.g. ``config.py``).

    Namespace packages are handled via the importlib spec.
    """
    try:
        spec = find_spec(package)
    except (ImportError, ValueError):
        # ImportError if a parent package is missing; ValueError if the module
        # was imported but has a mangled spec. Either way: nothing to discover.
        return
    if spec is None:
        return
    search_locations = list(spec.submodule_search_locations or [])
    for root_dir in search_locations:
        root = Path(root_dir)
        if not root.exists():
            continue
        for py_file in sorted(root.rglob("*.py")):
            name = py_file.name
            if name == "__init__.py":
                continue
            if name.startswith("_"):
                continue
            relative = py_file.relative_to(root).with_suffix("")
            module_path = package + "." + ".".join(relative.parts)
            yield module_path, py_file


def _register_decorator(
    cls_def: ast.ClassDef,
    *,
    aliases: set[str] | frozenset[str] = frozenset({"TASK_REGISTRY"}),
    file_path: Path | None = None,
) -> ast.expr | None:
    """Return the class's first ``@<alias>.register`` decorator, called or bare, or None.

    ``aliases`` is the set of local names the scanned module has bound to
    ``TASK_REGISTRY`` (see :func:`_task_registry_aliases`). The default
    ``{"TASK_REGISTRY"}`` lets this helper be called in isolation (e.g. from
    unit tests) without import-tracking.

    When a decorator literally uses the name ``TASK_REGISTRY`` but it's not in
    ``aliases`` (i.e. the module didn't import it from a recognized location),
    a warning is emitted so the miss is loud rather than silent. Decorators
    that use an unrelated ``*_REGISTRY`` name are ignored quietly — Eden has
    several term registries that share the ``.register()`` shape.
    """
    for dec in cls_def.decorator_list:
        func = dec.func if isinstance(dec, ast.Call) else dec
        # Must look like <name>.register
        if not (isinstance(func, ast.Attribute) and func.attr == "register" and isinstance(func.value, ast.Name)):
            continue
        if func.value.id in aliases:
            return dec
        if func.value.id == "TASK_REGISTRY":
            # Loud miss: the file used the canonical name but didn't import it.
            where = f" in '{file_path}'" if file_path is not None else ""
            warnings.warn(
                f"TaskRegistry auto-discovery: @TASK_REGISTRY.register on "
                f"'{cls_def.name}'{where} is not recognized — the module "
                f"does not import TASK_REGISTRY from 'eden.tasks' or "
                f"'eden.tasks.registry'. This task will not be discovered. "
                f"Add `from eden.tasks import TASK_REGISTRY` at the top of "
                f"the file.",
                stacklevel=2,
            )
    return None


def _decorator_keyword(dec: ast.expr, arg: str) -> ast.expr | None:
    """Return the value a decorator call passes for keyword ``arg``; None when it passes none or is bare."""
    keywords = dec.keywords if isinstance(dec, ast.Call) else []
    return next((kw.value for kw in keywords if kw.arg == arg), None)


def _task_name_from_decorators(
    cls_def: ast.ClassDef,
    *,
    aliases: set[str] | frozenset[str] = frozenset({"TASK_REGISTRY"}),
    file_path: Path | None = None,
) -> str | None:
    """Return the registered task name if a decorator is ``@<alias>.register``, else None.

    Matches ``@<alias>.register`` with or without call args (see :func:`_register_decorator`). A
    literal ``name=`` wins over the name derived from the class.
    """
    dec = _register_decorator(cls_def, aliases=aliases, file_path=file_path)
    if dec is None:
        return None
    name = _decorator_keyword(dec, "name")
    if isinstance(name, ast.Constant) and isinstance(name.value, str):
        return name.value
    return _default_task_name(cls_def.name)


def _modifier_signature(mods: tuple[TaskMod, ...]) -> list[tuple[type, dict[str, Any]]]:
    """What makes two modifier tuples interchangeable: each mod's class and instance state, in order."""
    return [(type(m), vars(m)) for m in mods]


# A value repr longer than this loses its middle in the ambiguity error. It is far above any path,
# name or short list a mod is built with, so those always print in full; torch and numpy summarize
# big arrays themselves, so in practice only a large container of plain values is cut.
_MODIFIER_VALUE_REPR_LIMIT = 1000


def _describe_modifiers(mods: tuple[TaskMod, ...]) -> list[str]:
    """Each mod as ``Class(field=value, ...)`` — the state :func:`_modifier_signature` compares.

    So two sets reported as different read as different: values print in full, up to
    :data:`_MODIFIER_VALUE_REPR_LIMIT` characters each.
    """
    return [f"{type(m).__name__}({', '.join(f'{k}={_describe_value(v)}' for k, v in vars(m).items())})" for m in mods]


def _describe_value(value: Any) -> str:
    """``repr(value)``, with its middle elided past the limit; never raises, since it builds an error message."""
    try:
        text = repr(value)
    except Exception:
        return f"<{type(value).__name__} whose __repr__ raised>"
    if len(text) <= _MODIFIER_VALUE_REPR_LIMIT:
        return text
    keep = _MODIFIER_VALUE_REPR_LIMIT // 2
    return f"{text[:keep]}...<{len(text) - 2 * keep} chars elided>...{text[-keep:]}"


def _normalize_modifier_entries(entries: tuple[ModifierEntry, ...]) -> tuple[TaskMod, ...]:
    """Normalize registration-time entries to a tuple of TaskMod instances.

    Bare TaskMod subclasses are instantiated with no args eagerly so that
    registration-time errors surface immediately; instances pass through.
    """
    out: list[TaskMod] = []
    for e in entries:
        if isinstance(e, TaskMod):
            out.append(e)
        elif isinstance(e, type) and issubclass(e, TaskMod):
            out.append(e())
        else:
            raise TypeError(f"Modifier entries must be TaskMod instances or TaskMod subclasses; got {e!r}")
    return tuple(out)


# ---------------------------------------------------------------------------
# Helpers used by the CLI parser (eden/tasks/parser.py)
# ---------------------------------------------------------------------------


def literal_choices(annotation: Any) -> tuple[Any, ...] | None:
    """Return the choices for a Literal annotation, or None if not a Literal."""
    return get_args(annotation) if get_origin(annotation) is Literal else None


# Module-level singleton; populated by the ``@TASK_REGISTRY.register()`` decorators as their modules are imported
# (a lookup imports the module the AST scan found the task in).
TASK_REGISTRY = TaskRegistry()
