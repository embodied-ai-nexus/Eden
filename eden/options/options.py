"""ConfigurableOptions: base class for all Eden options.

:class:`ConfigurableOptions` extends the Genesis ``Options`` Pydantic model with Eden
conventions: ``lock()``/``unlock()`` to freeze/unfreeze a config, ``clone()`` for a
lock-aware deep copy, and a best-effort ``__deepcopy__`` that tolerates unpicklable
objects (e.g. USD stages).

**Copying.** ``clone()`` is the canonical way to duplicate an options object — prefer
it (or bare ``copy.deepcopy``) over pydantic's ``model_copy(deep=True)``, because only
the former routes through this class's ``__deepcopy__`` and therefore inherits the
unpicklable-safe fallback (a value that can't be deep-copied is shared by reference
rather than raising). ``__deepcopy__`` is *faithful*: the clone mirrors the source's
lock state. ``clone(unlock=True)`` is the explicit "copy so I can edit it" path — it
returns a mutable clone regardless of the source, leaving the source untouched.
``updated(**fields)`` is the copy-with-changes path: a deep copy with those fields replaced,
whose names are validated like construction (so is ``model_copy(update=...)``).

**Locking.** Locking is *tree-scoped and downward-propagating*: ``lock()``/``unlock()``
apply to the node they are called on and recurse into all nested options (through
declared fields, extras, and any list/tuple/dict of options). They never walk upward,
so a child can lock/unlock its own subtree without affecting its parent — reaching a
mixed state (e.g. a locked config with one locally-unlocked child) is intentional and
supported; the ``__setattr__`` guard keys off each node's own ``_locked``. Both calls
are idempotent per node (a no-op if the node is already in the target state), so a
redundant parent ``lock()`` does not clobber a child that was toggled locally. The lock is not part of an
options object's value: a locked object ``==`` the same values unlocked.

The default is ``extra="forbid"`` so every fixed-schema options class (``EnvOptions``,
entity/material/camera options, rsl_rl configs, ...) rejects misspelled fields loudly.
Classes that legitimately carry dynamically-named children (scene entities, manager
terms, observation terms, cameras, sensors, splat PCDs) or receive framework-injected
``.configure()`` params (``name`` / ``params``) as extras subclass
:class:`DynamicConfigurableOptions` instead, which restores ``extra="allow"``.
"""

import copy
import difflib
import inspect
from collections.abc import Mapping
from functools import cached_property
from typing import Any, Self, TypeVar

from genesis.options.options import Options
from pydantic import BaseModel, PrivateAttr, ValidationError
from pydantic_core import InitErrorDetails, PydanticCustomError

from eden.options._metadata import OPTION_METADATA_KEYS, getattribute_with_extra, option_metadata
from eden.utils.params import ParamInfo, render_annotation, render_params

T = TypeVar("T", bound=Options)


def _field_specs(cls: type) -> list[ParamInfo]:
    """The declared fields of ``cls`` as :class:`ParamInfo`, for the same rendering ``eden terms``
    and the term-param errors use — so "unknown field" and "unknown term param" read alike.

    Types only, no defaults: a config field's default can be a whole nested options object whose
    repr would swamp the message (unlike a term param, which is always a scalar).
    """
    return [
        ParamInfo(name, render_annotation(field.annotation), "", required=field.is_required())
        for name, field in cls.model_fields.items()
        if name not in OPTION_METADATA_KEYS
    ]


def _is_cached_property(cls: type, key: str) -> bool:
    """Whether ``key`` in an instance ``__dict__`` is a materialized ``cached_property`` of ``cls``
    — derived state a copy must recompute, not model state it may carry over.
    """
    return isinstance(getattr(cls, key, None), cached_property)


def _unknown_field_message(cls: type, field: str) -> str:
    """The one wording for "``field`` is not a field of ``cls``" — names the valid fields and the
    closest one, which is the only thing that unsticks the author.

    Shared by construction (:func:`_enrich_extra_forbidden`) and ``model_copy(update=...)``
    (:meth:`ConfigurableOptions._validate_copy_update`) so the same typo reads the same either way.
    """
    valid = sorted(f for f in cls.model_fields if f not in OPTION_METADATA_KEYS)
    close = difflib.get_close_matches(field, valid, n=1)
    hint = f" Did you mean '{close[0]}'?" if close else ""
    return f"'{field}' is not a field of {cls.__name__}.{hint} Valid fields: {render_params(_field_specs(cls))}."


def _enrich_extra_forbidden(cls: type, exc: ValidationError) -> ValidationError | None:
    """Rewrite an ``extra_forbidden`` rejection to list the class's valid fields.

    Pydantic already names the offending field but not the ones that would have worked, which
    is the only thing that unsticks the author (an ``extra="forbid"`` config class like
    ``RslRlOnPolicyRunnerOptions`` would otherwise reject a typo without teaching the field set).
    When *every* error is an unknown-field rejection, return an equivalent ``ValidationError``
    whose message lists the valid fields — plus a ``did you mean`` when one is close. The rebuilt
    error keeps ``type=extra_forbidden`` (via :class:`PydanticCustomError`) so callers and tests
    that key on that type still see it. Returns ``None`` when any error is *not* extra_forbidden,
    so a genuine type error alongside is never masked or reworded.

    Only a **top-level** unknown field (``loc`` of length 1) can be attributed to *this* class.
    A nested one — ``loc == ("child", "bogus")`` on the config-load / dict-override path — was
    already validated against its own class (nested ``ConfigurableOptions`` get their ``__init__``
    called, so a nested typo arrives here *already* enriched with the right class). Enriching it
    again against ``cls`` would clobber that correct message with the wrong class's fields, so a
    nested error is passed through with the message it already carries. ``None`` is returned when
    *every* error is nested (re-raise the original untouched) or when any error is not
    extra_forbidden (never mask a real type error).
    """
    errors = exc.errors()
    if not errors or any(e["type"] != "extra_forbidden" for e in errors):
        return None
    if not any(len(e["loc"]) == 1 for e in errors):
        return None
    line_errors: list[InitErrorDetails] = []
    for e in errors:
        loc = e["loc"]
        if len(loc) == 1:
            message = _unknown_field_message(cls, str(loc[0]))
        else:
            # Nested: reuse the message pydantic already rendered (the nested class's own enriched
            # text, or its native "Extra inputs are not permitted") — attributing it to cls here
            # would name the wrong class. Safe as a template: extra_forbidden messages carry no
            # ``{}`` (field names are identifiers; field lists render with ``[]``).
            message = e["msg"]
        # error_type stays "extra_forbidden"; only the human message changes.
        line_errors.append(
            InitErrorDetails(type=PydanticCustomError("extra_forbidden", message), loc=loc, input=e.get("input"))
        )
    return ValidationError.from_exception_data(cls.__name__, line_errors)


class ConfigurableOptions(Options):
    """Base class for configurable options.

    Use with ConfigurableMixin.
    """

    # Reject unknown fields by default; DynamicConfigurableOptions opts back into extras.
    # strict=False (overriding the genesis Options base default of strict=True) so a JSON round-trip
    # coerces cleanly on validated load — list->tuple, str->enum, nested-list->ndarray — while still
    # type-checking (bad types / wrong shapes still raise). This is the enabler for native
    # model_validate load; embedded genesis Options keep their own strict=True.
    model_config = {"extra": "forbid", "strict": False}

    _locked: bool = PrivateAttr(default=False)

    def __eq__(self, other: Any) -> bool:
        """``BaseModel.__eq__`` without the lock state: two options are equal when their values are.

        Pydantic compares private attributes too, so without this a locked config (``save_as_json`` locks by default)
        never equals the same values freshly built, and a value check such as "does the declared camera already carry
        these overlays?" (``EdenConfig.add_recording_camera``) answers no. ``_locked`` is a mode, not a value; any other
        private attribute a subclass declares still counts.
        """
        if not isinstance(other, ConfigurableOptions):
            return super().__eq__(other)

        def values(options: ConfigurableOptions) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
            private = {key: value for key, value in (options.__pydantic_private__ or {}).items() if key != "_locked"}
            # Declared fields only: ``__dict__`` also holds materialized ``cached_property`` values, which are not
            # state. A field missing from ``__dict__`` (the deprecated ``copy()`` can leave one out) is left out here
            # too, so it does not equal a field set to None.
            fields = {name: options.__dict__[name] for name in type(options).model_fields if name in options.__dict__}
            return fields, options.__pydantic_extra__ or {}, private

        return type(self) is type(other) and values(self) == values(other)

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        """``BaseModel.model_dump`` that serializes nested options by their *actual* class.

        Eden's option tree is polymorphic throughout — ``surface: SurfaceOptions`` holds an
        ``Emission``, ``arm: RobotOptions`` holds a ``FrankaResearch3``, ``distribution_cfg:
        RslRlDistributionOptions`` holds a ``RslRlGaussianDistributionOptions``. Pydantic v2
        serializes a field by its *declared* type, so every field the subclass adds is
        silently dropped from the parent's dump: no error, no warning, just a smaller dict.

        That narrowing is wrong for every consumer Eden has (e.g. a ``distribution_cfg`` dumped
        without ``std_type`` and ``init_std``), and ~300 option fields share the polymorphic shape.
        This override makes duck-typed serialization the default, so ``model_dump()`` and
        ``dict()`` / ``to_dict()`` (through which the training wrappers dump ``runner_options``)
        agree; ``dict()`` relies on it rather than passing the flag itself.

        ``serialize_as_any=True`` restores duck-typed serialization. It is a default, not a
        forced value, so a caller that genuinely wants declared-type narrowing can still
        pass ``serialize_as_any=False``.

        Parameters
        ----------
        **kwargs
            Forwarded to :meth:`pydantic.BaseModel.model_dump`.

        Returns
        -------
        dict
            The serialized options, including fields contributed by subclasses.
        """
        kwargs.setdefault("serialize_as_any", True)
        return super().model_dump(**kwargs)

    def model_dump_json(self, **kwargs: Any) -> str:
        """``BaseModel.model_dump_json`` with the same duck-typed nesting as :meth:`model_dump`.

        Parameters
        ----------
        **kwargs
            Forwarded to :meth:`pydantic.BaseModel.model_dump_json`.

        Returns
        -------
        str
            The serialized options as JSON.
        """
        kwargs.setdefault("serialize_as_any", True)
        return super().model_dump_json(**kwargs)

    def __copy__(self):
        """Shallow-copy the options, leaving materialized cached properties behind.

        Pydantic's ``__copy__`` copies ``__dict__`` wholesale, which is where a ``cached_property``
        parks its value — and :attr:`keys`'s value is a closure over its owner. A copy that kept it
        would go on enumerating the *source's* children: it would not see a term added to it, and
        the build-time unused-parameter check (``ConfigurableMixin.__init__`` reads ``keys()``)
        would not see a typo parked on it. Same rule as :meth:`__deepcopy__`, and the one
        place shallow copies get it — ``copy.copy`` and ``model_copy`` both land here.
        """
        copied = super().__copy__()
        for key in [key for key in copied.__dict__ if _is_cached_property(type(self), key)]:
            del copied.__dict__[key]
        return copied

    def __deepcopy__(self, memo=None):
        """Deep-copy the options faithfully, sharing only values that cannot be copied.

        Copy the already-validated model state directly, following Pydantic's own
        ``BaseModel.__deepcopy__`` shape rather than rebuilding through ``model_construct``.
        Reconstruction re-runs ``model_post_init``; besides changing a mutated source while it
        is being copied, that can make a parent silently share a locked child when the hook tries
        to write through the child's lock and the best-effort fallback catches the exception.

        Some pxr objects (e.g. ``Usd.Stage`` inside USD morphs) are not picklable, so values are
        still copied independently and fall back to their original reference on failure.
        """
        if memo is None:
            memo = {}

        cls = type(self)
        new_obj = cls.__new__(cls)
        # Register before descending so repeated references stay shared within the clone and a
        # (defensive) cycle back to this model resolves to the clone rather than recursing.
        memo[id(self)] = new_obj

        def safe_deepcopy(value):
            try:
                return copy.deepcopy(value, memo)
            except Exception:
                return value

        object.__setattr__(
            new_obj,
            "__dict__",
            {
                key: safe_deepcopy(value)
                for key, value in self.__dict__.items()
                # A materialized cached_property is not model state. ``keys``'s cached value is a
                # closure over its owner, and deepcopy treats functions as atomic, so carrying it
                # over would leave the clone answering for the *source* — a scene clone would keep
                # enumerating the source's entities (EnvBase._load_scene iterates
                # ``scene_options.keys()``) and pin the source alive. Drop the cache instead; the
                # clone recomputes on first access.
                if not _is_cached_property(cls, key)
            },
        )
        extra = getattr(self, "__pydantic_extra__", None)
        object.__setattr__(
            new_obj,
            "__pydantic_extra__",
            None if extra is None else {key: safe_deepcopy(value) for key, value in extra.items()},
        )
        private = getattr(self, "__pydantic_private__", None)
        object.__setattr__(
            new_obj,
            "__pydantic_private__",
            None if private is None else {key: safe_deepcopy(value) for key, value in private.items()},
        )
        object.__setattr__(new_obj, "__pydantic_fields_set__", set(self.__pydantic_fields_set__))

        return new_obj

    def lock(self):
        """Freeze this options node against further mutation, recursing into all children.

        Tree-scoped and downward-propagating: applies to this node and every nested
        options object (declared fields, extras, and options inside list/tuple/dict).
        It never walks upward, so locking a child does not lock its parent. Idempotent
        (a no-op if already locked). See the module docstring for the full lock model.
        """
        if self._locked:
            return

        self._locked = True

        def lock_recursive(obj):
            if isinstance(obj, ConfigurableOptions):
                obj.lock()
            elif isinstance(obj, (list, tuple)):
                for item in obj:
                    lock_recursive(item)
            elif isinstance(obj, dict):
                for item in obj.values():
                    lock_recursive(item)

        for value in self.__dict__.values():
            lock_recursive(value)
        if self.__pydantic_extra__:
            for value in self.__pydantic_extra__.values():
                lock_recursive(value)

    def unlock(self):
        """Re-enable modification of this options node, recursing into all children.

        Exact mirror of :meth:`lock` (same downward propagation, no upward walk, same
        idempotency), so a child can unlock its own subtree without affecting its
        parent. ``_locked`` is exempt from the ``__setattr__`` guard, so this clears it
        even on a currently-locked object.
        """
        if not self._locked:
            return

        self._locked = False

        def unlock_recursive(obj):
            if isinstance(obj, ConfigurableOptions):
                obj.unlock()
            elif isinstance(obj, (list, tuple)):
                for item in obj:
                    unlock_recursive(item)
            elif isinstance(obj, dict):
                for item in obj.values():
                    unlock_recursive(item)

        for value in self.__dict__.values():
            unlock_recursive(value)

        if self.__pydantic_extra__:
            for value in self.__pydantic_extra__.values():
                unlock_recursive(value)

    def clone(self, unlock: bool = False) -> "ConfigurableOptions":
        """Return a deep copy of this options object — the canonical way to duplicate options.

        Prefer this (or bare ``copy.deepcopy``) over ``model_copy(deep=True)``: only these
        route through :meth:`__deepcopy__` and thus tolerate unpicklable values (shared by
        reference instead of raising). By default the clone mirrors this object's lock
        state (a copy of a locked config stays locked). Pass ``unlock=True`` for a mutable
        clone regardless of the source — the explicit "copy so I can edit it" path, so
        re-enabling mutation is visible at the call site rather than a silent side effect
        of copying. The source is never modified.
        """
        new_obj = copy.deepcopy(self)
        if unlock:
            new_obj.unlock()
        return new_obj

    def model_copy(self, *, update: Mapping[str, Any] | None = None, deep: bool = False) -> Self:
        """``BaseModel.model_copy`` whose ``update=`` keys are validated like construction.

        Pydantic documents ``update=`` as unvalidated ("you should trust this data"): a key that is
        not a field is parked on the copy and the field it was meant for keeps its old value.
        Under plain pydantic, ``EnvOptions().model_copy(update={"num_evns": 4})`` returns a copy still
        at 4096 envs and never raises; on a term options the typo surfaces only at manager build, far
        from the line that contains it. Overriding one field of an inherited option is exactly
        what ``model_copy(update=...)`` is reached for, so the *names* are checked here, through
        :meth:`_validate_copy_update` — the same rule, and the same message, as constructing the
        options with that keyword. The *values* stay unvalidated, as in pydantic.

        Parameters
        ----------
        update : Mapping[str, Any], optional
            Field values to change on the copy. Unknown names raise ``ValueError``.
        deep : bool
            Deep-copy the options instead of sharing the nested values.

        Returns
        -------
        Self
            The copy, mirroring this object's lock state.
        """
        copied = super().model_copy(update=update, deep=deep)
        if update:
            copied._validate_copy_update(update)
        return copied

    def _validate_copy_update(self, update: Mapping[str, Any]) -> None:
        """Reject ``model_copy(update=...)`` keys this class would have rejected at construction.

        Called on the *copy*, after ``update`` was applied, so a subclass whose rule is about the
        resulting object (term options checking extras against the registered term, manager options
        checking entry types) re-runs that rule instead of re-deriving it from the keys.
        ``extra="allow"`` classes have no fixed field set, so this base check applies to the
        ``extra="forbid"`` ones only.
        """
        if self.model_config.get("extra") == "allow":
            return
        unknown = [key for key in update if key not in type(self).model_fields]
        if unknown:
            raise ValueError(" ".join(_unknown_field_message(type(self), key) for key in unknown))

    def updated(self, **updates: Any) -> Self:
        """Return a deep copy with the given fields replaced — the way to override inherited options.

        ``base.track_linear_velocity.updated(weight=0.0)`` reads as what it does, and a misspelled
        name raises on that line (see :meth:`model_copy`, which this builds on: ``deep=True`` so the
        copy shares nothing with the inherited default, keywords so the names are not buried in a dict).

        The copy is always **unlocked**, even when this object is locked. A variant is made to be
        edited further, and the call already says "copy with changes" at the call site — the same
        visibility ``clone(unlock=True)`` asks for. There is no ``unlock=`` switch because every
        keyword here is a field name. ``model_copy`` stays faithful and mirrors the lock state; the
        source is never modified.

        Parameters
        ----------
        **updates
            Field values to change on the copy. Unknown names raise ``ValueError``.

        Returns
        -------
        Self
            The updated, unlocked deep copy.
        """
        copied = self.model_copy(update=updates, deep=True)
        copied.unlock()
        return copied

    def __setattr__(self, name, value):
        if getattr(self, "_locked", False) and name != "_locked":
            raise RuntimeError(f"Cannot modify attribute '{name}' of {self.__class__.__name__} because it is locked.")
        super().__setattr__(name, value)

    def __init__(self, **data):
        # Seed type metadata as construction data so BaseModel validates it into
        # __pydantic_extra__ (no-op for extra="forbid" classes, which use
        # the serialization envelope instead).
        data.update(option_metadata(self))
        # NOTE: all of the option modifications/validations should be done in the __init__ method
        try:
            BaseModel.__init__(self, **data)
        except ValidationError as exc:
            # A misspelled field on an extra="forbid" class should teach the field set, not just
            # name the typo. Only rewritten when every error is extra_forbidden.
            enriched = _enrich_extra_forbidden(type(self), exc)
            if enriched is None:
                raise
            raise enriched from None

    def __getattribute__(self, item: str):
        return getattribute_with_extra(self, item, super().__getattribute__)

    @cached_property
    def keys(self):
        """Mimic dict.keys(): declared field names + dynamic extras, without triggering value serialization."""
        excluded = OPTION_METADATA_KEYS

        def _names():
            for key in type(self).model_fields:
                if key not in excluded:
                    yield key
            extras = self.__pydantic_extra__
            if extras:
                for key in extras:
                    if key not in excluded:
                        yield key

        return lambda: tuple(_names())

    def dict(self, include_extra: bool = False):
        def clean_field(input_dict):
            return {
                key: clean_field(value) if isinstance(value, dict) else value
                for key, value in input_dict.items()
                if key not in OPTION_METADATA_KEYS
            }

        dumped = self.model_dump()
        if include_extra:
            return dumped
        return clean_field(dumped)

    def to_dict(self):
        """Interface for rsl_rl."""
        return self.dict()


class DynamicConfigurableOptions(ConfigurableOptions):
    """``ConfigurableOptions`` variant that keeps ``extra="allow"``.

    Only the classes that genuinely need it subclass this:

    * **Dynamic named children** — holders that store their children as extra fields:
      ``SceneOptions`` (entities), ``ManagerOptions`` (terms), ``ObservationGroupOptions``
      (obs terms), ``CamerasOptions`` / ``SensorsOptions``, and the splat holders
      ``SplatOptions`` / ``NyxSplatOptions`` (which collect named ``PCDOptions`` children).
    * **Framework-injected params** — the ``*TermOptions`` and other ``.configure()``
      targets, whose per-term parameters land in ``__pydantic_extra__``.

    Open extras are NOT unvalidated: manager term options subclass
    :class:`~eden.options.managers.base.ManagerTermOptions`, which validates extras against
    the registered term's parameter names at construction time whenever ``name`` resolves in
    the kind's term registry (``RewardTermOptions(name="upright", wieght=2.0)`` raises), and
    ``ConfigurableMixin.__init__`` re-checks at build time. Non-term ``.configure()`` targets
    (extension options) rely on the ``.configure()``-path name validation only.
    """

    # Re-enable extras; ConfigurableOptions defaults to extra="forbid".
    model_config = {"extra": "allow"}

    def _refuse_shadowing_child(self, name: str, *, noun: str, assigned: bool = False) -> None:
        """Raise if a child named ``name`` would collide with an attribute of this class (``clone``, ``lock``, …).

        A named child and the class's own API share one namespace, and neither spelling survives
        the collision. Constructed with one, the child is an extra, and :func:`getattribute_with_extra`
        reads extras first — so ``holder.lock`` is the child and ``config.lock()``, which calls it on
        every node, fails with "not callable". *Assigned* one, pydantic sees an existing attribute and
        writes the instance ``__dict__`` instead of the extras, so the child is silently not a child:
        no term is built, and nothing says why. The name is only the child's key, so renaming it is
        free; the check is for the holders of named children (manager options, observation groups).

        Parameters
        ----------
        name : str
            Key of the child being added.
        noun : str
            What this holder calls a child (``"term"``, ``"group"``); only used in the message.
        assigned : bool, optional
            Whether the child is being assigned rather than constructed, which decides which of the
            two failures the message describes (default=False).
        """
        if name.startswith("_") or name in type(self).model_fields:
            return
        try:
            attribute = inspect.getattr_static(type(self), name)
        except AttributeError:
            return
        if isinstance(attribute, (property, cached_property)):
            kind = "property"
        elif callable(attribute) or isinstance(attribute, (classmethod, staticmethod)):
            kind = "method"
        else:
            kind = "attribute"
        cls = type(self).__name__
        if assigned:
            consequence = f"assigning it replaces that {kind} on this object and never adds a {noun}"
        else:
            consequence = f"a {noun} under that name hides it on this object"
            if kind == "method":
                consequence += f", so .{name}() calls the {noun}"
        raise TypeError(
            f"'{name}' cannot name a {noun} of {cls}: {cls}.{name} is already a {kind}, and {consequence}. The "
            f"name is only the {noun}'s key, so rename it, e.g. '{name}_{noun}'."
        )

    def _removable_child(self, name: str, *, noun: str) -> Any:
        """Return the dynamically-named child ``name`` (an entity, a term), or raise if it cannot be removed.

        The checks behind :meth:`_remove_named_child`, split out for a caller that has its own
        preconditions to check before deleting (``SceneOptions.remove_entity``). Named children
        live in ``__pydantic_extra__``, so deleting one never passes through the ``__setattr__``
        lock guard — the lock is checked here instead, first. ``_option_*`` keys are
        serialization metadata, not children, and are never removable.

        Parameters
        ----------
        name : str
            Key of the child.
        noun : str
            What this holder calls a child (``"entity"``, ``"term"``); only used in error messages.

        Returns
        -------
        Any
            The child stored under ``name``.

        Raises
        ------
        RuntimeError
            If this holder is locked.
        KeyError
            If ``name`` is serialization metadata, or no child has that name (the message lists
            the children that do exist).
        """
        if self._locked:
            raise RuntimeError(f"Cannot remove {noun} '{name}' because {type(self).__name__} is locked.")
        if name.startswith("_option_"):
            raise KeyError(f"'{name}' is serialization metadata, not a removable {noun}.")
        extra = self.__pydantic_extra__ or {}
        if name not in extra:
            available = [key for key in extra if not key.startswith("_option_")]
            raise KeyError(f"{noun.capitalize()} '{name}' not found in {type(self).__name__}; available: {available}")
        return extra[name]

    def _remove_named_child(self, name: str, *, noun: str) -> None:
        """Remove the dynamically-named child ``name`` (an entity, a term) from this holder.

        The one deletion path behind ``SceneOptions.remove_entity`` and every ``disable_term``;
        raises as :meth:`_removable_child` does.

        Parameters
        ----------
        name : str
            Key of the child to remove.
        noun : str
            What this holder calls a child (``"entity"``, ``"term"``); only used in error messages.
        """
        self._removable_child(name, noun=noun)
        del self.__pydantic_extra__[name]
