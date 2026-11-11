"""How a configurable's parameters are described — and the wording that describes them.

A term's accepted parameter set has to be visible outside its signature: an author who guessed a
parameter name wrong learns nothing from an error that names only the offending key. This module is
the single place that renders a parameter (name, type, default) and the single place that words the
"that parameter is not accepted" errors, so:

* the three checks that run at construction (``ManagerTermOptions._validate_param_names``), at
  ``configure()`` and at build (``ConfigurableFuncWrapperMixin``) cannot drift in wording, and
* ``eden terms`` and :mod:`eden.managers.catalog` describe a term exactly the way its errors do.

No eden imports (only :mod:`inspect`, :mod:`typing` and pydantic's ``TypeAdapter``), so both
:mod:`eden.options.managers.base` and :mod:`eden.utils.common` can reach it — which is also why
``invalid_function_param_keys`` lives here rather than with the manager options (see its docstring).
"""

from __future__ import annotations

import inspect
import sys
from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from types import FunctionType, UnionType
from typing import Annotated, Any, Literal, Union, get_args, get_origin

from pydantic import TypeAdapter, ValidationError


@dataclass(frozen=True)
class ParamInfo:
    """One configurable parameter, as an author has to spell it: name, type, default.

    Every field is already-rendered text rather than a live object. The only consumers are
    human- and agent-readable listings, and the term modules use ``from __future__ import
    annotations``, so a signature's annotations arrive as source text anyway.
    """

    name: str

    #: The declared type as source text, or ``""`` when the parameter carries no annotation.
    annotation: str = ""

    #: The default as a one-line literal, or ``""`` when :attr:`required`. Not ambiguous with a
    #: default that *is* the empty string: that renders as ``"''"``.
    default: str = ""

    #: Whether the author must supply this parameter. Only function-term params are ever
    #: required — a class term's parameter is a class attribute, so it always has a default.
    required: bool = False

    def __str__(self) -> str:
        rendered = f"{self.name}: {self.annotation}" if self.annotation else self.name
        # No default to show (required, or an extra with no options field); a real empty-string
        # default is ``"''"`` and still renders.
        if self.required or not self.default:
            return rendered
        return f"{rendered} = {self.default}"


def render_annotation(annotation: Any) -> str:
    """``annotation`` as source text, or ``""`` when it is absent.

    An annotation of literal ``None`` also renders as ``""``: it is indistinguishable from
    "unannotated" for a class attribute (``dict.get`` returns ``None`` for both) and no term
    declares one.
    """
    if annotation is inspect.Parameter.empty or annotation is None:
        return ""
    return annotation if isinstance(annotation, str) else _format_type(annotation)


def _format_type(tp: Any) -> str:
    """A live type as the source text an author would write.

    ``inspect.formatannotation`` alone is not enough for the options-only fields, whose annotations
    are read off a pydantic model and so arrive fully resolved: ``ObservationTermOptions.clip`` is
    declared ``Vec2FType | None`` but reprs as ``Optional[Annotated[tuple[Annotated[float,
    FieldInfo(...)], ...], FieldInfo(...)]]``. Validation metadata is not something a caller can
    act on, so ``Annotated`` is unwrapped and the rest rebuilt by hand.

    The hand rebuild only covers ``Annotated``/``Union`` and generics whose args are themselves
    types. ``Ellipsis`` (the ``...`` in ``tuple[float, ...]``) and ``Literal`` — whose args are
    *values*, where quoting is load-bearing (``Literal['a']``, not ``Literal[a]``) — are handled
    explicitly and, for ``Literal``, delegated back to the stdlib formatter, which renders them
    exactly and never wraps ``Annotated``.
    """
    if tp is Ellipsis:
        return "..."
    if isinstance(tp, str):
        return tp
    if tp is type(None):
        return "None"
    origin, args = get_origin(tp), get_args(tp)
    if origin is Annotated:
        return _format_type(args[0])
    if origin is Literal:
        return inspect.formatannotation(tp)
    if origin is None or not args:
        return inspect.formatannotation(tp)
    if origin in (Union, UnionType):
        return " | ".join(_format_type(arg) for arg in args)
    name = getattr(origin, "__name__", None) or inspect.formatannotation(origin)
    return f"{name}[{', '.join(_format_type(arg) for arg in args)}]"


def render_default(value: Any) -> str:
    """``value`` as a one-line literal an author could paste back into a config.

    Enum members by name — ``repr`` gives ``<ReferenceSource.DEFAULT: 'default'>``, which is not
    something you can write — and any line breaks collapsed to single spaces, so a multi-line repr
    cannot split the markdown table row or terminal line it is rendered into.
    """
    if isinstance(value, Enum):
        return f"{type(value).__name__}.{value.name}"
    # Collapse only line breaks, not every space run (which would corrupt a string default's content).
    return repr(value).replace("\r\n", " ").replace("\r", " ").replace("\n", " ")


def function_param_specs(func: FunctionType, *, exclude: Collection[str] = ()) -> tuple[ParamInfo, ...]:
    """The ``params={...}`` entries an author supplies for function term ``func``, in signature order.

    Every named parameter except those in ``exclude`` (the manager-injected ``env`` / ``envs_idx``)
    is listed, because every one of them is a legal ``params`` key: ``compute`` calls
    ``func(env=..., **params)`` and ``invalid_function_param_keys`` validates against the whole
    signature. That deliberately includes positional-or-keyword params — Eden's built-ins declare
    author-supplied params keyword-only, but nothing stops an out-of-tree term from doing otherwise,
    and omitting one from the listing would describe the term wrongly.

    ``*args`` / ``**kwargs`` are dropped: neither is a parameter an author can name. A ``**kwargs``
    term accepts *any* key (see ``invalid_function_param_keys``), which a name list cannot express.
    """
    return tuple(
        ParamInfo(
            name,
            render_annotation(param.annotation),
            "" if param.default is inspect.Parameter.empty else render_default(param.default),
            required=param.default is inspect.Parameter.empty,
        )
        for name, param in inspect.signature(func).parameters.items()
        if param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        and name not in exclude
    )


def render_params(specs: Iterable[ParamInfo]) -> str:
    """``specs`` as one comma-separated line, required first — or ``"(none)"``.

    Required first because that is the part an author is missing when this text appears in an
    error; the rest is reference. Stable within each half (input order is signature order).
    """
    ordered = sorted(specs, key=lambda spec: not spec.required)
    return ", ".join(str(spec) for spec in ordered) or "(none)"


def invalid_function_param_keys(func, params) -> set[str]:
    """Return the ``params`` keys that are not parameters of ``func``.

    Empty when every key is a parameter of ``func`` — or when ``func`` accepts ``**kwargs``,
    in which case any key is legal. Shared by the construction-time check
    (:meth:`~eden.options.managers.base.ManagerTermOptions._validate_param_names`) and the
    build-time backstop (``ConfigurableFuncWrapperMixin.__init__``) so the two can't drift.
    Callers that need the accepted set for a message build it from :func:`function_param_specs`,
    which also carries the types and defaults a bare name list cannot.

    Here rather than in :mod:`eden.options.managers.base` because :mod:`eden.utils.common` is its
    other caller and ``eden.options.entities`` reaches back into ``eden.utils`` (via
    ``eden.utils.terrains``) — so a utils module that imports ``eden.options`` at module scope is a
    cycle as soon as it is the first of the three to be imported.
    This module's contract is exactly that: no eden imports, reachable from either side.

    Parameters
    ----------
    func : Callable
        The function term whose signature defines the legal keys.
    params : Mapping | None
        The ``params`` dict to check; ``None`` and ``{}`` are both legal and yield no keys.

    Returns
    -------
    set[str]
        The offending keys, empty when there are none.
    """
    signature_params = inspect.signature(func).parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in signature_params.values()):
        return set()
    return set(params or {}) - set(signature_params)


def invalid_params_message(term_name: str, invalid_keys: Collection[str], accepted: Iterable[ParamInfo]) -> str:
    """The one wording for "these ``params`` keys are not parameters of this term".

    Shared by the construction-time, ``configure()``-time and build-time checks so an author who
    hits it twice reads the same sentence, and so the accepted set is never omitted by one of them.
    """
    return (
        f"The following params keys are invalid: {sorted(invalid_keys)} for function term "
        f"'{term_name}', accepted params: {render_params(accepted)}"
    )


def unknown_parameter_names(extras: Iterable[str], configurable: type, *, tolerated: Collection[str] = ()) -> set[str]:
    """The names in ``extras`` (an options object's extra fields) that ``configurable`` takes no parameter for.

    The one check behind every options class that carries a configurable's parameters as extras (term options,
    camera overlay options): a parameter is a class parameter (``get_parameter_names``) or an options-only one
    (``_extra_option_params``). The options' own metadata (``_option_*``) always passes, and ``tolerated`` names what
    else a caller's options carry that is not a parameter (term options' ``class_name``).
    """
    allowed = set(configurable.get_parameter_names()) | set(getattr(configurable, "_extra_option_params", ()))
    return {key for key in extras if not key.startswith("_option_")} - allowed - set(tolerated)


def invalid_parameters_message(
    subject: str,
    invalid_keys: Collection[str],
    accepted: Iterable[ParamInfo],
    retired: Mapping[str, str],
) -> str:
    """The one wording for "these are not parameters of ``subject``" — a class term or configurable.

    The counterpart of :func:`invalid_params_message` for top-level parameters, shared by the same
    three checks. A key in ``retired`` (a class's ``_retired_params``) also gets that class's
    migration note: a retired name is still invalid, but a config saved before it was retired should
    fail with directions, not only with the list of what is accepted now.
    """
    message = (
        f"The following parameters are invalid: {sorted(invalid_keys)} for {subject}, "
        f"allowed parameters: {render_params(accepted)}"
    )
    retired_keys_by_note: dict[str, list[str]] = {}
    for key in sorted(invalid_keys):
        if key in retired:
            retired_keys_by_note.setdefault(retired[key], []).append(key)
    notes = [f"Retired {keys}: {note}" for note, keys in retired_keys_by_note.items()]
    return ". ".join([message, *notes])


_UNANNOTATED = object()


@lru_cache(maxsize=None)
def _parameter_validator(configurable: type, name: str) -> tuple[Any, TypeAdapter] | None:
    """The declared type of ``configurable``'s parameter ``name`` and the validator for it, or ``None`` when the
    parameter carries no annotation.

    Only this one annotation is resolved, in the module of the class that declares it (the subclass-most one, as
    :meth:`~eden.utils.common.ConfigurableMixin.get_parameter_specs` reads it). ``typing.get_type_hints`` would resolve
    every annotation of the whole hierarchy, so one that names a type imported only under ``TYPE_CHECKING`` (a
    ``_camera: Camera`` attribute, not a parameter) would break every parameter's validation.

    Raises
    ------
    TypeError
        When the annotation cannot be resolved or validated, naming the parameter, where it is declared and the fix.
    """
    for owner in configurable.__mro__:
        annotation = (owner.__dict__.get("__annotations__") or {}).get(name, _UNANNOTATED)
        if annotation is not _UNANNOTATED:
            break
    else:
        return None
    declared = f"{configurable.__name__}'s parameter `{name}: {render_annotation(annotation)}` (declared on {owner.__module__}.{owner.__qualname__})"
    try:
        if isinstance(annotation, str):
            # The module's names first, then the class's, as ``typing.get_type_hints`` looks them up (eval reads
            # locals before globals): a class attribute cannot shadow a type the annotation names. A class whose
            # module is not importable (built by ``type()`` or ``exec``) resolves against its own names only.
            module = sys.modules.get(owner.__module__)
            annotation = eval(annotation, dict(vars(owner)), vars(module) if module else {})  # noqa: S307 - the class's own annotation
        adapter = TypeAdapter(annotation)
        if not adapter.pydantic_complete:  # a quoted name nested in the type defers its error to the first validation
            adapter.rebuild(raise_errors=True)
        return annotation, adapter
    except NameError as e:  # pydantic's PydanticUndefinedAnnotation, for a quoted name nested in the type, is one too
        raise TypeError(
            f"{declared} names a type that does not exist at runtime ({type(e).__name__}: {str(e).strip().splitlines()[0]}), so its values cannot be "
            "checked. A parameter's type has to be importable when the class is used: import it at the module's top "
            "level, not under `if TYPE_CHECKING:`."
        ) from None
    except Exception as e:
        # Pydantic's first sentence only: the rest advises ``arbitrary_types_allowed``, a model config overlays
        # have not.
        reason = str(e).strip().split(". ")[0].splitlines()[0] if str(e).strip() else ""
        raise TypeError(
            f"{declared} cannot be validated ({type(e).__name__}: {reason}). A parameter is written to and read "
            "back from config.json, so its type has to be one a pydantic options field could declare: a number, a "
            "string, a bool, a tuple or list of those, a Literal, or None with one of them."
        ) from None


def validated_parameter_values(configurable: type, values: Mapping[str, Any], *, subject: str) -> dict[str, Any]:
    """``values`` with each of ``configurable``'s parameters validated against its declared type and converted the way
    pydantic converts a declared options field: a list a JSON file read back becomes the declared tuple, ``"0.5"``
    becomes ``0.5``.

    For options that carry a configurable's parameters as extras, which pydantic leaves exactly as written. A key that
    is not a parameter passes through untouched; :func:`unknown_parameter_names` is what refuses those.

    Raises
    ------
    ValueError
        For a value the declared type does not admit (``None`` or a bool for a number, a string for a list), naming
        ``subject``, the parameter, its type and its default.
    TypeError
        For a parameter whose annotation cannot be resolved (a type imported only for type checking) or validated.
    """
    params = set(configurable.get_parameter_names())
    validated = dict(values)
    for key, value in values.items():
        if key not in params:
            continue
        validator = _parameter_validator(configurable, key)
        if validator is None:
            continue
        annotation, adapter = validator
        try:
            # Lax, as an options field converts (``"0.5"`` to 0.5, a list to a tuple), except that a bool counts as a
            # number only where the type admits a bool: ``shadow_strength: yes`` in a YAML file is a mistake, not 1.0.
            validated[key] = adapter.validate_python(value, strict=True if isinstance(value, bool) else None)
        except ValidationError as e:
            raise ValueError(
                f"{subject} needs {key} as {render_annotation(annotation)}; got {value!r} ({type(value).__name__}): "
                f"{e.errors()[0]['msg']}. E.g. {key}={render_default(getattr(configurable, key))} (the default)."
            ) from None
    return validated
