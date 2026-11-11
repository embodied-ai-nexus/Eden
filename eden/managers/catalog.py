"""A catalog of every registered term: the parameters it accepts, their types, and their defaults.

Without it, a term's accepted parameter set exists in exactly one place — the function signature or
class fields in the source — so the only way to learn that ``dofs_vel_limits`` takes ``entity_name``
and ``max_vel`` is to open ``eden/managers/terms/rewards/common.py``. Measured on autonomous coding
agents without a catalog, a capable one paid for that with a median 29-62 reads of Eden's installed
source per task against 2-3 documentation reads; a weaker one skipped the reading and guessed until
its wall clock ran out.

Nothing here is new information: the registries already enumerate the terms and
``get_parameter_specs`` / ``get_function_parameter_specs`` already extract the parameters. This
module emits them, for three consumers:

* ``eden terms`` (:mod:`eden.utils.cli.terms`) — in the terminal, ``--json`` for tooling.
* two generated pages, both written by ``python -m eden.managers.catalog`` (:func:`main`), checked in,
  and verified current by ``tests/resources/test_term_catalog.py``:

  - ``eden/resources/term_catalog.md`` (:func:`render_index`) ships *inside the package*, because an
    agent working against an installed Eden has ``eden/`` but not ``docs/``. It lists every
    term by name, form, defining module and summary, and **no parameters**: an agent reads a page it
    finds whole and keeps it in context for the rest of its session, so this page is sized to be
    read whole. A term's parameters are one ``eden terms --grep <name>`` away.
  - ``docs/api/terms/catalog.md`` (:func:`render_markdown`) is the full catalog, parameters
    included, for the docs site. It is what ``eden terms --markdown`` prints.
* the same ``ParamInfo`` rendering the ``configure()`` errors use (:mod:`eden.utils.params`), so the
  catalog and the error that fires when you ignore it describe a term identically.

:data:`KINDS` is a curated in-tree table rather than a discovered one — the set of term registries is
closed and each entry carries wiring (the import alias, the wrapper class) that no registry knows
about. ``tests/managers/test_catalog.py`` asserts it covers every :class:`~eden.utils.registry.TermRegistry`
in the tree, so adding one and forgetting the catalog fails a test instead of quietly shipping a term
nobody can find.
"""

from __future__ import annotations

import importlib
import inspect
import re
import sys
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from functools import partial
from itertools import groupby
from pathlib import Path
from typing import Any

from eden.utils.params import ParamInfo, render_params
from eden.utils.registry import TermRegistry

REPO_ROOT = Path(__file__).resolve().parents[2]

#: The generated index page (:func:`render_index`), shipped *inside the package* — in the
#: ``eden/resources/`` data directory — so it reaches an installed Eden (``pip install`` carries
#: ``eden/`` but not ``docs/``). Resolved package-relative (like :data:`REPO_ROOT` above), so it
#: holds both in a repo checkout and in an installed package; ``pyproject``'s
#: ``[tool.setuptools.package-data]`` ships it and ``tests/resources/test_term_catalog.py`` guards that. Written
#: by ``python -m eden.managers.catalog``; a stale copy fails ``tests/resources/test_term_catalog.py``.
CATALOG_PAGE = Path(__file__).resolve().parents[1] / "resources" / "term_catalog.md"

#: The full catalog (:func:`render_markdown`, parameters included) for the docs site. Repo-only: it
#: is not in the wheel, which carries :data:`CATALOG_PAGE` instead.
DOCS_CATALOG_PAGE = REPO_ROOT / "docs" / "api" / "terms" / "catalog.md"

#: Upper bound on :data:`CATALOG_PAGE`, in bytes. The page is read whole by the agents that find it
#: and then re-sent on every later turn of their session, so its size is paid many times over; the
#: full catalog (:data:`DOCS_CATALOG_PAGE`) is over twice this and grows with every term.
#: ``tests/resources/test_term_catalog.py`` fails above this.
INDEX_PAGE_BUDGET = 32_000


@dataclass(frozen=True)
class TermKind:
    """One term registry, plus how a config reaches the terms inside it.

    ``module``/``registry_attr`` are resolved lazily (:meth:`registry`) so importing this module
    costs only what the caller already imported — ``eden.managers`` does not pull the motion manager
    or the modifiers, and listing rewards should not either.
    """

    #: Kind name as ``eden terms --kind`` spells it, and the catalog's section title.
    name: str
    module: str
    registry_attr: str

    #: ``from eden import <terms_module>`` — where a function term of this kind is imported from
    #: in a config. ``None`` for the kinds with no such alias (motion, modifiers).
    terms_module: str | None = None

    #: The ``eden.terms`` alias whose ``.configure(func=...)`` wires a function term of this kind
    #: in. ``None`` for the class-only kinds, which have no function-wrapper class at all.
    wrapper: str | None = None

    def registry(self) -> TermRegistry:
        """This kind's registry, which fills itself on the first read.

        Which modules have to be imported for a registry to hold its terms is the registry's own
        business (``populated_by``, see :meth:`~eden.utils.registry.Registry._populate`), so it is
        not declared here a second time.
        """
        return getattr(importlib.import_module(self.module), self.registry_attr)


#: Every term registry in the tree, manager kinds first in the order the docs nav lists them.
KINDS: tuple[TermKind, ...] = (
    TermKind("action", "eden.managers", "ACTION_TERM_REGISTRY", "actions"),
    TermKind("observation", "eden.managers", "OBSERVATION_TERM_REGISTRY", "observations", "ObsTerm"),
    TermKind("reward", "eden.managers", "REWARD_TERM_REGISTRY", "rewards", "RewardTerm"),
    TermKind("termination", "eden.managers", "TERMINATION_TERM_REGISTRY", "terminations", "DoneTerm"),
    TermKind("command", "eden.managers", "COMMAND_TERM_REGISTRY", "commands"),
    TermKind("event", "eden.managers", "EVENT_TERM_REGISTRY", "events", "EventTerm"),
    TermKind("curriculum", "eden.managers", "CURRICULUM_TERM_REGISTRY", "curricula", "CurrTerm"),
    TermKind("metric", "eden.managers", "METRIC_TERM_REGISTRY", "metrics", "MetricTerm"),
    TermKind("recorder", "eden.managers", "RECORDER_TERM_REGISTRY", "recorders"),
    TermKind("motion", "eden.managers.motion_manager", "MOTION_TERM_REGISTRY"),
    TermKind("noise_model", "eden.managers.modifiers.base", "NOISE_MODEL_REGISTRY"),
    TermKind("observation_modifier", "eden.managers.modifiers.base", "OBSERVATION_MODIFIER_REGISTRY"),
    TermKind("action_modifier", "eden.managers.modifiers.base", "ACTION_MODIFIER_REGISTRY"),
)

KIND_NAMES: tuple[str, ...] = tuple(kind.name for kind in KINDS)


@dataclass(frozen=True)
class TermEntry:
    """One registered term, described the way an author has to write it."""

    #: The registry key — what a config's ``name`` field holds and what every error message
    #: quotes. For a function term this is the function's name; for a class term, the
    #: snake-cased class name.
    name: str
    kind: str

    #: ``"function"`` or ``"class"``. It decides how the term is configured: ``params={...}``
    #: through the kind's wrapper, versus ``<ClassName>.configure(field=...)`` directly.
    form: str

    #: The Python name to import — the function's, or the class's (which is *not* :attr:`name`).
    qualname: str
    module: str

    #: Repo-relative ``path:line`` of the definition, for when the catalog is not enough.
    source: str

    #: First line of the docstring, or ``""`` when the term has none.
    summary: str

    #: The term's own parameters: a function term's ``params={...}`` keys, or a class term's
    #: fields minus the ones every term of the kind carries (those are reported once per kind,
    #: by :func:`kind_shared_specs`).
    params: tuple[ParamInfo, ...]

    #: ``EnvOptions`` flags the term needs set to ``True``
    #: (:attr:`~eden.managers.base.ManagerTermBase._requires_env_options`) — a scene-wide
    #: precondition no parameter of the term can satisfy, so a config that reads only the
    #: parameter column configures the term correctly and still fails at build. Empty for
    #: every term that has none, which is nearly all of them.
    requires_env_options: tuple[str, ...] = ()

    @property
    def haystack(self) -> str:
        """Everything ``eden terms --grep`` searches, as one lowercase line.

        Name, kind, class/function name, defining module, summary *and* parameter names and types
        — because "which term takes ``max_vel``?" and "is there an observation for base height?"
        are the two questions that send a reader to the full catalog, and only one of them is
        answerable from term names alone.
        """
        params = " ".join(f"{param.name} {param.annotation}" for param in self.params)
        return " ".join((self.name, self.kind, self.form, self.qualname, self.module, self.summary, params)).lower()

    def to_dict(self) -> dict[str, Any]:
        """A JSON-serializable form, with each param expanded — what ``eden terms --json`` emits."""
        return {
            "name": self.name,
            "kind": self.kind,
            "form": self.form,
            "qualname": self.qualname,
            "module": self.module,
            "source": self.source,
            "summary": self.summary,
            "params": [
                {
                    "name": param.name,
                    "type": param.annotation,
                    "default": param.default,
                    "required": param.required,
                }
                for param in self.params
            ],
            "required": [param.name for param in self.params if param.required],
            "requires_env_options": list(self.requires_env_options),
        }


#: ``:class:`~a.b.C``` / ``:func:`a.b.c``` — Sphinx roles, which a markdown table renders verbatim.
_SPHINX_ROLE = re.compile(r":[a-z:]+:`~?([^`]+)`")


def _first_line(doc: str) -> str:
    """The summary line of ``doc``, with Sphinx cross-reference roles reduced to their target.

    A role's *tilde* form already means "show the last component", so honouring it is not a loss:
    ``:class:`~eden.managers.metric_manager.MetricTerm``` becomes ``` `MetricTerm` ```.
    """
    line = doc.strip().split("\n", 1)[0].strip()
    return _SPHINX_ROLE.sub(lambda m: f"`{m.group(1).rsplit('.', 1)[-1]}`", line)


def _summary(obj: Any, stop_at: type | None = None) -> str:
    """The first line of ``obj``'s own docstring — the one-line description, or ``""``.

    Deliberately not ``inspect.getdoc``: that inherits, and an undocumented term would then be
    described by its base class — all three noise models reading "Base class for noise models
    applied to observation terms" is worse than an empty cell, which at least shows the gap.

    The MRO is walked only across the synthesized-defaults layers ``TermRegistry._with_defaults``
    inserts (a subclass sharing the original's ``__qualname__`` but not its ``__dict__``); the walk
    stops at the first base with a *different* qualname, so a genuinely distinct base's docstring is
    never reported as this term's.
    """
    if not inspect.isclass(obj):
        return _first_line(obj.__doc__ or "")
    for base in obj.__mro__:
        if base is stop_at or base is object or base.__qualname__ != obj.__qualname__:
            break
        doc = base.__dict__.get("__doc__")
        if doc:
            return _first_line(doc)
    return ""


def _source(obj: Any) -> str:
    """Repo-relative ``path:line`` for ``obj``'s definition, or ``""`` if it cannot be located.

    A class term registered with defaults is a synthesized subclass (``TermRegistry._with_defaults``)
    that shares the original's ``__module__`` and ``__qualname__``, so the *file* is always right;
    only the line number can be missing, and it degrades to the file alone rather than raising.
    """
    try:
        file = inspect.getsourcefile(obj)
    except TypeError:
        return ""
    if file is None:
        return ""
    try:
        path = Path(file).resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        # Installed outside the repo (site-packages): the module path is the useful anchor.
        path = Path(file).name
    try:
        return f"{path}:{inspect.getsourcelines(obj)[1]}"
    except (OSError, TypeError):
        return path


def kind_shared_specs(kind: TermKind) -> tuple[ParamInfo, ...]:
    """The fields *every* term of ``kind`` accepts, alongside whatever the term itself declares.

    A reward term's ``weight``, an event term's ``mode``, an observation term's ``noise``/``scale``
    — declared on the kind's base term class (and its options), not on any one term. Reported once
    per kind so the reward rows do not each repeat ``weight``.
    """
    registry = kind.registry()
    base = registry.func_wrapper_class or registry.term_class
    return tuple(spec for spec in base.get_configure_specs() if spec.name != "params")


def _entry(kind: TermKind, key: str, registered: Any, shared: frozenset[str], term_base: type) -> TermEntry:
    if isinstance(registered, partial):
        # Function term: the registry stores ``partial(wrapper_cls, func=obj)``.
        wrapper_cls, func = registered.func, registered.keywords["func"]
        return TermEntry(
            name=key,
            kind=kind.name,
            form="function",
            qualname=func.__name__,
            module=func.__module__,
            source=_source(func),
            summary=_summary(func),
            params=wrapper_cls.get_function_parameter_specs(func),
            # Read off the wrapper class: a function has no class attributes of its own, so a
            # function term declares this through its registration
            # (``@REGISTRY.register(_requires_env_options=(...))``), which synthesizes the
            # per-registration wrapper subclass this reads. A function term registered without it
            # shares the kind's plain wrapper, where it is the empty base default.
            requires_env_options=tuple(wrapper_cls._requires_env_options),
        )
    return TermEntry(
        name=key,
        kind=kind.name,
        form="class",
        qualname=registered.__name__,
        module=registered.__module__,
        source=_source(registered),
        summary=_summary(registered, stop_at=term_base),
        params=tuple(spec for spec in registered.get_configure_specs() if spec.name not in shared),
        requires_env_options=tuple(registered._requires_env_options),
    )


def iter_kinds(kind: str | None = None) -> Iterator[TermKind]:
    """The requested kind, or every kind in :data:`KINDS` order.

    Raises
    ------
    ValueError
        If ``kind`` is not a known kind name — with the known ones listed, since a caller who
        guessed wrong (``rewards`` for ``reward``) needs the vocabulary, not just a refusal.
    """
    if kind is None:
        yield from KINDS
        return
    for entry in KINDS:
        if entry.name == kind:
            yield entry
            return
    raise ValueError(f"unknown term kind '{kind}'. Available kinds: {', '.join(KIND_NAMES)}")


def catalog(kind: str | None = None) -> tuple[TermEntry, ...]:
    """Every registered term, name-sorted within each kind.

    Sorted rather than left in registration (import) order: the catalog is looked up by name and
    diffed between revisions, and import order is neither stable nor meaningful.
    """
    entries: list[TermEntry] = []
    for term_kind in iter_kinds(kind):
        shared = frozenset(spec.name for spec in kind_shared_specs(term_kind))
        registry = term_kind.registry()
        entries.extend(
            _entry(term_kind, key, registry.get(key), shared, registry.term_class) for key in sorted(registry.keys())
        )
    return tuple(entries)


def builtin_catalog(kind: str | None = None) -> tuple[TermEntry, ...]:
    """:func:`catalog` restricted to the terms Eden itself ships — what the generated page holds.

    A term registry is a global, open table: an out-of-tree term registers into the same one, and
    ``eden terms`` listing it is the point. The *checked-in* page must not depend on which of those
    happen to be imported in the process that generated it, or it would read as stale as often as it
    is right (the test suite alone registers a dozen throwaway terms).

    "Shipped" is the ``eden.managers`` namespace — the term modules each registry names in its
    ``populated_by``. ``eden.tasks.*`` / ``eden.extensions.*`` terms register when their task
    module is imported and would otherwise leak in by import order — they match a plain
    ``startswith("eden.")`` but not the ``eden.managers`` prefix tested here.
    """
    return tuple(
        entry for entry in catalog(kind) if entry.module == "eden.managers" or entry.module.startswith("eden.managers.")
    )


def search(entries: Sequence[TermEntry], pattern: str) -> tuple[TermEntry, ...]:
    """``entries`` whose :attr:`TermEntry.haystack` matches ``pattern`` — what ``--grep`` filters by.

    Case-insensitive, and a regular expression rather than a plain substring so alternation
    (``height|altitude``) answers "is there a term for X, under whatever name Eden gave it" in one
    call. A plain substring is a valid regex, so the simple spelling keeps working.

    Parameters
    ----------
    entries : Sequence[TermEntry]
        The catalog (or one kind of it) to filter.
    pattern : str
        A Python regular expression, matched case-insensitively against each entry's haystack.

    Returns
    -------
    tuple[TermEntry, ...]
        The matching entries, in the order given.

    Raises
    ------
    ValueError
        If ``pattern`` is not a valid regular expression — naming the offending pattern and what
        ``re`` objected to, since the alternative is a bare ``re.error`` several frames in.
    """
    try:
        matches = re.compile(pattern, re.IGNORECASE).search
    except re.error as exc:
        raise ValueError(
            f"--grep {pattern!r} is not a valid regular expression: {exc}. Escape the metacharacter "
            f"(e.g. 'dofs\\[0\\]') or pass a plain substring."
        ) from exc
    return tuple(entry for entry in entries if matches(entry.haystack))


def _configure_recipes(kind: TermKind, entries: Sequence[TermEntry]) -> list[str]:
    """How a term of ``kind`` is written into a config — one line per form the kind actually has.

    Driven by ``entries`` rather than by the kind's capabilities: a kind whose registry happens to
    hold only class terms should not be told how to wire up a function term it has none of.
    """
    forms = {entry.form for entry in entries}
    recipes = []
    if "function" in forms and kind.wrapper is not None and kind.terms_module is not None:
        recipes.append(
            f"Function terms: `{kind.wrapper}.configure(func={kind.terms_module}.<term>, params={{...}})` "
            f"— `from eden import {kind.terms_module}`, `from eden.terms import {kind.wrapper}`"
        )
    if "class" in forms:
        recipes.append(
            "Class terms: `<ClassName>.configure(<field>=...)` — import it from the module its section names"
        )
    return recipes


def _by_kind(entries: Sequence[TermEntry]) -> Iterator[tuple[TermKind, list[TermEntry]]]:
    """``entries`` split into their kinds' runs, each paired with its :class:`TermKind`."""
    by_kind = {kind.name: kind for kind in KINDS}
    for kind_name, kind_entries in groupby(entries, key=lambda entry: entry.kind):
        yield by_kind[kind_name], list(kind_entries)


def _by_module(kind_entries: Sequence[TermEntry]) -> Iterator[tuple[str, list[TermEntry]]]:
    """One kind's entries grouped by defining module, modules and the terms inside each in name order.

    Re-sorted by module: ``catalog()`` orders a kind by term name, which would fragment a module into
    several sections.
    """
    by_module = sorted(kind_entries, key=lambda entry: (entry.module, entry.name))
    for module, module_entries in groupby(by_module, key=lambda entry: entry.module):
        yield module, list(module_entries)


def render_text(entries: Sequence[TermEntry], grep: str | None = None) -> str:
    """``entries`` as indented plain text, grouped by kind — what ``eden terms`` prints.

    Not a ``PrettyTable`` like ``eden list``: a term's parameter list is the long column here, and
    boxed cells either wrap it into unreadable slivers or blow past the terminal width.

    Parameters
    ----------
    entries : Sequence[TermEntry]
        The entries to render, in catalog order (grouped by kind).
    grep : str | None
        The ``--grep`` pattern ``entries`` was filtered by, when it was. Named in the per-kind
        header so a count of 3 reads as "3 matched", not "Eden ships 3 reward terms".
    """
    lines: list[str] = []
    matching = f" matching '{grep}'" if grep is not None else ""
    for kind, kind_entries in _by_kind(entries):
        lines.append(f"\n{kind.name} terms ({len(kind_entries)}{matching})")
        shared = kind_shared_specs(kind)
        if shared:
            lines.append(f"  every {kind.name} term also accepts: {render_params(shared)}")
        lines.append("")
        for entry in kind_entries:
            form = "fn" if entry.form == "function" else f"class {entry.qualname}"
            lines.append(f"  {entry.name}  [{form}]  {entry.source}")
            lines.append(f"    params:  {render_params(entry.params)}")
            if entry.requires_env_options:
                lines.append(f"    requires:  {_render_requires(entry)}")
            if entry.summary:
                lines.append(f"    {entry.summary}")
    return "\n".join(lines).strip() + "\n"


def _render_requires(entry: TermEntry) -> str:
    """A term's env-options preconditions, written the way a config states them."""
    return ", ".join(f"EnvOptions({flag}=True)" for flag in entry.requires_env_options)


def _markdown_requires(entry: TermEntry) -> str:
    """A term's env-options preconditions for a table cell, or ``—`` when it has none."""
    rendered = _render_requires(entry)
    return f"`{_markdown_cell(rendered)}`" if rendered else "—"


def _markdown_cell(text: str) -> str:
    """``text`` as a markdown table cell: pipes escaped so a ``str | None`` type cannot split it."""
    return text.replace("|", "\\|")


def _markdown_params(specs: Sequence[ParamInfo]) -> str:
    """Parameters for a table cell, required ones bold — the column an author scans for."""
    if not specs:
        return "—"
    return ", ".join(
        f"**`{_markdown_cell(str(spec))}`**" if spec.required else f"`{_markdown_cell(str(spec))}`" for spec in specs
    )


def render_markdown(entries: Sequence[TermEntry], grep: str | None = None) -> str:
    """``entries`` as the full catalog page, parameters included: ``docs/api/terms/catalog.md``.

    What ``eden terms --markdown`` prints. The page that ships in the wheel is :func:`render_index`.

    One row per term, so a grep for a parameter name lands on a line that also carries the term
    name, its form and its summary — the shape that answers "which term takes ``max_vel``?" in one
    hit. Rows are grouped by defining module, which doubles as the source pointer and keeps the
    ``Source`` column (and its width) off every row.

    Parameters
    ----------
    entries : Sequence[TermEntry]
        The entries to render, in catalog order (grouped by kind).
    grep : str | None
        The ``--grep`` pattern ``entries`` was filtered by, when it was. The intro otherwise
        claims to be the complete list, which a filtered page is not.
    """
    # Derive the "complete list" claim from the entries rather than a flag, so the header can never
    # claim more than it shows: a single-kind view (``eden terms --kind reward --markdown``) says so,
    # while the full catalog — which spans every kind — keeps its original wording byte-for-byte
    # (the checked-in ``catalog.md`` parity test pins it).
    kinds_present = {entry.kind for entry in entries}
    only_kind = next(iter(kinds_present)) if len(kinds_present) == 1 else None
    if grep is not None:
        scope = f"{only_kind} term" if only_kind else "term"
        intro = (
            f"Every {scope} registered in Eden whose name, parameters or summary match `{grep}` "
            f"({len(entries)} of them) — a filtered view, not the whole catalog (`eden terms --markdown` "
            "emits every kind). The guide's [Terms & Registries](../../guides/terms.md) page explains how "
            "to write and wire one up."
        )
    elif only_kind is None:
        intro = (
            f"Every term registered in Eden ({len(entries)} of them) with the parameters it accepts, generated "
            "from the registries themselves. This is the complete list; the guide's "
            "[Terms & Registries](../../guides/terms.md) page explains how to write and wire one up."
        )
    else:
        intro = (
            f"Every {only_kind} term registered in Eden ({len(entries)} of them) with the parameters it accepts, "
            f"generated from the registries themselves — the {only_kind} section only, not the whole catalog "
            "(`eden terms --markdown` emits every kind). The guide's "
            "[Terms & Registries](../../guides/terms.md) page explains how to write and wire one up."
        )
    lines = [
        "<!-- Generated by `python -m eden.managers.catalog`. Do not edit by hand. -->",
        "",
        "# Term Catalog",
        "",
        intro,
        "",
        "- **Bold** parameters are required — omit one and `configure()` raises before the env is built.",
        (
            "- A term's parameters are the keys of its `params={...}` dict (function terms) or its "
            "`configure()` fields (class terms). The fields *every* term of a kind carries are listed "
            "once, under that kind."
        ),
        (
            "- A **Requires** column appears in the tables that have one: an `EnvOptions` flag the term "
            "needs set, which no parameter of the term can supply. The env build raises without it."
        ),
        (
            "- `eden terms` prints the same catalog in a terminal, `eden terms --kind reward` one section "
            "of it, `eden terms --grep height` only the terms whose name, parameters or summary match, and "
            "`eden terms --json` emits it as JSON."
        ),
        "",
    ]
    for kind, kind_entries in _by_kind(entries):
        lines += [f"## {kind.name} terms", "", f"{len(kind_entries)} registered.", ""]
        for recipe in _configure_recipes(kind, kind_entries):
            lines.append(f"- {recipe}")
        shared = kind_shared_specs(kind)
        if shared:
            lines.append(f"- Every {kind.name} term also accepts: {_markdown_params(shared)}")
        lines.append("")
        for module, module_entries in _by_module(kind_entries):
            # The ``Requires`` column only appears in the tables that have one. A fifth column of
            # "—" on all ~200 rows would cost every reader width to tell nearly all of them
            # nothing; the header is per-module already, so a module that has preconditions can
            # carry the column without the rest of the catalog paying for it.
            has_requires = any(entry.requires_env_options for entry in module_entries)
            header = "| Term | Form | Parameters |" + (" Requires |" if has_requires else "") + " Summary |"
            lines += [
                f"### `{module}`",
                "",
                header,
                "|---|---|---|---|---|" if has_requires else "|---|---|---|---|",
            ]
            for entry in module_entries:
                form = "function" if entry.form == "function" else f"class `{entry.qualname}`"
                summary = _markdown_cell(entry.summary) or "—"
                requires = f" {_markdown_requires(entry)} |" if has_requires else ""
                lines.append(f"| `{entry.name}` | {form} | {_markdown_params(entry.params)} |{requires} {summary} |")
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_index(entries: Sequence[TermEntry]) -> str:
    """``entries`` as the shipped ``eden/resources/term_catalog.md`` index: every term, no parameters.

    What :func:`render_markdown` carries minus its ``Parameters`` column, which was most of that page:
    grouped by kind and then by defining module, one line per term with its registry key, the class
    to import for a class term, and its summary; each kind keeps its configure recipe and the names of
    the fields all its terms share. Module headings are fully qualified, because a class term's recipe
    says to import it from the module its section names. An agent that finds a catalog in the installed
    package reads it whole and carries it for the rest of its session, so this page answers "is there a
    term for X, and where is it defined?" and leaves "what does it take?" to ``eden terms --grep <name>``.

    Parameters
    ----------
    entries : Sequence[TermEntry]
        The entries to render, in catalog order (grouped by kind).
    """
    lines = [
        "<!-- Generated by `python -m eden.managers.catalog`. Do not edit by hand. -->",
        "",
        "# Term Catalog",
        "",
        (
            f"Every term registered in Eden ({len(entries)} of them): registry key, class name for a class "
            "term, and summary, under the module that defines it. Parameters are not listed — "
            "`eden terms --grep <name>` (or `python -m eden terms --grep <name>`) prints a term's "
            "parameters, types and defaults, and marks the required ones."
        ),
        "",
    ]
    for kind, kind_entries in _by_kind(entries):
        lines += [f"## {kind.name} terms ({len(kind_entries)})", ""]
        for recipe in _configure_recipes(kind, kind_entries):
            lines.append(f"- {recipe}")
        shared = kind_shared_specs(kind)
        if shared:
            # Names only, like the rest of the page: the types and defaults are parameters too.
            lines.append(f"- Every {kind.name} term also accepts: {', '.join(f'`{p.name}`' for p in shared)}")
        lines.append("")
        for module, module_entries in _by_module(kind_entries):
            lines.append(f"`{module}`:")
            for entry in module_entries:
                cls = f" (`{entry.qualname}`)" if entry.form != "function" else ""
                # Kept inline: a precondition no parameter can supply, so leaving it out would let a
                # config written from this page fail at build.
                # A bullet list, not a table: no cell escaping, so a summary's `|x|` reads as written.
                requires = f" Requires `{_render_requires(entry)}`." if entry.requires_env_options else ""
                lines.append(f"- `{entry.name}`{cls} — {entry.summary or '—'}{requires}")
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    """Regenerate both checked-in catalog pages from the registries (``--check``: report, don't write).

    Run it from a repo checkout after adding, renaming or re-parameterizing a term;
    ``tests/resources/test_term_catalog.py`` fails while either page is stale::

        uv run python -m eden.managers.catalog          # write both pages
        uv run python -m eden.managers.catalog --check  # exit 1 if either would change
    """
    import argparse

    parser = argparse.ArgumentParser(prog="python -m eden.managers.catalog", description=main.__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="report staleness instead of writing")
    args = parser.parse_args(argv)

    # REPO_ROOT is `parents[2]` of this file, which in an installed eden is the site-packages parent: the
    # write below would `mkdir -p` a `docs/` there and succeed silently.
    if not ((REPO_ROOT / "pyproject.toml").is_file() and (REPO_ROOT / "docs").is_dir()):
        print(
            f"python -m eden.managers.catalog regenerates pages checked into an Eden repo checkout, but the eden "
            f"it imported ({Path(__file__).resolve()}) is not in one: '{REPO_ROOT}' lacks pyproject.toml or "
            "docs/. Run it from a clone (`git clone https://github.com/embodied-ai-nexus/Eden && cd Eden && "
            "uv run python -m eden.managers.catalog`). An installed eden needs no regeneration: it ships a current "
            "eden/resources/term_catalog.md, and `eden terms --markdown` prints the full catalog.",
            file=sys.stderr,
        )
        return 2

    entries = builtin_catalog()
    pages = ((CATALOG_PAGE, render_index(entries)), (DOCS_CATALOG_PAGE, render_markdown(entries)))
    stale = [(page, rendered) for page, rendered in pages if not page.exists() or page.read_text() != rendered]

    if args.check:
        for page, _ in stale:
            print(f"{page.relative_to(REPO_ROOT)} is stale — run: uv run python -m eden.managers.catalog")
        if not stale:
            print(f"both catalog pages are up to date ({len(entries)} terms)")
        return 1 if stale else 0

    for page, rendered in stale:
        page.parent.mkdir(parents=True, exist_ok=True)
        page.write_text(rendered)
        print(f"wrote {page.relative_to(REPO_ROOT)} ({len(rendered.encode()):,} bytes)")
    if not stale:
        print(f"both catalog pages already up to date ({len(entries)} terms)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
