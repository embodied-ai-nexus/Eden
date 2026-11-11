"""The command-line convention every example under ``examples/`` shares.

The convention itself is written down in ``examples/AGENTS.md``; this is the machine
check that keeps it true. It enforces six things:

1. **Declarations.** Each ``add_argument`` uses a reserved short flag bound to its one
   canonical long name, short-first, dash-cased, and carries a ``help=``.
2. **Documentation.** Every short flag in a command line quoted in a docstring or a
   README is one the referenced script actually accepts.
3. **Headless safety.** A viewer-first script — one that takes ``--no-vis``, so the
   viewer is on unless the user opts out — gates ``show_viewer`` on ``en.is_smoke()``.
4. **Defaults that exist.** Every statically identifiable ``default=`` naming a bundled
   file inside the repo names one that is actually there.
5. **Handoffs that resolve.** A script that chains one subcommand into another names
   arguments the way its own parser does.
6. **Tasks built through the registry.** A registered task is constructed with
   ``TASK_REGISTRY.build(name)`` — never ``TASK_REGISTRY.get(name)()``, and never by
   importing the config class and calling it.

Check 2 is the one that matters in practice: renaming a flag in the parser and leaving
the docstring behind produces a copy-pasteable command that dies on
``unrecognized arguments``, which nothing else in the tree would catch. Check 3 covers the
other end: the smoke harness invokes an example with no arguments, so a viewer left on by
default opens one and the run dies on ``No display detected``. Every other example
is headless-safe only because its viewer happens to default off; this is what tells the
two apart. Check 4 is what a tree reorganization breaks: a default pointing at a moved
asset fails only when the example runs, which no other test does. Check 5 guards the
end of a training run, the most expensive place to discover a typo: the train-then-render
chain runs only after ``runner.learn`` returns. Check 6 is the one whose failure is
furthest from its cause: ``get`` returns the bare config class, as does importing it, so
calling either runs none of the task's TaskMods, and a task whose robot binding lives in a
mod comes back with its link names still at their empty sentinels — a config
that constructs and serializes fine and dies in ``env.build()`` with a lookup error that
never mentions modifiers.

Single source of truth for the contract; ``tests/utils/test_example_flags.py`` runs the same six
functions over every example.
"""

from __future__ import annotations

import ast
import difflib
import os
import pathlib
import re

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
EXAMPLES = REPO_ROOT / "examples"

# The closed set. `-o` names the output destination, spelled for a directory or a single
# file; a script exposes at most one of the two.
RESERVED: dict[str, tuple[str, ...]] = {
    "-v": ("--vis",),
    "-c": ("--cpu",),
    "-b": ("--num-envs",),
    "-s": ("--steps",),
    "-t": ("--seconds",),
    "-e": ("--exp-name",),
    "-r": ("--record",),
    "-o": ("--output-dir", "--output"),
    "-d": ("--debug-vis",),
}

# `python foo.py --flag ...` / `uv run python path/to/foo.py -v`, capturing script + args.
_COMMAND = re.compile(r"(?P<script>[\w./-]+\.py)(?P<args>(?:\s+[^\s`|]+)*)")
_SHORT = re.compile(r"(?<![\w-])-([a-zA-Z])(?![\w-])")

# Path expressions the resolver understands, spelled either way round: `os.path.join(...)`
# / `os.path.dirname(__file__)` or `Path(__file__).parent / "..."`.
_JOIN = {"os.path.join", "join"}
_DIRNAME = {"os.path.dirname", "dirname"}
_NORMALIZE = {"os.path.abspath", "os.path.normpath", "os.path.realpath", "abspath", "normpath", "realpath"}
_PATH_CTOR = {"Path", "pathlib.Path"}


def _tree(path: pathlib.Path) -> ast.AST | None:
    """Parsed source, or None when the file is unreadable or not valid Python."""
    try:
        return ast.parse(path.read_text())
    except (SyntaxError, OSError):
        return None


def _add_argument_calls(tree: ast.AST):
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument":
            yield node


def _flags(node: ast.Call) -> list[str]:
    return [a.value for a in node.args if isinstance(a, ast.Constant) and isinstance(a.value, str)]


def accepted_shorts(path: pathlib.Path) -> set[str] | None:
    """Every short flag ``path``'s parser accepts, or None if it has no parser."""
    if (tree := _tree(path)) is None:
        return None
    shorts, seen_parser = set(), False
    for node in _add_argument_calls(tree):
        seen_parser = True
        shorts.update(f for f in _flags(node) if re.fullmatch(r"-[a-zA-Z]", f))
    return shorts if seen_parser else None


def check_declarations(path: pathlib.Path, problems: list[str]) -> None:
    if (tree := _tree(path)) is None:
        return
    for node in _add_argument_calls(tree):
        flags = _flags(node)
        longs = [f for f in flags if f.startswith("--")]
        shorts = [f for f in flags if f.startswith("-") and not f.startswith("--")]
        if not longs:
            continue  # positional
        where = f"{path.relative_to(REPO_ROOT)}:{node.lineno}"
        long = longs[0]
        if "_" in long:
            problems.append(f"{where}: {long} is snake_case; long flags use dashes")
        for short in shorts:
            if len(short) != 2:
                problems.append(f"{where}: {short} is a multi-character single-dash flag")
            elif short not in RESERVED:
                problems.append(f"{where}: {short} is not in the reserved short-flag set")
            elif long not in RESERVED[short]:
                expected = " / ".join(RESERVED[short])
                problems.append(f"{where}: {short} is reserved for {expected}, not {long}")
        if shorts and flags[0] != shorts[0]:
            problems.append(f"{where}: {longs[0]} lists its short flag second; short comes first")
        if not any(k.arg == "help" for k in node.keywords):
            problems.append(f"{where}: {long} has no help=")


def check_documented_commands(doc: pathlib.Path, problems: list[str]) -> None:
    """Flag every documented short that the referenced script would reject."""
    try:
        text = doc.read_text()
    except OSError:
        return
    for lineno, line in enumerate(text.splitlines(), 1):
        for match in _COMMAND.finditer(line):
            named = match.group("script")
            # Resolve against the repo root first, then next to the doc that quotes it.
            for candidate in (REPO_ROOT / named, doc.parent / named):
                if candidate.is_file():
                    script = candidate
                    break
            else:
                continue  # unresolvable path; path correctness is the header checker's job
            shorts = accepted_shorts(script)
            if shorts is None:
                continue
            for short in {f"-{m.group(1)}" for m in _SHORT.finditer(match.group("args"))}:
                if short not in shorts:
                    where = f"{doc.relative_to(REPO_ROOT)}:{lineno}"
                    problems.append(f"{where}: {named} does not accept {short}")


def _viewer_on_by_default(tree: ast.AST) -> int | None:
    """Line of the ``add_argument`` that leaves the viewer on, or None if it defaults off."""
    for node in _add_argument_calls(tree):
        if not any(f in ("--vis", "--no-vis") for f in _flags(node)):
            continue
        consts = {k.arg: k.value.value for k in node.keywords if isinstance(k.value, ast.Constant)}
        # --no-vis (store_false), or an explicit default=True: viewer on unless opted out
        if consts.get("action") == "store_false" or consts.get("default") is True:
            return node.lineno
    return None


def check_headless_safe(path: pathlib.Path, problems: list[str]) -> None:
    """Flag a viewer-first script whose ``show_viewer`` isn't gated on ``en.is_smoke()``.

    ``eden examples smoke`` invokes the script with no arguments, so a viewer that is on by
    default opens one under ``EDEN_SMOKE`` and the run dies where there is no display.

    Anchors on ``show_viewer=`` keywords and accepts any expression that mentions
    ``is_smoke`` — a lint, so it errs toward a false negative over a false positive. A
    literal ``show_viewer=False`` passes too: a viewer-first demo may still build a second
    env headless.
    """
    if (tree := _tree(path)) is None:
        return
    lineno = _viewer_on_by_default(tree)
    if lineno is None:
        return
    for kw in ast.walk(tree):
        if not isinstance(kw, ast.keyword) or kw.arg != "show_viewer":
            continue
        source = ast.unparse(kw.value)
        if "is_smoke" in source or source == "False":
            continue
        where = f"{path.relative_to(REPO_ROOT)}:{kw.value.lineno}"
        problems.append(
            f"{where}: show_viewer={source}, but the viewer flag on line {lineno} leaves the "
            f"viewer on by default; gate it on en.is_smoke() so smoke runs stay headless"
        )


def _dotted(node: ast.AST) -> str | None:
    """The dotted name an expression spells (``os.path.join``), or None if it spells none."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    parts.append(node.id)
    return ".".join(reversed(parts))


def _static_path(node: ast.AST, env: dict[str, str]) -> str | None:
    """The path ``node`` denotes, or None when it is not knowable without running the script."""
    if isinstance(node, ast.Constant):
        return node.value if isinstance(node.value, str) else None
    if isinstance(node, ast.Name):
        return env.get(node.id)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):  # Path(...) / "sample" / "clip.npz"
        left, right = _static_path(node.left, env), _static_path(node.right, env)
        return None if left is None or right is None else os.path.join(left, right)
    if isinstance(node, ast.Attribute) and node.attr == "parent":
        base = _static_path(node.value, env)
        return None if base is None else os.path.dirname(base)
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Attribute) and node.func.attr == "resolve":  # Path(__file__).resolve()
            return _static_path(node.func.value, env)
        args = [_static_path(a, env) for a in node.args]
        if any(a is None for a in args) or node.keywords:
            return None
        name = _dotted(node.func)
        if name in _JOIN and args:
            return os.path.join(*args)
        if len(args) == 1:
            if name in _DIRNAME:
                return os.path.dirname(args[0])
            if name in _NORMALIZE or name in _PATH_CTOR:
                return os.path.normpath(args[0])
    return None


def _module_paths(script: pathlib.Path, tree: ast.AST) -> dict[str, str]:
    """Every module-level name bound to a statically knowable path, ``__file__`` seeded."""
    env = {"__file__": str(script)}
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and (value := _static_path(node.value, env)) is not None
        ):
            env[node.targets[0].id] = value
    return env


def _repo_target(target: str) -> pathlib.Path | None:
    """Resolve an unambiguous in-repo target, or None for user-owned/choice strings."""
    target_path = pathlib.Path(os.path.normpath(target))
    if target_path.is_absolute():
        return target_path if target_path.is_relative_to(REPO_ROOT) else None
    if target_path.parts and target_path.parts[0] == "examples":
        return REPO_ROOT / target_path
    return None


def check_path_defaults(path: pathlib.Path, problems: list[str]) -> None:
    """Flag every statically identifiable bundled-file ``default=`` which is missing.

    An example is run with no arguments — by ``eden examples smoke``, and by the reader
    following its docstring — so a default that points at a moved asset is a broken
    example, and one that no other check sees, because nothing else runs an uninstrumented
    example.

    ``__file__``-anchored defaults under ``REPO_ROOT`` and explicit ``examples/...``
    literals are judged. Other relative strings may be choices (``--logger tensorboard``)
    or user-populated paths (``eden/assets/amass``), and absolute paths outside the repo
    belong to the user's machine. ``-o``'s destination is written rather than read, so it
    is exempt.
    """
    if (tree := _tree(path)) is None:
        return
    env = _module_paths(path, tree)
    for node in _add_argument_calls(tree):
        flags = _flags(node)
        if {"-o", *RESERVED["-o"]} & set(flags):
            continue
        for kw in node.keywords:
            if kw.arg != "default" or (target := _static_path(kw.value, env)) is None:
                continue
            if (target_path := _repo_target(target)) is None or target_path.exists():
                continue
            where = f"{path.relative_to(REPO_ROOT)}:{node.lineno}"
            flag = next((f for f in flags if f.startswith("--")), "positional")
            problems.append(f"{where}: {flag} defaults to {target_path.relative_to(REPO_ROOT)}, which does not exist")


def _declared(tree: ast.AST) -> tuple[set[str], set[str]]:
    """The flags the file's parsers accept and the dests argparse derives from them."""
    flags, dests = set(), set()
    for node in _add_argument_calls(tree):
        names = _flags(node)
        flags.update(f for f in names if f.startswith("-"))
        explicit = next((k for k in node.keywords if k.arg == "dest" and isinstance(k.value, ast.Constant)), None)
        if explicit is not None:
            dests.add(explicit.value)
        elif names:  # argparse derives the dest from the long option, falling back to the short
            longs = [f for f in names if f.startswith("--")]
            dests.add((longs[0] if longs else names[0]).lstrip("-").replace("-", "_"))
    return flags, dests


def _did_you_mean(name: str, known: set[str]) -> str:
    close = difflib.get_close_matches(name, sorted(known), n=1, cutoff=0.6)
    return f" (did you mean {close[0]}?)" if close else ""


def check_arg_handoffs(path: pathlib.Path, problems: list[str]) -> None:
    """Flag a subcommand chain that names arguments the script's own parser never produces.

    A training example chains into its own ``inference`` to render a video of the policy it
    just trained, and that handoff runs *after* ``runner.learn`` returns — so a name
    argparse never produced surfaces as an ``AttributeError`` at the finish line, with the
    whole run already paid for. A typical slip hands ``Namespace(rec=True, out=None)`` to an
    ``inference`` reading ``args.record`` / ``args.output``: ``rec`` and ``out`` are
    abbreviations, and argparse derives the dest from the long option.

    Two spellings of the handoff are checked against the flags and dests the file's own
    ``add_argument`` calls declare — the keywords of a hand-built ``argparse.Namespace``,
    and the literal flags of a ``parse_args([...])`` argv. Subparsers are pooled rather
    than tracked separately: a lint, so a flag borrowed from a sibling subcommand passes.
    ``Namespace(**vars(args))`` carries no literal keywords and is a copy of a real parse,
    so it has nothing to drift from and nothing to check.
    """
    if (tree := _tree(path)) is None:
        return
    flags, dests = _declared(tree)
    if not flags:
        return
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        where = f"{path.relative_to(REPO_ROOT)}:{node.lineno}"
        called = _dotted(node.func) or ""
        if called.split(".")[-1] in ("Namespace", "SimpleNamespace"):
            for kw in node.keywords:
                if kw.arg is not None and kw.arg not in dests:
                    problems.append(
                        f"{where}: Namespace({kw.arg}=...) is no parser's dest{_did_you_mean(kw.arg, dests)}"
                    )
        elif called.endswith("parse_args") and node.args and isinstance(node.args[0], ast.List):
            for item in node.args[0].elts:
                if not isinstance(item, ast.Constant) or not isinstance(item.value, str):
                    continue
                flag = item.value.split("=", 1)[0]
                if re.fullmatch(r"--?[A-Za-z][\w-]*", flag) and flag not in flags:
                    problems.append(
                        f"{where}: parse_args passes {flag}, which no parser accepts{_did_you_mean(flag, flags)}"
                    )


def _registered_task_classes(tree: ast.AST, path: pathlib.Path, problems: list[str]) -> dict[str, str]:
    """Local names that ``from eden.tasks... import X`` binds to a registered task config → task name.

    Resolved through the registry's import-free discovery map (the one ``eden list`` uses), so
    a same-named class from anywhere else is not mistaken for a task and no task module gets
    imported. A class reached through a package re-export (``from eden.tasks.foo import X``
    when ``X`` is defined in ``eden.tasks.foo.config``) is not recognized — a lint's false
    negative. An import the registry cannot answer for (a task name that clashes, a broken task
    plugin) is reported as a problem of its own, so one clash cannot crash the whole lint.
    """
    from eden.tasks import TASK_REGISTRY

    bound: dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or not (node.module or "").startswith("eden.tasks"):
            continue
        for alias in node.names:
            qualname = f"{node.module}.{alias.name}"
            try:
                task = TASK_REGISTRY.task_for_config_qualname(qualname)
            except (ValueError, RuntimeError) as exc:  # the registry's own error, reported below
                summary = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
                problems.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}: importing {qualname}: {summary}")
                continue
            if task is not None:
                bound[alias.asname or alias.name] = task
    return bound


def check_task_builds(path: pathlib.Path, problems: list[str]) -> None:
    """Flag a task config instantiated around its TaskMods.

    Two spellings do it: ``TASK_REGISTRY.get(<name>)()`` — ``get`` returns the bare config
    class — and importing that class from ``eden.tasks`` and calling it (``ReacherConfig()``).
    Neither runs the mods registered on the task, so a task whose robot binding lives in a
    mod comes back unbuildable, and a mod added later is silently skipped by any script that
    constructs the class itself. ``TASK_REGISTRY.build(<name>)`` is the same call for a
    mod-less task and the only correct one for the rest, so the fix is mechanical.

    Both matches are literal: an aliased registry, a two-step ``cls = TASK_REGISTRY.get(x)``
    then ``cls()``, and a class reached through a package re-export slip through, as a lint
    should err toward a false negative. A base-class position, an ``isinstance`` and an
    annotation are not calls and are not flagged.
    """
    if (tree := _tree(path)) is None:
        return
    classes = _registered_task_classes(tree, path, problems)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        where = f"{path.relative_to(REPO_ROOT)}:{node.lineno}"
        if isinstance(node.func, ast.Call) and _dotted(node.func.func) == "TASK_REGISTRY.get":
            name = ast.unparse(node.func.args[0]) if node.func.args else "<name>"
            problems.append(
                f"{where}: {ast.unparse(node)} instantiates the task around its TaskMods; "
                f"use TASK_REGISTRY.build({name}) instead"
            )
        elif isinstance(node.func, ast.Name) and node.func.id in classes:
            task = classes[node.func.id]
            problems.append(
                f"{where}: {ast.unparse(node)} instantiates task '{task}' around its TaskMods; "
                f"use TASK_REGISTRY.build({task!r}) instead"
            )
