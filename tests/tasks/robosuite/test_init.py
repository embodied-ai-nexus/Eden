"""The layout rule ``eden/tasks/robosuite/__init__.py`` states: one task's code in its directory, shared code at the top.

Read off the source with :mod:`ast` (imports and names, never docstrings), so a mention in prose neither
counts as a use nor hides one. Nothing here imports a task module.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import eden.tasks.robosuite as robosuite
from eden.tasks import TASK_REGISTRY
from eden.tasks.robosuite import custom_terms

ROOT = Path(robosuite.__file__).parent
PACKAGE = robosuite.__name__
#: The family's shared term package; every other directory is one task.
SHARED_TERMS = f"{PACKAGE}.custom_terms"
TASKS = sorted(p.name for p in ROOT.iterdir() if p.is_dir() and p.name not in ("custom_terms", "__pycache__"))
#: Top-level modules no task imports: the package docstring, and the ``eden datagen`` registry, which
#: imports the tasks (their routines), not the other way round.
NOT_IMPORTED_BY_TASKS = {"__init__", "scripted_demos"}
TOP_LEVEL = sorted(path.stem for path in ROOT.glob("*.py") if path.stem not in NOT_IMPORTED_BY_TASKS)


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(), str(path))


def _task_modules(task: str) -> list[Path]:
    """Every module of ``task``'s directory, sub-packages included."""
    return sorted(path for path in (ROOT / task).rglob("*.py") if "__pycache__" not in path.parts)


def _imported_modules(path: Path) -> set[str]:
    """Every absolute module ``path`` imports, anywhere in it (``TYPE_CHECKING`` blocks and functions included).

    ``from <module> import NAME`` records ``<module>.NAME`` as well, since ``NAME`` may be a submodule.
    """
    found = set()
    for node in ast.walk(_parse(path)):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            found.add(node.module)
            found.update(f"{node.module}.{alias.name}" for alias in node.names)
    return found


def _shared_terms_used(path: Path) -> set[str]:
    """The shared terms ``path`` reaches through the package: ``custom_terms.X`` via its binding, or imported from it.

    Only a name reached *through* the package counts, so a task's own name that happens to match a shared
    term's does not.
    """
    tree = _parse(path)
    bindings = set()
    used = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module == PACKAGE:
            bindings.update(alias.asname or alias.name for alias in node.names if alias.name == "custom_terms")
        elif isinstance(node, ast.Import):
            bindings.update(alias.asname for alias in node.names if alias.name == SHARED_TERMS and alias.asname)
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith(SHARED_TERMS):
            used.update(alias.name for alias in node.names)
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id in bindings:
            used.add(node.attr)
    return used & set(custom_terms.__all__)


def _task_of(module: str) -> str | None:
    """The task directory ``eden.tasks.robosuite.<task>...`` names, or ``None`` for anything else."""
    parts = module.split(".")
    if parts[:3] == PACKAGE.split(".") and len(parts) > 3 and parts[3] in TASKS:
        return parts[3]
    return None


def _importers(module: str) -> set[str]:
    """The tasks whose directories import top-level ``module``."""
    return {task for task in TASKS for path in _task_modules(task) if f"{PACKAGE}.{module}" in _imported_modules(path)}


def test_each_task_directory_holds_the_config_of_the_task_it_is_named_after():
    # Import-free: the registry answers from its AST scan. This also guards ``TASKS`` itself.
    registered = {
        name.removeprefix("robosuite_"): qualname
        for name in TASK_REGISTRY.list_tasks()
        if (qualname := TASK_REGISTRY.config_qualname_for_task(name) or "").startswith(f"{PACKAGE}.")
    }
    misplaced = {
        task: qualname for task, qualname in registered.items() if not qualname.startswith(f"{PACKAGE}.{task}.config.")
    }
    assert set(registered) == set(TASKS) and not misplaced, (
        f"Task directories {TASKS} and the robosuite tasks registered {sorted(registered)} disagree "
        f"(misplaced configs: {misplaced}). Each task is a directory eden/tasks/robosuite/<task>/ whose config.py "
        f"registers robosuite_<task>; a directory that is not a task belongs at the top level or in custom_terms/."
    )


@pytest.mark.parametrize(
    "module", sorted(str(p.relative_to(ROOT)) for p in ROOT.rglob("*.py") if p.name != "__init__.py")
)
def test_no_module_imports_relatively(module):
    relative = [n for n in ast.walk(_parse(ROOT / module)) if isinstance(n, ast.ImportFrom) and n.level > 0]
    assert not relative, (
        f"eden/tasks/robosuite/{module} line {relative[0].lineno if relative else '?'} imports relatively. Write the "
        f"absolute eden.tasks.robosuite... path, as AGENTS.md asks outside __init__.py — a relative import also "
        f"hides a cross-task import from the layout checks here."
    )


@pytest.mark.parametrize("task", TASKS)
def test_a_task_imports_no_other_tasks_code(task):
    crossings = []
    for path in _task_modules(task):
        foreign = {module for module in _imported_modules(path) if _task_of(module) not in (None, task)}
        # ``from <module> import NAME`` records ``<module>.NAME`` as well: report the module once.
        crossings += [
            f"{path.relative_to(ROOT)} imports {module}"
            for module in sorted(foreign)
            if not any(module.startswith(f"{other}.") for other in foreign)
        ]
    assert not crossings, (
        f"{'; '.join(crossings)}. A module more than one task uses lives at the top of eden/tasks/robosuite "
        f"(a term in custom_terms/, anything else in common.py or a module of its own, as staged_objects.py and "
        f"variants.py are); move it there instead of importing it from another task's directory."
    )


@pytest.mark.parametrize("task", TASKS)
def test_a_call_site_says_which_custom_terms_a_term_comes_from(task):
    own = f"{PACKAGE}.{task}.custom_terms"
    shared_names = set(custom_terms.__all__)
    wrong = []
    for path in _task_modules(task):
        where = path.relative_to(ROOT)
        for node in ast.walk(_parse(path)):
            if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = {alias.name for alias in node.names}
                if node.module == f"{PACKAGE}.{task}" and "custom_terms" in names:
                    wrong.append(f"{where} imports {own} as a module")
                if node.module.startswith(SHARED_TERMS) and names & shared_names:
                    wrong.append(f"{where} imports shared terms {sorted(names & shared_names)} by name")
            elif isinstance(node, ast.Import) and any(alias.name == own for alias in node.names):
                wrong.append(f"{where} imports {own} as a module")
    assert not wrong, (
        f"{'; '.join(wrong)}. Both levels are called custom_terms, so the call site tells them apart: a task's own "
        f"terms by bare name (from {own} import ...), the shared ones as custom_terms.X after "
        f"`from eden.tasks.robosuite import custom_terms`. The mixins in custom_terms.common are helpers, not "
        f"terms, and import by name."
    )


def test_every_shared_term_is_used_by_more_than_one_task():
    # A term a top-level module uses counts as used by every task importing that module: the check works at
    # the module's granularity, not its functions', so a one-task factory in common.py would slip past it.
    users_by_term: dict[str, set[str]] = {name: set() for name in custom_terms.__all__}
    for task in TASKS:
        for path in _task_modules(task):
            for name in _shared_terms_used(path):
                users_by_term[name].add(task)
    for module in TOP_LEVEL:
        for name in _shared_terms_used(ROOT / f"{module}.py"):
            users_by_term[name] |= _importers(module)
    single = {name: sorted(users) for name, users in users_by_term.items() if len(users) < 2}
    assert not single, (
        f"custom_terms exports terms fewer than two tasks use: {single}. Move each into "
        f"eden/tasks/robosuite/<task>/custom_terms.py (and drop it from custom_terms.__all__), or delete an unused one."
    )


@pytest.mark.parametrize("module", TOP_LEVEL)
def test_every_top_level_module_is_imported_by_more_than_one_task(module):
    importers = sorted(_importers(module))
    assert len(importers) >= 2, (
        f"eden/tasks/robosuite/{module}.py is imported by {importers or 'no task'}, but the top level is for code "
        f"more than one task uses. Move it into eden/tasks/robosuite/{importers[0] if importers else '<task>'}/, "
        f"or delete it if nothing uses it."
    )
