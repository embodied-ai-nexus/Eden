"""The conventions of ``docs/guides/task-config.md`` § Task Package Layout, checked across ``eden/tasks``.

Import-free, like the registry's own discovery: names and config classes come from
:meth:`~eden.tasks.registry.TaskRegistry.config_qualname_for_task` (the AST scan's record), and the task modules and
``__init__.py`` files are read with :mod:`ast`.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import eden.tasks
from eden.tasks import TASK_REGISTRY, TASK_TAGS
from eden.tasks.registry import (
    TaskRegistry,
    _decorator_keyword,
    _default_task_name,
    _register_decorator,
    _task_registry_aliases,
)

TASKS_ROOT = Path(eden.tasks.__file__).parent
SNAKE_CASE = re.compile(r"[a-z][a-z0-9]*(_[a-z0-9]+)*")

#: Package ``__init__``s that are a documented import surface (``docs/tasks/libero.md`` renders it).
API_FACADE_INITS = {"libero/__init__.py"}
#: Packages whose tasks are not named after the package: ``benchmark`` holds Eden's own reference tasks
#: (``reacher`` is the canonical task-config template), not a ported suite.
UNPREFIXED_PACKAGES = {"benchmark"}


def _builtin_tasks() -> dict[str, str]:
    """Task name -> ``module.Class`` for every task Eden itself ships (not ones a test or plugin registered)."""
    qualnames = {name: TASK_REGISTRY.config_qualname_for_task(name) for name in TASK_REGISTRY.list_tasks()}
    return {name: q for name, q in qualnames.items() if q is not None and q.startswith("eden.tasks.")}


def test_builtin_tasks_are_found():
    """Guards the tests below against passing vacuously on an empty scan."""
    assert len(_builtin_tasks()) > 50


def test_task_names_are_snake_case():
    offenders = sorted(name for name in _builtin_tasks() if not SNAKE_CASE.fullmatch(name))
    assert not offenders, (
        f"task names {offenders} are not snake_case; name the task with lower-case words joined by '_' "
        f"(e.g. 'robosuite_two_arm_lift'), starting with the name of its package under eden/tasks."
    )


def test_task_names_start_with_their_package():
    """``polaris_pan_clean`` reads as PolaRiS's in ``eden list``, a log path or a dataset; ``pan_clean`` does not."""
    offenders = {}
    for name, qualname in _builtin_tasks().items():
        package = qualname.split(".")[2]
        if package in UNPREFIXED_PACKAGES:
            continue
        if name != package and not name.startswith(f"{package}_"):
            offenders[name] = f"eden/tasks/{package}"
    assert not offenders, (
        f"task names {offenders} do not start with their package's name; name the config class "
        f"'<Package><Task>Config' so the key derives as '<package>_<task>' (PolarisPanCleanConfig -> "
        f"'polaris_pan_clean' in eden/tasks/polaris)."
    )


def test_tasks_are_defined_in_a_config_module():
    """One place to look: ``<package>/config.py``, or ``<suite>/<task>/config.py`` in a suite."""
    offenders = {}
    for name, qualname in _builtin_tasks().items():
        *folders, module = qualname.rpartition(".")[0].split(".")[2:]  # below eden.tasks: folders, then the module
        # One folder is <package>; two are <suite>/<task>.
        if module != "config" or len(folders) not in (1, 2):
            offenders[name] = "eden/tasks/" + "/".join([*folders, module]) + ".py"
    assert not offenders, (
        f"tasks {offenders} are not defined in a config module; move each config class to "
        f"eden/tasks/<package>/config.py or eden/tasks/<suite>/<task>/config.py, and keep the "
        f"suite's shared base config and helpers in other modules at its top level."
    )


def test_config_classes_end_in_config():
    offenders = sorted(
        f"{qualname.rpartition('.')[2]} ({name})"
        for name, qualname in _builtin_tasks().items()
        if not qualname.endswith("Config")
    )
    assert not offenders, (
        f"registered config classes {offenders} do not end in 'Config'; name the class '<Package><Task>Config' — "
        f"the registry strips the suffix, so the task name does not change."
    )


def test_no_registration_passes_the_name_its_class_already_derives():
    """``name=`` marks a key the class name cannot produce; one that repeats the derived key hides that signal."""
    redundant = []
    for path in sorted(TASKS_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(), str(path))
        aliases = _task_registry_aliases(tree)  # the registry's own reading of the decorator, aliases included
        for cls in (node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)):
            decorator = _register_decorator(cls, aliases=aliases, file_path=path)
            name = None if decorator is None else _decorator_keyword(decorator, "name")
            if isinstance(name, ast.Constant) and name.value == _default_task_name(cls.name):
                redundant.append(f"{path.relative_to(TASKS_ROOT.parent.parent).as_posix()}: {cls.name}")
    assert not redundant, (
        f"these registrations pass name= equal to the key their class name already derives: {redundant}. Drop the "
        f"name= argument; keep name= only where no class name produces the key (a run-together package name such as "
        f"'beyondmimic', an acronym such as 'manipulation_lift_ycb_rgb', or a class name taken elsewhere)."
    )


def test_task_package_inits_import_nothing():
    """A task package's ``__init__`` is its docstring: importing one of its modules must not build its config."""
    offenders = {}
    for path in sorted(TASKS_ROOT.rglob("__init__.py")):
        relative = path.relative_to(TASKS_ROOT).as_posix()
        if relative == "__init__.py" or "custom_terms" in path.parts or relative in API_FACADE_INITS:
            continue
        imports = [
            ast.unparse(node)
            for node in ast.parse(path.read_text(), str(path)).body
            if isinstance(node, (ast.Import, ast.ImportFrom)) and getattr(node, "module", None) != "__future__"
        ]
        if imports:
            offenders[f"eden/tasks/{relative}"] = imports
    assert not offenders, (
        f"task package __init__ modules import code: {offenders}. Keep the package docstring and delete the imports; "
        f"callers import from the defining module (eden.tasks.<pkg>.config), and the task registry needs no import "
        f"to find a task. A custom_terms/__init__.py may re-export its terms."
    )


#: Each family heading of ``docs/tasks/index.md`` and the tag every task its tables name must carry.
INDEX_FAMILY_TAGS = {
    "Locomotion": "locomotion",
    "Motion tracking": "motion_tracking",
    "Manipulation and dexterous hands": "manipulation",
    "Ported benchmarks": "benchmark_suite",
}
TASK_INDEX = TASKS_ROOT.parent.parent / "docs" / "tasks" / "index.md"


def _scanned_tags() -> dict[str, tuple[str, ...]]:
    """Each built-in task's tags as ``eden list`` reads them: off the decorator, in a registry nothing was imported into.

    Not ``TASK_REGISTRY.task_tags``, which answers from the live registration once another test has imported the
    module, so a ``tags=`` the scan cannot read would pass here and still list as untagged.
    """
    scanned = TaskRegistry()
    return {name: scanned.task_tags(name) for name in _builtin_tasks()}


def test_every_builtin_task_is_tagged():
    """An untagged task drops out of every ``eden list --tag`` filter."""
    untagged = sorted(name for name, tags in _scanned_tags().items() if not tags)
    assert not untagged, (
        f"tasks {untagged} carry no tag; pass tags=(...) to their @TASK_REGISTRY.register(), with every key of "
        f"TASK_TAGS (eden/tasks/registry.py) whose rule applies, e.g. tags=('manipulation', 'dexterous')."
    )


def _index_tables() -> dict[str, set[str]]:
    """Family heading -> the task names in backticks on the table rows under it, in ``docs/tasks/index.md``."""
    tables: dict[str, set[str]] = {}
    heading = None
    for line in TASK_INDEX.read_text().splitlines():
        if line.startswith("## "):
            heading = line[3:].strip()
        elif heading in INDEX_FAMILY_TAGS and line.startswith("|"):
            tables.setdefault(heading, set()).update(re.findall(r"`([a-z0-9_]+)`", line))
    return tables


def test_task_index_tables_agree_with_the_tags():
    """The family tables of ``docs/tasks/index.md`` and ``eden list --tag`` must group the tasks the same way."""
    tables = _index_tables()
    assert set(tables) == set(INDEX_FAMILY_TAGS), (
        f"docs/tasks/index.md has no table under {sorted(set(INDEX_FAMILY_TAGS) - set(tables))}; "
        f"update INDEX_FAMILY_TAGS in this test if a family heading was renamed."
    )
    tags = _scanned_tags()
    mistagged = {
        name: f"listed under '{heading}', tags {tags[name]}"
        for heading, names in tables.items()
        for name in sorted(names & set(tags))
        if INDEX_FAMILY_TAGS[heading] not in tags[name]
    }
    assert not mistagged, (
        f"{mistagged}: a task in a family table of docs/tasks/index.md must carry that family's tag "
        f"{INDEX_FAMILY_TAGS}. Fix whichever is wrong: the task's tags=(...) or the table it is listed in."
    )
    # The Ported benchmarks table lists a PolaRiS `_batched` variant through its base task ("and a `_batched` variant
    # of each"), so a suite task counts as listed when its name without that suffix is.
    unlisted = sorted(
        name
        for name, task_tags in tags.items()
        if not any(
            {name, name.removesuffix("_batched")} & tables[h]
            for h, tag in INDEX_FAMILY_TAGS.items()
            if tag in task_tags
        )
    )
    assert not unlisted, (
        f"tasks {unlisted} are in no family table of docs/tasks/index.md that matches their tags; add a row for "
        f"each to the table of one of its tags ({INDEX_FAMILY_TAGS})."
    )


def test_the_task_config_guide_lists_every_tag():
    """``docs/guides/task-config.md`` § Tags is where a contributor picks tags; it must not drift from ``TASK_TAGS``."""
    guide = (TASKS_ROOT.parent.parent / "docs" / "guides" / "task-config.md").read_text()
    section = guide.split("### Tags", 1)[1].split("\n## ", 1)[0]
    documented = set(re.findall(r"^\| `([a-z_]+)` \|", section, flags=re.MULTILINE))
    assert documented == set(TASK_TAGS), (
        f"the Tags table of docs/guides/task-config.md lists {sorted(documented)}, but TASK_TAGS defines "
        f"{sorted(TASK_TAGS)}; give each tag in TASK_TAGS a row with its rule, and drop rows for removed tags."
    )
