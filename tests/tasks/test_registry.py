"""Tests for ``eden.tasks.registry``: ``TaskMod``, ``TaskRegistry`` build/inheritance, and the AST discovery scan."""

from __future__ import annotations

import importlib
import pathlib
import sys
import time
import types
import warnings
from typing import Annotated, Literal

import pytest

import eden
from eden.__main__ import main as eden_main
from eden.tasks import TASK_REGISTRY
from eden.tasks.benchmark.move_forward.config import MoveForwardConfig
from eden.tasks.registry import (
    TaskMod,
    TaskRegistry,
    _default_task_name,
    _scan_packages_for_tasks,
    cli_fields,
    literal_choices,
    unwrap_optional,
)
from eden.utils.configs import EdenConfig, EdenRLConfig


class _Cfg(EdenConfig):
    """A real config with two fields the test mods write."""

    value: int = 0
    label: str = "default"


class _ChildConfig(_Cfg):
    pass


class _MidConfig(_Cfg):
    """Never registered: inheritance must walk through it to the registered base."""


class _LeafConfig(_MidConfig):
    pass


class _GrandChildConfig(_ChildConfig):
    pass


class _UnrelatedConfig(EdenConfig):
    """Not a ``_Cfg``, so shadowing a name with it breaks the inheritance chain."""

    value: int = 0
    label: str = "default"


class _AddValueMod(TaskMod):
    def __init__(self, delta: int = 0):
        super().__init__()
        self.delta = delta

    def apply(self, config):
        return config.updated(value=config.value + self.delta)


class _SetLabelMod(TaskMod):
    def __init__(self, label: Literal["a", "b", "c"] = "a"):
        super().__init__()
        self.label = label

    def apply(self, config):
        return config.updated(label=self.label)


class _EntityLabelMod(TaskMod):
    """A CLI field plus keyword-only (non-CLI) ctor state."""

    def __init__(self, label: Literal["a", "b", "c"] = "a", *, target: str = "default"):
        super().__init__()
        self.label = label
        self.target = target

    def apply(self, config):
        return config.updated(label=f"{self.target}={self.label}")


class _NonStringMarker:
    """Non-string ``Annotated`` metadata (module scope so ``get_type_hints`` resolves it)."""


def _registry(*registrations) -> TaskRegistry:
    """A fresh registry with ``(cls, name, kwargs)`` registrations applied in order."""
    reg = TaskRegistry()
    for cls, name, kwargs in registrations:
        reg.register(cls, name=name, **kwargs)
    return reg


_IMPORT = "from eden.tasks import TASK_REGISTRY\n"


def _task_class(name: str, cls_name: str) -> str:
    """Source for an empty class registered as task ``name``, for a task file after ``_IMPORT``."""
    return f"@TASK_REGISTRY.register(name='{name}')\nclass {cls_name}:\n    pass\n"


# Two files of one package registering the same name.
_CLASHING_FILES = {"b.py": _IMPORT + _task_class("dup", "OneConfig"), "c.py": _IMPORT + _task_class("dup", "TwoConfig")}


def _install_distribution(tmp_path, monkeypatch, dist: str, entry_points: dict[str, str]) -> None:
    """Write the ``.dist-info`` an install leaves, declaring ``entry_points`` under ``eden.tasks``.

    ``importlib.metadata`` reads it straight off ``sys.path``, so this is an install, not a stand-in for one.
    """
    info = tmp_path / f"{dist.replace('-', '_')}-0.1.dist-info"
    info.mkdir()
    (info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {dist}\nVersion: 0.1\n")
    declared = "".join(f"{key} = {value}\n" for key, value in entry_points.items())
    (info / "entry_points.txt").write_text(f"[eden.tasks]\n{declared}")
    if str(tmp_path) not in sys.path:
        monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()


# ── TaskMod ─────────────────────────────────────────────────────────────────


def test_taskmod_default_apply_raises_notimplemented():
    with pytest.raises(NotImplementedError, match="must implement apply"):
        TaskMod().apply(_Cfg())


def test_with_prefix_renames_flags_on_that_instance_only():
    """``with_prefix`` chains, re-prefixes, and never leaks to other instances or subclasses."""

    class _NoSuperInit(TaskMod):
        def __init__(self, x: int = 0):  # no super().__init__(): the class-level default carries it
            self.x = x

        def apply(self, config):
            return config.updated(value=config.value + self.x)

    mod = _NoSuperInit()
    assert (mod.prefix, mod.flag_for("x")) == ("", "x")
    assert mod.with_prefix("left") is mod
    assert mod.flag_for("x") == "left_x"
    mod.with_prefix("right")
    assert mod.flag_for("x") == "right_x"

    assert _NoSuperInit().prefix == ""
    assert _SetLabelMod().prefix == ""
    assert TaskMod().prefix == ""
    assert _registry((_Cfg, "no_super", {"modifiers": (_NoSuperInit,)})).build("no_super", x=5).value == 5


# ── cli_fields / annotation helpers ─────────────────────────────────────────


def test_cli_fields_reads_positional_params_and_their_annotated_help():
    """Keyword-only params are not CLI fields; help is the first string ``Annotated`` metadata."""

    class _Mod(TaskMod):
        def __init__(
            self,
            level: Annotated[Literal["a", "b"], "Pick a level."] = "a",
            a: Annotated[int, "first help", "second help"] = 0,
            b: Annotated[int, _NonStringMarker(), "actual help"] = 0,
            c: Annotated[int, _NonStringMarker()] = 0,
            plain: str = "x",
            *,
            internal: str = "z",
        ):
            self.level, self.a, self.b, self.c, self.plain, self.internal = level, a, b, c, plain, internal

    fields = {name: (annotation, default, help_text) for name, annotation, default, help_text in cli_fields(_Mod)}
    assert fields == {
        "level": (Literal["a", "b"], "a", "Pick a level."),
        "a": (int, 0, "first help"),
        "b": (int, 0, "actual help"),
        "c": (int, 0, None),
        "plain": (str, "x", None),
    }


def test_cli_fields_warns_when_type_hints_do_not_resolve():
    """A typo'd forward reference warns instead of flowing on as a raw string."""

    class _Mod(TaskMod):
        def __init__(self, x: int = 0):
            self.x = x

    _Mod.__init__.__annotations__["x"] = "DefinitelyNotAType"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cli_fields(_Mod)
    messages = [str(w.message) for w in caught]
    assert any("could not resolve type hints" in m and "_Mod" in m for m in messages), messages


def test_literal_choices_and_unwrap_optional():
    assert literal_choices(Literal["a", "b"]) == ("a", "b")
    assert literal_choices(int) is None
    assert unwrap_optional(str) == (str, False)
    assert unwrap_optional(str | None) == (str, True)
    # Two real members are not narrowable.
    assert unwrap_optional(int | str) == (int | str, False)


def test_optional_field_is_still_type_checked():
    """Widening a field to ``T | None`` must not switch its validation off."""

    class _OptionalPathMod(TaskMod):
        def __init__(self, path: Annotated[str | None, "A path, or None."] = None):
            self.path = path

        def apply(self, config):
            return config.updated(label="none" if self.path is None else self.path)

    reg = _registry((_Cfg, "opt", {"modifiers": (_OptionalPathMod,)}))
    with pytest.raises(ValueError, match="expected str, got int"):
        reg.build("opt", path=5)
    assert reg.build("opt", path=None).label == "none"
    assert reg.build("opt", path="/tmp/x").label == "/tmp/x"


def test_optional_literal_field_still_checks_membership():
    class _OptionalLevelMod(TaskMod):
        def __init__(self, level: Literal["a", "b"] | None = None):
            self.level = level

        def apply(self, config):
            return config.updated(label=self.level or "unset")

    reg = _registry((_Cfg, "optlit", {"modifiers": (_OptionalLevelMod,)}))
    with pytest.raises(ValueError, match="not one of"):
        reg.build("optlit", level="z")
    assert reg.build("optlit", level="b").label == "b"
    assert reg.build("optlit", level=None).label == "unset"


# ── register / get_modifiers / build ────────────────────────────────────────


def test_register_and_build_applies_mod_defaults_then_kwargs():
    reg = _registry(
        (_Cfg, "dummy", {"modifiers": (_AddValueMod, _SetLabelMod)}),
        (_UnrelatedConfig, "bare", {}),
    )
    assert reg.get("dummy") is _Cfg
    assert tuple(type(m) for m in reg.get_modifiers("dummy")) == (_AddValueMod, _SetLabelMod)

    default = reg.build("dummy")
    assert (type(default), default.value, default.label) == (_Cfg, 0, "a")
    built = reg.build("dummy", delta=42, label="c")
    assert (built.value, built.label) == (42, "c")
    assert reg.build("bare").label == "default"


def test_registered_instance_default_applies_and_is_never_mutated_by_a_build():
    mod = _AddValueMod(delta=11)
    reg = _registry((_Cfg, "preset", {"modifiers": (mod,)}))
    assert reg.get_modifiers("preset") == (mod,)
    assert reg.build("preset").value == 11
    assert reg.build("preset", delta=3).value == 3
    assert mod.delta == 11


def test_build_deep_copies_mutable_instance_state():
    """A list mutated inside ``apply()`` must not accumulate on the registered mod across builds."""

    class _StatefulMod(TaskMod):
        def __init__(self, delta: int = 0):
            super().__init__()
            self.delta = delta
            self.history: list[int] = []

        def apply(self, config):
            self.history.append(self.delta)
            return config

    mod = _StatefulMod()
    reg = _registry((_Cfg, "stateful", {"modifiers": (mod,)}))
    for delta in (1, 2, 3):
        reg.build("stateful", delta=delta)
    assert mod.history == []


def test_modifiers_apply_in_registration_order():
    class _Set5(TaskMod):
        def apply(self, config):
            return config.updated(value=5)

    class _Set9(TaskMod):
        def apply(self, config):
            return config.updated(value=9)

    assert _registry((_Cfg, "ordered", {"modifiers": (_Set5, _Set9)})).build("ordered").value == 9


def test_prefixed_instances_of_one_mod_are_addressed_independently():
    reg = _registry(
        (
            _Cfg,
            "two_entities",
            {
                "modifiers": (
                    _EntityLabelMod(target="first").with_prefix("a"),
                    _EntityLabelMod(target="second").with_prefix("b"),
                )
            },
        ),
        (_Cfg, "prefixed", {"modifiers": (_EntityLabelMod(target="t").with_prefix("scope"),)}),
    )
    assert reg.build("two_entities").label == "second=a"
    assert reg.build("two_entities", a_label="b", b_label="c").label == "second=c"
    assert reg.build("prefixed", scope_label="b").label == "t=b"
    with pytest.raises(ValueError, match="Unknown modifier kwargs"):
        reg.build("prefixed", label="a")


def test_build_unknown_kwarg_raises():
    reg = _registry((_Cfg, "dummy", {"modifiers": (_AddValueMod,)}))
    with pytest.raises(ValueError, match="Unknown modifier kwargs"):
        reg.build("dummy", unknown_field=1)


@pytest.mark.parametrize(("prefix", "kwarg"), [("", "target"), ("scope", "scope_target")])
def test_build_unknown_kwarg_hints_when_field_is_keyword_only(prefix, kwarg):
    """Declaring a CLI knob after ``*`` gets a hint naming the flag typed and the field to move."""

    class _MisplacedMod(TaskMod):
        def __init__(self, *, target: Literal["a", "b"] = "a"):
            self.target = target

        def apply(self, config):
            return config

    reg = _registry((_Cfg, "misplaced", {"modifiers": (_MisplacedMod().with_prefix(prefix),)}))
    with pytest.raises(ValueError) as exc:
        reg.build("misplaced", **{kwarg: "b"})
    msg = str(exc.value)
    assert "Unknown modifier kwargs" in msg
    assert f"'{kwarg}' is keyword-only on _MisplacedMod (as 'target')" in msg
    assert "before `*`" in msg


def test_build_flag_collision_raises():
    reg = _registry((_Cfg, "colliding", {"modifiers": (_SetLabelMod(), _EntityLabelMod(target="t"))}))
    with pytest.raises(ValueError, match="Modifier flag collision"):
        reg.build("colliding")


def test_get_modifiers_unknown_task_raises():
    with pytest.raises(KeyError):
        TaskRegistry().get_modifiers("nonexistent")


def test_register_rejects_bogus_modifier_entry():
    with pytest.raises(TypeError, match="Modifier entries must be"):
        TaskRegistry().register(_Cfg, name="bogus", modifiers=(123,))  # type: ignore[arg-type]


def test_register_refuses_a_duplicate_name_unless_overriding():
    reg = _registry((_Cfg, "my_task", {}))
    with pytest.raises(ValueError, match="already registered"):
        reg.register(_Cfg, name="my_task")
    reg.register(_ChildConfig, name="my_task", override=True)
    assert reg.get("my_task") is _ChildConfig


def test_register_refuses_anything_but_a_class():
    """Every lookup reads the class's module and name, so a factory must fail here, not at its first lookup."""
    with pytest.raises(TypeError, match=r"takes a config class; got <function .*> \(function\)"):
        TaskRegistry().register(lambda: _Cfg(), name="factory")


def test_get_imports_a_listed_task_and_raises_on_an_unknown_one():
    assert "reacher" in TASK_REGISTRY.list_tasks()
    assert issubclass(TASK_REGISTRY.get("reacher"), EdenRLConfig)
    with pytest.raises(KeyError, match="nonexistent_task"):
        TASK_REGISTRY.get("nonexistent_task")


def test_explicit_register_takes_precedence_over_discovery():
    reg = _registry((_Cfg, "reacher", {"override": True}))
    assert reg.get("reacher") is _Cfg


# ── config qualname lookups ─────────────────────────────────────────────────


def test_config_qualname_for_task_is_forward_and_alias_safe():
    """Forward lookup answers for every name of a class; the reverse returns only the first."""
    reg = _registry((_Cfg, "primary", {}), (_Cfg, "secondary", {}))
    qualname = f"{_Cfg.__module__}._Cfg"

    assert reg.config_qualname_for_task("primary") == qualname
    assert reg.config_qualname_for_task("secondary") == qualname
    assert reg.task_for_config_qualname(qualname) == "primary"
    assert reg.config_qualname_for_task("nowhere_task") is None
    assert TASK_REGISTRY.task_for_config_qualname(f"{MoveForwardConfig.__module__}.MoveForwardConfig") == "move_forward"
    assert TASK_REGISTRY.task_for_config_qualname("nowhere.NotAConfig") is None


def test_qualname_and_modifier_lookups_do_not_import_the_task_module(task_package):
    """A discovered-only task answers from the AST scan: ``eden list`` must never import it."""
    task_package(
        "noimport_pkg",
        {
            "task.py": _IMPORT
            + "@TASK_REGISTRY.register(name='quiet', modifiers=(FooMod,))\nclass QuietConfig:\n    pass\n"
        },
    )
    reg = TaskRegistry()
    reg.add_search_path("noimport_pkg")

    assert reg.config_qualname_for_task("quiet") == "noimport_pkg.task.QuietConfig"
    assert reg.task_for_config_qualname("noimport_pkg.task.QuietConfig") == "quiet"
    assert reg.modifier_names("quiet") == ["FooMod"]
    assert reg.config_qualname_for_task("reacher") == "eden.tasks.benchmark.reacher.config.ReacherConfig"
    assert "noimport_pkg.task" not in sys.modules


@pytest.mark.parametrize("task_name", sorted(TASK_REGISTRY._ensure_discovered()))
def test_discovered_task_qualname_matches_its_runtime_class(task_name):
    """The scan's ``module:Class`` must be the qualname the live class reports.

    Imported directly rather than via ``TASK_REGISTRY.get``, which would report a wrong class
    name as a registration failure instead of as the drift it is.
    """
    scanned = TASK_REGISTRY._ensure_discovered()[task_name]
    module_path, _, class_name = scanned.partition(":")
    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as e:
        # A missing scanned module (or parent) is the drift itself; anything else is an optional dep.
        missing = e.name or ""
        if missing == module_path or module_path.startswith(f"{missing}."):
            pytest.fail(f"scan recorded module {module_path!r} for task '{task_name}', which does not import")
        pytest.skip(f"'{task_name}' needs optional dependency '{missing}'")

    cls = getattr(module, class_name, None)
    assert cls is not None, f"scan recorded class {class_name!r} in {module_path}, which does not define it"
    assert f"{cls.__module__}.{cls.__name__}" == scanned.replace(":", ".", 1)


@pytest.mark.parametrize("task_name", sorted(TASK_REGISTRY._ensure_discovered()))
def test_listed_modifiers_match_the_ones_a_build_applies(task_name):
    """``eden list``'s import-free column names the mods ``build`` really runs, inherited ones included."""
    listed = TaskRegistry().modifier_names(task_name)
    try:
        live = TASK_REGISTRY.get_modifiers(task_name)
    except ModuleNotFoundError as e:
        pytest.skip(f"'{task_name}' needs optional dependency '{e.name}'")
    assert listed == [type(m).__name__ for m in live]


# ── auto-discovery (AST scan) ───────────────────────────────────────────────


@pytest.mark.parametrize(
    ("class_name", "expected"),
    [
        ("ReacherConfig", "reacher"),
        ("MoveForwardConfig", "move_forward"),
        ("BeyondMimicConfig", "beyond_mimic"),
        ("HandStand", "hand_stand"),
        ("Config", "config"),
    ],
)
def test_default_task_name_strips_config_suffix(class_name, expected):
    assert _default_task_name(class_name) == expected


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (_IMPORT + "class Foo:\n    pass\n@TASK_REGISTRY.register\nclass BarConfig:\n    pass\n", {"bar"}),
        (_IMPORT + "@TASK_REGISTRY.register()\nclass BazConfig:\n    pass\n", {"baz"}),
        (_IMPORT + _task_class("custom", "SomethingConfig"), {"custom"}),
        (
            "from eden.tasks import TASK_REGISTRY as REG\n@REG.register(name='aliased')\nclass AliasedConfig:\n    pass\n",
            {"aliased"},
        ),
        (_IMPORT + "@OTHER_REGISTRY.register()\nclass NotATaskConfig:\n    pass\n", set()),
    ],
    ids=["bare", "call", "explicit-name", "import-alias", "other-registry"],
)
def test_discovery_reads_register_decorators(task_package, source, expected):
    task_package("decorator_pkg", {"task.py": source})
    reg = TaskRegistry()
    reg.add_search_path("decorator_pkg")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        found = {name for name in reg.list_tasks() if reg.config_qualname_for_task(name).startswith("decorator_pkg.")}
    assert found == expected
    assert not [w for w in caught if "TASK_REGISTRY" in str(w.message)]


def test_discovery_warns_when_task_registry_is_not_imported(task_package):
    """A literal ``@TASK_REGISTRY.register`` without the import warns rather than silently skipping."""
    task_package(
        "missing_import_pkg",
        {"task.py": _task_class("ghost", "GhostConfig")},
    )
    reg = TaskRegistry()
    reg.add_search_path("missing_import_pkg")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert "ghost" not in reg
    messages = [str(w.message) for w in caught]
    assert any("GhostConfig" in m and "TASK_REGISTRY" in m for m in messages), messages


def test_discovery_collision_raises_naming_both_locations(task_package):
    task_package(
        "collision_pkg",
        _CLASHING_FILES,
    )
    reg = TaskRegistry()
    reg.add_search_path("collision_pkg")
    with pytest.warns(UserWarning, match="'dup' is registered in 2 locations"):
        assert "reacher" in reg, "a collision fails the lookups of its own name, not every task"
    for lookup in (reg.list_tasks, lambda: reg.get("dup"), lambda: "dup" in reg):
        with pytest.raises(
            ValueError,
            match=r"(?s)'dup' is registered in 2 locations:\n.*b\.py.*\n.*c\.py.*"
            r"Do NOT add override=True",
        ):
            lookup()
    with pytest.raises(ValueError, match="'dup'"):
        reg.task_for_config_qualname("collision_pkg.c.TwoConfig")


def test_add_search_path_rescans(task_package):
    """A package added after the first scan is picked up; an unimportable one is skipped."""
    task_package(
        "late_pkg",
        {"task.py": _IMPORT + "@TASK_REGISTRY.register()\nclass LateConfig:\n    pass\n"},
    )
    reg = TaskRegistry()
    listed = reg.list_tasks()
    assert "reacher" in listed and "reacher_config" not in listed
    assert "late" not in listed

    reg.add_search_path("definitely.not.a.package")
    reg.add_search_path("late_pkg")
    assert "late" in reg.list_tasks()


# ── out-of-tree tasks (``eden.tasks`` entry points) ──────────────────────────


_ACME_TASK = (
    _IMPORT
    + "from eden.tasks.registry import TaskMod\n"
    + "from eden.utils.configs import EdenConfig\n"
    + "class SpeedMod(TaskMod):\n"
    + "    def __init__(self, speed: int = 1):\n"
    + "        super().__init__()\n"
    + "        self.speed = speed\n"
    + "    def apply(self, config):\n"
    + "        return config.updated(speed=self.speed)\n"
    + "@TASK_REGISTRY.register(name='acme_reach', modifiers=(SpeedMod,))\n"
    + "class AcmeReachConfig(EdenConfig):\n"
    + "    speed: int = 0\n"
)


def _table_rows(out: str) -> list[list[str]]:
    """The cells of each body row of an agent-theme table (``  cell | cell``)."""
    return [[cell.strip() for cell in line.split("|")] for line in out.splitlines() if line.startswith("  ")]


def test_installed_entry_point_task_is_listed_and_builds(
    tmp_path, monkeypatch, task_package, capsys, isolated_task_registry
):
    """An installed package's tasks reach ``eden list`` and ``build`` with no code of the user's run first."""
    task_package("acme_tasks", {"reach.py": _ACME_TASK})
    _install_distribution(tmp_path, monkeypatch, "acme-robots", {"acme": "acme_tasks"})

    monkeypatch.setenv("EDEN_LOG_THEME", "agent")  # one "name | source | modifiers" line per task
    monkeypatch.setattr(sys, "argv", ["eden", "list"])
    eden_main()
    cells = {row[0]: row[1:] for row in _table_rows(capsys.readouterr().out)}
    assert cells["acme_reach"] == ["acme-robots", "SpeedMod"]
    assert cells["reacher"][0] == "eden"
    assert "acme_tasks.reach" not in sys.modules, "listing must read the task file, not import it"

    config = isolated_task_registry.build("acme_reach", speed=3)
    assert type(config).__module__ == "acme_tasks.reach"
    assert config.speed == 3


@pytest.mark.parametrize(
    ("value", "problem"),
    [
        ("acme_absent", "no package of that name is importable"),
        ("acme_absent.tasks", "no package 'acme_absent' is importable"),
        ("acme_module:tasks", "names an object, not a package"),
        ("acme_module", "is a module"),
    ],
    ids=["missing", "missing-parent", "object-reference", "module"],
)
def test_task_entry_point_that_does_not_resolve_fails_the_lookups_it_could_answer(
    tmp_path, monkeypatch, value, problem
):
    """A broken entry point fails a miss and the listing, naming the distribution; found tasks still resolve."""
    (tmp_path / "acme_module.py").write_text("")
    _install_distribution(tmp_path, monkeypatch, "broken-robots", {"acme": value})
    reg = TaskRegistry()
    with pytest.warns(UserWarning, match="Distribution 'broken-robots' names") as record:
        assert "reacher" in reg
    # Attributed to the registry, not to whichever caller happened to trigger the scan.
    assert {pathlib.Path(w.filename).name for w in record} == {"registry.py"}
    assert "reacher" in repr(reg), "a repr renders while a plugin is broken"
    assert reg.config_qualname_for_task("reacher") == "eden.tasks.benchmark.reacher.config.ReacherConfig"

    for lookup in (reg.list_tasks, lambda: reg.get("acme_task"), lambda: "acme_task" in reg):
        with pytest.raises(RuntimeError) as excinfo:
            lookup()
        message = str(excinfo.value)
        assert f"Distribution 'broken-robots' names '{value}'" in message
        assert problem in message and "set the entry point to the dotted package" in message

    # A reverse lookup can only be in the package its module is under: the rest still answer.
    assert reg.task_for_config_qualname("eden.tasks.NotATaskConfig") is None
    package = value.partition(":")[0]
    with pytest.raises(RuntimeError, match="lies under a task package the scan could not read"):
        reg.task_for_config_qualname(f"{package}.reach.AcmeReachConfig")


def test_task_entry_point_whose_parent_fails_to_import_blames_the_import_not_the_entry_point(
    tmp_path, monkeypatch, task_package
):
    """A parent package that raises on import is a different fix from a wrong entry point."""
    parent = task_package("acme_parent", {})
    (parent / "__init__.py").write_text("import acme_dependency_not_installed\n")
    (parent / "tasks").mkdir()
    (parent / "tasks" / "__init__.py").write_text("")
    _install_distribution(tmp_path, monkeypatch, "acme-robots", {"acme": "acme_parent.tasks"})

    with pytest.warns(UserWarning), pytest.raises(RuntimeError) as excinfo:
        TaskRegistry().get("acme_task")
    message = str(excinfo.value)
    assert "importing the packages above it ('acme_parent') raised ModuleNotFoundError" in message
    assert "acme_dependency_not_installed" in message and "the entry point is fine" in message
    assert "set the entry point to the dotted package" not in message
    assert isinstance(excinfo.value.__cause__, ModuleNotFoundError), "the import's traceback rides along"


def test_a_plugin_whose_parent_package_looks_up_a_task_is_scanned_once(
    tmp_path, monkeypatch, task_package, isolated_task_registry
):
    """Locating the package imports its parent, whose lookup scans; the scan that imported it uses that one."""
    parent = task_package("acme_lookup", {})
    (parent / "__init__.py").write_text(_IMPORT + "assert 'reacher' in TASK_REGISTRY\n")
    (parent / "tasks").mkdir()
    (parent / "tasks" / "__init__.py").write_text("")
    (parent / "tasks" / "reach.py").write_text(_IMPORT + _task_class("acme_lookup_reach", "AcmeLookupReachConfig"))
    _install_distribution(tmp_path, monkeypatch, "acme-robots", {"acme": "acme_lookup.tasks"})
    scans = []
    monkeypatch.setattr(
        "eden.tasks.registry._scan_packages_for_tasks",
        lambda packages: scans.append(packages) or _scan_packages_for_tasks(packages),
    )

    assert isolated_task_registry.task_source("acme_lookup_reach") == "acme-robots"
    assert len(scans) == 1


def test_a_typo_beside_a_broken_entry_point_still_names_the_close_match(tmp_path, monkeypatch):
    """The scan cannot rule the broken package out, but the reader still gets what the KeyError would say."""
    _install_distribution(tmp_path, monkeypatch, "broken-robots", {"acme": "acme_absent"})
    with pytest.warns(UserWarning), pytest.raises(RuntimeError) as excinfo:
        TaskRegistry().get("reachr")
    message = str(excinfo.value)
    assert message.startswith("No task named 'reachr' found in task registry. Did you mean 'reacher'?")
    assert "Available: " in message and "Distribution 'broken-robots' names 'acme_absent'" in message


def test_rescan_rereads_entry_points(tmp_path, monkeypatch, task_package):
    """A plugin fixed after the first scan is picked up by the next one, like a new search path."""
    task_package("acme_tasks", {"reach.py": _ACME_TASK})
    _install_distribution(tmp_path, monkeypatch, "acme-robots", {"acme": "acme_taskz"})
    reg = TaskRegistry()
    with pytest.warns(UserWarning), pytest.raises(RuntimeError, match="'acme_taskz'"):
        reg.config_qualname_for_task("acme_reach")

    (tmp_path / "acme_robots-0.1.dist-info" / "entry_points.txt").write_text("[eden.tasks]\nacme = acme_tasks\n")
    importlib.invalidate_caches()
    reg._invalidate_discovery_cache()
    assert reg.config_qualname_for_task("acme_reach") == "acme_tasks.reach.AcmeReachConfig"
    assert reg.task_source("acme_reach") == "acme-robots"


def test_top_level_entry_point_whose_lookup_raises_names_the_exception(tmp_path, monkeypatch):
    """No parent to blame: the message quotes the lookup's own exception (and never a ``'None'`` package)."""
    monkeypatch.setitem(sys.modules, "acme_specless", types.ModuleType("acme_specless"))  # __spec__ None
    _install_distribution(tmp_path, monkeypatch, "acme-robots", {"acme": "acme_specless"})
    with pytest.warns(UserWarning), pytest.raises(RuntimeError) as excinfo:
        TaskRegistry().list_tasks()
    message = str(excinfo.value)
    assert "but looking it up raised ValueError: acme_specless.__spec__ is None" in message
    assert "'None'" not in message


def test_built_in_tasks_register_no_name_twice():
    """The in-tree guard the scan used to give by failing hard: a clash now fails only its own name."""
    reg = TaskRegistry()
    reg._ensure_discovered()
    assert reg._collisions == {}


def test_a_scanned_clash_fails_its_name_whatever_was_imported_first(task_package, isolated_task_registry):
    """Neither clashing class registers on import, so the answer cannot depend on import order."""
    task_package(
        "clash_pkg",
        {**_CLASHING_FILES, "b.py": _CLASHING_FILES["b.py"] + _task_class("b_only", "BOnlyConfig")},
    )
    reg = isolated_task_registry
    reg.add_search_path("clash_pkg")
    with pytest.warns(UserWarning, match="'dup' is registered in 2 locations"):
        # Importing b.py for its other task neither raises nor registers b's 'dup'.
        assert reg.get("b_only").__name__ == "BOnlyConfig"
    importlib.import_module("clash_pkg.c")
    for lookup in (lambda: reg.get("dup"), lambda: "dup" in reg, lambda: reg.config_qualname_for_task("dup")):
        with pytest.raises(ValueError, match="'dup' is registered in 2 locations"):
            lookup()

    # A run-time register(..., override=True) of another class is a deliberate choice, and answers.
    reg.register(_Cfg, name="dup", override=True)
    assert reg.get("dup") is _Cfg


# A config with a value field and a mod that adds to it, for task files written by a test: a
# package of its own rather than an import of this test module, which pytest may load under
# another name.
_VALUE_TASK_BASE = (
    _IMPORT
    + "from eden.tasks.registry import TaskMod\n"
    + "from eden.utils.configs import EdenConfig\n"
    + "class ValueConfig(EdenConfig):\n    value: int = 0\n"
    + "class AddValueMod(TaskMod):\n"
    + "    def __init__(self, delta: int = 0):\n        super().__init__()\n        self.delta = delta\n"
    + "    def apply(self, config):\n        return config.updated(value=config.value + self.delta)\n"
)


def test_a_subclass_inherits_through_a_class_set_aside_for_a_clash(task_package, isolated_task_registry):
    """The clash is over the name, not the class: a subclass still gets the modifiers the class declares."""
    task_package(
        "inherit_clash_pkg",
        {
            "base.py": _VALUE_TASK_BASE + "@TASK_REGISTRY.register(name='dup', modifiers=(AddValueMod(5),))\n"
            "class BaseConfig(ValueConfig):\n    pass\n"
            "@TASK_REGISTRY.register(name='inherits_dup')\nclass InheritsDupConfig(BaseConfig):\n    pass\n",
            "mid.py": _IMPORT + "from inherit_clash_pkg.base import BaseConfig\n"
            "@TASK_REGISTRY.register(name='other_dup')\nclass MidConfig(BaseConfig):\n    pass\n"
            "@TASK_REGISTRY.register(name='leaf')\nclass LeafConfig(MidConfig):\n    pass\n",
            "other.py": _IMPORT + _task_class("dup", "OtherConfig") + _task_class("other_dup", "OtherDupConfig"),
        },
    )
    reg = isolated_task_registry
    reg.add_search_path("inherit_clash_pkg")
    with pytest.warns(UserWarning, match="is registered in 2 locations"):  # 'dup' and 'other_dup'
        assert reg.build("inherits_dup").value == 5
    # Through a clashing class that itself inherits (MidConfig, set aside for 'other_dup').
    assert reg.build("leaf").value == 5
    with pytest.raises(ValueError, match="'dup' is registered in 2 locations"):
        reg.get("dup")


def test_a_set_aside_registration_runs_the_inheritance_check_an_ordinary_one_does(task_package, isolated_task_registry):
    """An ambiguous inheritance raises when the class registers, naming it, not later from a subclass's build."""
    task_package(
        "ambiguous_clash_pkg",
        {
            # Base answers to two run-time names with different mods (call form: the scan skips it).
            "base.py": _VALUE_TASK_BASE + "class BaseConfig(ValueConfig):\n    pass\n"
            "TASK_REGISTRY.register(BaseConfig, name='base_one', modifiers=(AddValueMod(1),))\n"
            "TASK_REGISTRY.register(BaseConfig, name='base_two', modifiers=(AddValueMod(2),))\n"
            "@TASK_REGISTRY.register(name='dup')\nclass ChildConfig(BaseConfig):\n    pass\n",
            "other.py": _IMPORT + _task_class("dup", "OtherConfig"),
        },
    )
    reg = isolated_task_registry
    reg.add_search_path("ambiguous_clash_pkg")
    with pytest.warns(UserWarning, match="'dup' is registered in 2 locations"):
        reg._ensure_discovered()
    with pytest.raises(ValueError, match=r"'dup' \('ChildConfig'\) would inherit the modifiers of BaseConfig"):
        importlib.import_module("ambiguous_clash_pkg.base")


def test_a_clashing_class_imported_before_the_scan_fails_the_reverse_lookup_too(task_package, isolated_task_registry):
    """Registered before the scan knew of the clash, the class still answers to no task, by name or by class."""
    task_package(
        "early_clash_pkg",
        _CLASHING_FILES,
    )
    reg = isolated_task_registry
    importlib.import_module("early_clash_pkg.b")  # no scan has run: registers like any import
    reg.add_search_path("early_clash_pkg")
    with pytest.warns(UserWarning, match="'dup' is registered in 2 locations"):
        with pytest.raises(ValueError, match="'dup' is registered in 2 locations"):
            reg.task_for_config_qualname("early_clash_pkg.b.OneConfig")
    with pytest.raises(ValueError, match="'dup' is registered in 2 locations"):
        reg.get("dup")


@pytest.mark.parametrize("imported", ["keep_pkg.k", "plug_pkg.p"], ids=["kept-side", "removed-side"])
def test_a_rescan_that_ends_a_clash_gives_the_name_to_the_class_the_scan_still_finds(
    tmp_path, monkeypatch, task_package, isolated_task_registry, imported
):
    """A set-aside class the rescan still finds is registered, since its import will not run again; one
    whose plugin left is not, imported or not.
    """
    task_package("keep_pkg", {"k.py": _IMPORT + _task_class("dup", "KeepConfig")})
    task_package("plug_pkg", {"p.py": _IMPORT + _task_class("dup", "PlugConfig")})
    _install_distribution(tmp_path, monkeypatch, "plug-robots", {"plug": "plug_pkg"})
    reg = isolated_task_registry
    reg.add_search_path("keep_pkg")
    with pytest.warns(UserWarning, match="'dup' is registered in 2 locations"):
        reg._ensure_discovered()
    importlib.import_module(imported)  # set aside: 'dup' clashes
    assert "dup" not in reg._registry

    (tmp_path / "plug_robots-0.1.dist-info" / "entry_points.txt").write_text("[eden.tasks]\n")  # plugin renamed away
    importlib.invalidate_caches()
    reg._invalidate_discovery_cache()
    assert reg.get("dup") is importlib.import_module("keep_pkg.k").KeepConfig


def test_entry_point_task_reusing_a_built_in_name_fails_only_that_name(tmp_path, monkeypatch, task_package):
    task_package(
        "acme_tasks",
        {"reach.py": _IMPORT + _task_class("reacher", "AcmeReacherConfig")},
    )
    _install_distribution(tmp_path, monkeypatch, "acme-robots", {"acme": "acme_tasks"})
    reg = TaskRegistry()
    with pytest.warns(UserWarning, match="'reacher' is registered in 2 locations"):
        assert "move_forward" in reg
    with pytest.raises(ValueError, match=r"(?s)reacher/config\.py.*acme_tasks/reach\.py"):
        reg.get("reacher")


def test_task_source_names_where_each_task_comes_from(tmp_path, monkeypatch, task_package):
    """``eden list``'s Source column: built-in, entry point, search path, or the module that shadowed one."""
    task_package(
        "plain_tasks",
        {"t.py": _IMPORT + "@TASK_REGISTRY.register()\nclass PlainConfig:\n    pass\n"},
    )
    task_package("acme_tasks", {"reach.py": _ACME_TASK})
    _install_distribution(tmp_path, monkeypatch, "acme-robots", {"acme": "acme_tasks"})
    reg = TaskRegistry()
    reg.add_search_path("plain_tasks")

    assert reg.task_source("reacher") == "eden"
    assert reg.task_source("acme_reach") == "acme-robots"
    assert reg.task_source("plain") == "plain_tasks"
    assert reg.task_source("nowhere_task") is None

    reg.register(_Cfg, name="reacher", override=True)
    assert reg.task_source("reacher") == _Cfg.__module__


def test_unknown_task_message_says_where_an_out_of_tree_task_is_declared():
    with pytest.raises(KeyError, match=r"no_such_task.*project\.entry-points\.\"eden\.tasks\""):
        TaskRegistry().get("no_such_task")
    with pytest.raises(KeyError) as excinfo:
        TaskRegistry().get("reachr")
    assert excinfo.value.args[0].startswith("No task named 'reachr' found in task registry. Did you mean 'reacher'?")


def test_modifier_names_resolves_inheritance_without_importing(task_package):
    """Inherited mods are listed by simple class name across files, through unregistered intermediates."""
    task_package(
        "inherit_tasks_pkg",
        {
            "base.py": _IMPORT
            + "@TASK_REGISTRY.register(name='base_task', modifiers=(FooMod, BarMod()))\nclass BaseConfig:\n    pass\n",
            "mid.py": "from inherit_tasks_pkg.base import BaseConfig\nclass MidConfig(BaseConfig):\n    pass\n",
            "leaf.py": _IMPORT + "import inherit_tasks_pkg.mid as mid\n"
            "@TASK_REGISTRY.register(name='leaf_task')\nclass LeafConfig(mid.MidConfig):\n    pass\n"
            "@TASK_REGISTRY.register\nclass BareConfig(BaseConfig):\n    pass\n"
            "@TASK_REGISTRY.register(name='optout_task', modifiers=())\nclass OptOutConfig(BaseConfig):\n    pass\n"
            "@TASK_REGISTRY.register(name='grand_task')\nclass GrandConfig(OptOutConfig):\n    pass\n"
            "@TASK_REGISTRY.register(name='orphan_task')\nclass OrphanConfig(SomewhereElse):\n    pass\n"
            # MRO: Mix, OptOutMid, OptOut (registered, no mods), Mid, Base -- the opt-out is nearest.
            "class OptOutMidConfig(OptOutConfig):\n    pass\n"
            "@TASK_REGISTRY.register(name='mix_task')\nclass MixConfig(OptOutMidConfig, mid.MidConfig):\n    pass\n",
        },
    )
    reg = TaskRegistry()
    reg.add_search_path("inherit_tasks_pkg")
    assert reg.modifier_names("base_task") == ["FooMod", "BarMod"]
    assert reg.modifier_names("leaf_task") == ["FooMod", "BarMod"]
    assert reg.modifier_names("bare") == ["FooMod", "BarMod"]
    assert reg.modifier_names("optout_task") == []
    assert reg.modifier_names("grand_task") == []
    assert reg.modifier_names("orphan_task") == []
    assert reg.modifier_names("mix_task") == []
    assert not {"inherit_tasks_pkg.base", "inherit_tasks_pkg.leaf"} & set(sys.modules)


# ── modifier inheritance ────────────────────────────────────────────────────
# A config registered without ``modifiers=`` takes the mods of its nearest registered base;
# ``modifiers=()`` opts out.


def test_subclass_registered_without_modifiers_inherits_the_parents():
    reg = _registry(
        (_Cfg, "parent", {"modifiers": (_AddValueMod(delta=3), _SetLabelMod)}),
        (_ChildConfig, "child", {}),
    )
    assert reg.get_modifiers("child") == reg.get_modifiers("parent")
    assert reg.build("child").value == 3
    cfg = reg.build("child", delta=4, label="b")
    assert (cfg.value, cfg.label) == (4, "b")
    assert reg.build("parent").value == 3


@pytest.mark.parametrize(
    ("registrations", "task", "expected_value"),
    [
        # modifiers=() opts out
        ([(_Cfg, "parent", (_AddValueMod(delta=3),)), (_ChildConfig, "child", ())], "child", 0),
        # explicit modifiers replace the inherited ones
        ([(_Cfg, "parent", (_AddValueMod(delta=3),)), (_ChildConfig, "child", (_AddValueMod(delta=5),))], "child", 5),
        # the walk passes through an unregistered intermediate
        ([(_Cfg, "parent", (_AddValueMod(delta=3),)), (_LeafConfig, "leaf", None)], "leaf", 3),
        # a nearer registered base that opted out shadows the one above it
        ([(_Cfg, "parent", (_AddValueMod(delta=3),)), (_MidConfig, "mid", ()), (_LeafConfig, "leaf", None)], "leaf", 0),
        # no registered base, no mods
        ([(_ChildConfig, "child", None)], "child", 0),
        # a second name for one class builds what the first does
        ([(_Cfg, "primary", (_AddValueMod(delta=3),)), (_Cfg, "alias", None)], "alias", 3),
        # override=True with a subclass replaces the class, not the mods that bind it
        ([(_Cfg, "task", (_AddValueMod(delta=3),)), (_ChildConfig, "task", None)], "task", 3),
        # re-registering a class under its own name keeps its mods
        ([(_Cfg, "task", (_AddValueMod(delta=3),)), (_Cfg, "task", None)], "task", 3),
        # shadowing with modifiers=() drops them
        ([(_Cfg, "task", (_AddValueMod(delta=3),)), (_ChildConfig, "task", ())], "task", 0),
        # shadowing with an unrelated class inherits nothing
        ([(_Cfg, "task", (_AddValueMod(delta=3),)), (_UnrelatedConfig, "task", None)], "task", 0),
        # a base registered twice with equal mods is not ambiguous
        (
            [(_Cfg, "a", (_AddValueMod(delta=2),)), (_Cfg, "b", (_AddValueMod(delta=2),)), (_ChildConfig, "c", None)],
            "c",
            2,
        ),
        # a replaced name takes the mods of the registration it replaces, not a sibling name's
        (
            [
                (_Cfg, "easy", (_AddValueMod(delta=1),)),
                (_Cfg, "hard", (_AddValueMod(delta=9),)),
                (_ChildConfig, "hard", None),
            ],
            "hard",
            9,
        ),
    ],
    ids=[
        "opt-out",
        "explicit-replaces",
        "through-unregistered",
        "nearest-base-wins",
        "no-base",
        "alias",
        "shadowing-subclass",
        "reregister-self",
        "shadow-opt-out",
        "shadow-unrelated",
        "equal-duplicates",
        "shadow-one-of-several",
    ],
)
def test_modifier_inheritance(registrations, task, expected_value):
    reg = TaskRegistry()
    for cls, name, modifiers in registrations:
        reg.register(cls, name=name, override=True, modifiers=modifiers)
    assert reg.build(task).value == expected_value


def test_a_shadowing_subclass_inherits_the_same_with_or_without_an_alias():
    with_alias, without_alias = TaskRegistry(), TaskRegistry()
    for reg in (with_alias, without_alias):
        reg.register(_Cfg, name="task", modifiers=(_AddValueMod(delta=3),))
    with_alias.register(_ChildConfig, name="child")
    for reg in (with_alias, without_alias):
        reg.register(_ChildConfig, name="task", override=True)
    assert with_alias.build("task").value == without_alias.build("task").value == 3


@pytest.mark.parametrize(
    ("aliases", "shadow"),
    [
        ([], _ChildConfig),
        ([(_ChildConfig, "child")], _ChildConfig),
        ([(_ChildConfig, "child")], _GrandChildConfig),
        ([(_ChildConfig, "child"), (_GrandChildConfig, "grandchild")], _GrandChildConfig),
    ],
    ids=["direct", "aliased-self", "through-aliased-parent", "through-two-aliased-levels"],
)
def test_a_shadowing_subclass_keeps_the_replaced_mods_when_the_shadowed_class_has_other_names(aliases, shadow):
    """Every name for each aliased class inherits, so the walk goes on from its base for them."""
    reg = _registry((_Cfg, "task", {"modifiers": (_AddValueMod(delta=3),)}))
    for cls, name in aliases:
        reg.register(cls, name=name)
    reg.register(_Cfg, name="other", modifiers=(_AddValueMod(delta=9),))
    reg.register(shadow, name="task", override=True)
    assert reg.build("task").value == 3
    reg.register(_Cfg, name="other", override=True, modifiers=(_AddValueMod(delta=7),))
    assert reg.build("task").value == 3  # a snapshot: the registration it came from is gone


def test_a_class_in_between_that_pins_its_own_mods_wins_over_the_replaced_registration():
    """``_ChildConfig`` is the nearest registered base, so the shadow takes its mods, not the replaced task's."""
    reg = _registry(
        (_Cfg, "task", {"modifiers": (_AddValueMod(delta=3),)}),
        (_ChildConfig, "child", {"modifiers": (_AddValueMod(delta=5),)}),
    )
    reg.register(_GrandChildConfig, name="task", override=True)
    assert reg.build("task").value == 5
    reg.register(_ChildConfig, name="child", override=True, modifiers=(_AddValueMod(delta=7),))
    assert reg.build("task").value == 7  # live: ``child`` still answers for those mods


def test_an_alias_of_the_shadowing_class_takes_the_mods_the_shadow_kept():
    reg = _registry(
        (_Cfg, "task", {"modifiers": (_AddValueMod(delta=3),)}),
        (_ChildConfig, "child", {}),
        (_Cfg, "other", {"modifiers": (_AddValueMod(delta=9),)}),
        (_ChildConfig, "task", {"override": True}),
    )
    assert reg.build("child").value == 3  # a second name for one class builds what the first does


def test_shadowing_an_unrelated_task_through_an_inheriting_parent_still_raises_on_the_ambiguity():
    """Only the registration ``override=True`` replaces may settle it, and that one is not a ``_Cfg``."""
    reg = _registry(
        (_Cfg, "one", {"modifiers": (_AddValueMod(delta=1),)}),
        (_ChildConfig, "child", {}),
        (_Cfg, "two", {"modifiers": (_AddValueMod(delta=2),)}),
        (_UnrelatedConfig, "task", {"modifiers": (_AddValueMod(delta=5),)}),
    )
    with pytest.raises(ValueError, match=r"(?s)'task' \('_GrandChildConfig'\).*_Cfg.*'one'.*'two'.*modifiers="):
        reg.register(_GrandChildConfig, name="task", override=True)
    assert reg.get("task") is _UnrelatedConfig
    assert reg.build("task").value == 5


@pytest.mark.parametrize("aliased", [False, True], ids=["direct", "through-aliased-parent"])
def test_a_shadowed_task_that_inherits_its_own_mods_leaves_the_shadow_following_the_class(aliased):
    """The shadowed ``task`` answers with what ``_ChildConfig`` holds, which ``other`` still holds after it."""
    reg = _registry(
        (_Cfg, "root", {"modifiers": (_AddValueMod(delta=3),)}),
        (_ChildConfig, "task", {}),
        (_ChildConfig, "other", {"modifiers": (_AddValueMod(delta=9),)}),
    )
    if aliased:
        reg.register(_GrandChildConfig, name="grandchild")
    reg.register(_GrandChildConfig, name="task", override=True)
    assert reg.build("task").value == 9
    reg.register(_ChildConfig, name="other", override=True, modifiers=(_AddValueMod(delta=7),))
    assert reg.build("task").value == 7  # live: ``other`` still answers for those mods


def test_inheriting_from_a_base_registered_with_conflicting_modifiers_raises():
    reg = _registry(
        (_Cfg, "easy", {"modifiers": (_AddValueMod(delta=1),)}), (_Cfg, "hard", {"modifiers": (_AddValueMod(delta=9),)})
    )
    with pytest.raises(ValueError, match=r"(?s)'child'.*_Cfg.*easy.*hard.*modifiers="):
        reg.register(_ChildConfig, name="child")
    assert "child" not in reg
    reg.register(_ChildConfig, name="child", modifiers=())  # choosing explicitly is the way out


# ── inheritance is live, not frozen at registration ─────────────────────────


def test_reregistering_a_parent_updates_the_inheritance_chain():
    reg = _registry(
        (_Cfg, "root", {"modifiers": (_AddValueMod(delta=3),)}),
        (_ChildConfig, "mid", {}),
        (_GrandChildConfig, "leaf", {}),
    )
    assert reg.build("leaf").value == 3
    reg.register(_Cfg, name="root", override=True, modifiers=(_AddValueMod(delta=7),))
    assert reg.build("mid").value == reg.build("leaf").value == 7


def test_inherited_modifiers_do_not_depend_on_registration_order():
    before, after = TaskRegistry(), TaskRegistry()
    for reg in (before, after):
        reg.register(_Cfg, name="task", modifiers=(_AddValueMod(delta=3),))
    before.register(_GrandChildConfig, name="leaf")
    for reg in (before, after):
        reg.register(_ChildConfig, name="task", override=True, modifiers=(_AddValueMod(delta=7),))
    after.register(_GrandChildConfig, name="leaf")
    assert before.build("leaf").value == after.build("leaf").value == 7


def test_a_subclass_registered_before_its_parent_still_inherits():
    reg = _registry((_ChildConfig, "child", {}))
    assert reg.get_modifiers("child") == ()
    reg.register(_Cfg, name="parent", modifiers=(_AddValueMod(delta=3),))
    assert reg.build("child").value == 3


def test_explicit_modifiers_survive_a_reregistered_parent():
    reg = _registry(
        (_Cfg, "parent", {"modifiers": (_AddValueMod(delta=3),)}),
        (_ChildConfig, "pinned", {"modifiers": (_AddValueMod(delta=5),)}),
        (_ChildConfig, "opted_out", {"modifiers": ()}),
    )
    reg.register(_Cfg, name="parent", override=True, modifiers=(_AddValueMod(delta=7),))
    assert reg.build("pinned").value == 5
    assert reg.build("opted_out").value == 0


def test_a_shadowing_subclass_keeps_its_snapshot_when_the_registry_moves_on():
    """The one frozen case: the registration it inherited from is gone."""
    reg = _registry(
        (_Cfg, "task", {"modifiers": (_AddValueMod(delta=3),)}),
        (_ChildConfig, "task", {"override": True}),
        (_Cfg, "elsewhere", {"modifiers": (_AddValueMod(delta=9),)}),
    )
    assert reg.build("task").value == 3


def test_a_replacement_that_inherits_from_a_live_base_follows_it():
    reg = _registry(
        (_Cfg, "base", {"modifiers": (_AddValueMod(delta=3),)}),
        (_UnrelatedConfig, "task", {"modifiers": (_AddValueMod(delta=5),)}),
        (_ChildConfig, "task", {"override": True}),
    )
    assert reg.build("task").value == 3
    reg.register(_Cfg, name="base", override=True, modifiers=(_AddValueMod(delta=7),))
    assert reg.build("task").value == 7


def test_two_names_for_one_class_both_inheriting_resolve_without_recursing_forever():
    reg = _registry(
        (_Cfg, "parent", {"modifiers": (_AddValueMod(delta=3),)}), (_ChildConfig, "a", {}), (_ChildConfig, "b", {})
    )
    assert reg.build("a").value == reg.build("b").value == 3
    reg.register(_UnrelatedConfig, name="parent", override=True, modifiers=())
    assert reg.get_modifiers("a") == reg.get_modifiers("b") == ()


def test_a_name_that_inherits_prefers_a_pinned_name_for_the_same_class():
    reg = _registry(
        (_Cfg, "base", {"modifiers": (_AddValueMod(delta=3),)}),
        (_ChildConfig, "pinned", {"modifiers": (_AddValueMod(delta=5),)}),
        (_ChildConfig, "alias", {}),
    )
    assert reg.build("alias").value == 5
    reg.register(_ChildConfig, name="pinned", override=True, modifiers=(_AddValueMod(delta=9),))
    assert reg.build("alias").value == 9


def test_modifier_names_follows_a_reregistered_parent():
    reg = _registry((_Cfg, "parent", {"modifiers": (_AddValueMod(delta=3),)}), (_ChildConfig, "child", {}))
    assert reg.modifier_names("child") == ["_AddValueMod"]
    reg.register(_Cfg, name="parent", override=True, modifiers=(_SetLabelMod(),))
    assert reg.modifier_names("child") == ["_SetLabelMod"]
    assert [type(m).__name__ for m in reg.get_modifiers("child")] == ["_SetLabelMod"]


def test_a_base_made_ambiguous_after_the_fact_raises_on_use_but_not_in_the_listing():
    reg = _registry((_Cfg, "easy", {"modifiers": (_AddValueMod(delta=1),)}), (_ChildConfig, "child", {}))
    assert reg.build("child").value == 1

    reg.register(_Cfg, name="hard", modifiers=(_AddValueMod(delta=9),))
    with pytest.raises(ValueError, match=r"(?s)'child'.*_Cfg.*easy.*hard.*modifiers="):
        reg.get_modifiers("child")
    assert reg.modifier_names("child") == []  # ``eden list`` still renders


def test_the_ambiguity_names_the_task_being_resolved_not_a_sibling_that_inherits():
    reg = _registry(
        (_ChildConfig, "one", {"modifiers": (_AddValueMod(delta=1),)}),
        (_ChildConfig, "alias_a", {}),
        (_ChildConfig, "alias_b", {}),
        (_ChildConfig, "two", {"modifiers": (_AddValueMod(delta=2),)}),
    )
    with pytest.raises(ValueError, match=r"(?s)'alias_a'.*'one'.*'two'.*modifiers="):
        reg.get_modifiers("alias_a")


@pytest.mark.parametrize(
    ("asked", "cls_name"), [("alias_a", "_ChildConfig"), ("alias_b", "_ChildConfig"), ("leaf", "_GrandChildConfig")]
)
def test_the_ambiguity_names_the_task_being_resolved_when_every_name_for_its_class_inherits(asked, cls_name):
    """No name for ``_ChildConfig`` pins its own mods, so the group resolves through a stand-in."""
    reg = _registry(
        (_Cfg, "one", {"modifiers": (_AddValueMod(delta=1),)}),
        (_ChildConfig, "alias_a", {}),
        (_ChildConfig, "alias_b", {}),
        (_GrandChildConfig, "leaf", {}),
        (_Cfg, "two", {"modifiers": (_AddValueMod(delta=2),)}),
    )
    with pytest.raises(ValueError) as excinfo:
        reg.get_modifiers(asked)
    message = str(excinfo.value)
    assert message.startswith(f"Task '{asked}' ('{cls_name}') would inherit the modifiers of _Cfg")
    assert [name for name in ("alias_a", "alias_b", "leaf") if f"'{name}'" in message] == [asked]


def test_the_ambiguity_lists_the_values_that_make_the_modifiers_differ():
    reg = _registry(
        (_Cfg, "one", {"modifiers": (_AddValueMod(delta=1), _SetLabelMod(label="b"))}),
        (_ChildConfig, "child", {}),
        (_Cfg, "two", {"modifiers": (_AddValueMod(delta=2),)}),
    )
    with pytest.raises(ValueError) as excinfo:
        reg.get_modifiers("child")
    assert "{'one': ['_AddValueMod(delta=1)', \"_SetLabelMod(label='b')\"], 'two': ['_AddValueMod(delta=2)']}" in str(
        excinfo.value
    )


def test_the_ambiguity_message_survives_a_modifier_value_whose_repr_raises():
    class _Unprintable:
        def __repr__(self) -> str:
            raise RuntimeError("repr of a freed handle")

    class _HandleMod(TaskMod):
        def __init__(self) -> None:
            self.handle = _Unprintable()

        def apply(self, config):
            return config

    reg = _registry(
        (_Cfg, "one", {"modifiers": (_HandleMod(),)}),
        (_ChildConfig, "child", {}),
        (_Cfg, "two", {"modifiers": (_HandleMod(),)}),
    )
    with pytest.raises(
        ValueError, match=r"'child'.*'one': \['_HandleMod\(handle=<_Unprintable whose __repr__ raised>\)"
    ):
        reg.get_modifiers("child")


class _ItemsMod(TaskMod):
    def __init__(self, items: object = ()):
        super().__init__()
        self.items = items

    def apply(self, config):
        return config


@pytest.mark.parametrize(
    ("one", "two"),
    [
        ("/data/motions/lafan1/subject1/walk_01.npz", "/data/motions/amass_v2/subject1/walk_01.npz"),
        ([0, 1, 2, 3, 4, 5, 6], [0, 1, 2, 3, 4, 5, 99]),
    ],
)
def test_the_ambiguity_prints_values_that_differ_past_a_short_repr_in_full(one, two):
    """A middle-elided path or a list cut after six items would print the two sets alike again."""
    reg = _registry(
        (_Cfg, "one", {"modifiers": (_ItemsMod(one),)}),
        (_ChildConfig, "child", {}),
        (_Cfg, "two", {"modifiers": (_ItemsMod(two),)}),
    )
    with pytest.raises(ValueError) as excinfo:
        reg.get_modifiers("child")
    assert f"_ItemsMod(items={one!r})" in str(excinfo.value)
    assert f"_ItemsMod(items={two!r})" in str(excinfo.value)


def test_the_ambiguity_elides_the_middle_of_a_huge_value():
    reg = _registry(
        (_Cfg, "one", {"modifiers": (_ItemsMod("a" * 5000),)}),
        (_ChildConfig, "child", {}),
        (_Cfg, "two", {"modifiers": (_ItemsMod("b" * 5000),)}),
    )
    with pytest.raises(ValueError, match=r"'child'.*chars elided") as excinfo:
        reg.get_modifiers("child")
    assert len(str(excinfo.value)) < 5000


@pytest.mark.parametrize("registered", ["leaf", "alias"])
def test_a_registration_that_inherits_an_ambiguity_names_itself_and_its_own_class(registered):
    """``leaf`` resolves through ``alias`` as a stand-in; ``alias`` replaces an entry made for another class."""
    reg = _registry(
        (_Cfg, "one", {"modifiers": (_AddValueMod(delta=1),)}),
        (_ChildConfig, "alias", {}),
        (_Cfg, "two", {"modifiers": (_AddValueMod(delta=2),)}),
    )
    with pytest.raises(ValueError, match=rf"^Task '{registered}' \('_GrandChildConfig'\) would inherit"):
        reg.register(_GrandChildConfig, name=registered, override=registered == "alias")


def test_a_registration_that_raises_leaves_the_replaced_one_in_place():
    """Nothing is written until both walks are done, so a raise cannot half-apply a replacement."""

    class _Incomparable:
        def __eq__(self, other: object) -> bool:
            raise RuntimeError("Boolean value of Tensor with more than one element is ambiguous")

        __hash__ = None  # type: ignore[assignment]

    class _TensorLikeMod(TaskMod):
        def __init__(self) -> None:
            self.blob = _Incomparable()

        def apply(self, config):
            return config

    replaced = _TensorLikeMod()
    reg = _registry((_Cfg, "base", {"modifiers": (_TensorLikeMod(),)}), (_Cfg, "task", {"modifiers": (replaced,)}))
    with pytest.raises(RuntimeError, match="ambiguous"):
        reg.register(_ChildConfig, name="task", override=True)
    assert reg.get("task") is _Cfg
    assert reg.get_modifiers("task") == (replaced,)


def test_many_names_for_one_class_resolve_in_linear_time():
    """Resolving aliases through each other costs k! walks — ~10 s at k=11, a hang soon after."""
    reg = _registry((_Cfg, "parent", {"modifiers": (_AddValueMod(delta=3),)}))
    for i in range(11):
        reg.register(_ChildConfig, name=f"alias{i}")
    start = time.perf_counter()
    assert reg.build("alias0").value == 3
    assert time.perf_counter() - start < 2.0


@pytest.mark.parametrize("task_name", sorted(TASK_REGISTRY.list_tasks()))
def test_no_task_mod_default_points_into_the_source_checkout(task_name):
    """``Path(eden.__file__).parent.parent`` is ``site-packages/`` once Eden is installed.

    Under the package (``eden/assets``), outside the repo, or relative is fine.
    """
    package_dir = pathlib.Path(eden.__file__).resolve().parent
    repo_root = package_dir.parent
    for mod in TASK_REGISTRY.get_modifiers(task_name):
        for field_name, _annotation, default, _help in cli_fields(type(mod)):
            if not isinstance(default, str) or not default:
                continue
            path = pathlib.Path(default)
            if path.is_absolute() and path.is_relative_to(repo_root):
                assert path.is_relative_to(package_dir), (
                    f"{task_name}: {type(mod).__name__}.{field_name} defaults to {path.relative_to(repo_root)}, which "
                    "ships only in a source checkout. Fetch it through the dataset/asset HuggingFace mechanism, or "
                    "default to a per-user cache path, instead of deriving one from the repo layout."
                )
