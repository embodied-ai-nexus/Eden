"""Tests for ``eden.utils.registry``: keys, duplicates, lookup misses, defaults and lazy population."""

from __future__ import annotations

import importlib
import sys
import threading
import time
import warnings

import pytest

from eden.managers.base import ManagerTermBase, ManagerTermFuncWrapperBase
from eden.options.options import ConfigurableOptions
from eden.utils.common import ConfigurableFuncWrapperMixin, ConfigurableMixin
from eden.utils.registry import Registry, TermRegistry, VariantRegistry


class _Opts(ConfigurableOptions):
    model_config = {"extra": "allow"}

    weight: float = 1.0


class _Term(ManagerTermBase, ConfigurableMixin[_Opts]):
    weight: float = 1.0

    def compute(self):
        return None


def _term_registry(name: str = "TERM") -> tuple[TermRegistry, type]:
    """A registry over a throwaway term kind, with a class term base and a function wrapper."""

    class Wrapper(ManagerTermFuncWrapperBase, ConfigurableFuncWrapperMixin[_Opts]):
        weight: float = 1.0

    return TermRegistry(name, _Term, Wrapper), Wrapper


class Configurable(ConfigurableMixin[_Opts]):
    pass


# ---------------------------------------------------------------------------
# Keys
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("form", ["direct", "decorator"])
@pytest.mark.parametrize("cls_name, key", [("MyTestClass", "my_test_class"), ("HTTPServer", "http_server")])
def test_register_derives_the_snake_case_key_and_returns_the_object(form, cls_name, key):
    registry = Registry("TEST")
    cls = type(cls_name, (), {})

    returned = registry.register(cls) if form == "direct" else registry.register()(cls)

    assert returned is cls  # `Foo = REGISTRY.register(Foo)` used to rebind Foo to None
    assert registry.get(key) is cls


def test_an_explicit_name_replaces_the_derived_key():
    registry = Registry("TEST")
    registry.register(type("MyClass", (), {}), name="custom_name")

    assert "custom_name" in registry
    assert "my_class" not in registry


def test_keys_lists_every_registered_key():
    registry = Registry("TEST")
    registry.register(type("Class1", (), {}), name="class1")
    registry.register(type("Class2", (), {}))

    assert set(registry.keys()) == {"class1", "class2"}


@pytest.mark.parametrize(
    "obj",
    [pytest.param(str, id="builtin-type"), pytest.param(lambda: "x", id="function"), pytest.param(object(), id="obj")],
)
def test_register_stores_any_object_verbatim(obj):
    registry = Registry("TYPES")
    registry.register(obj, name="thing")
    assert registry.get("thing") is obj


def test_a_derived_key_starting_with_an_underscore_warns_and_an_explicit_one_does_not():
    registry = Registry("TEST")

    with pytest.warns(UserWarning, match="_internal_thing"):
        registry.register(type("_InternalThing", (), {}))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        registry.register(type("_Other", (), {}), name="_other")

    assert "_internal_thing" in registry and "_other" in registry


@pytest.mark.parametrize("populated", [False, True], ids=["empty", "populated"])
def test_repr_and_str_name_the_registry_and_its_keys(populated):
    registry = Registry("COMMANDS")
    if populated:
        registry.register(type("RunCommand", (), {}), name="run")

    for rendered in (repr(registry), str(registry)):
        assert "COMMANDS" in rendered
        assert ("run" if populated else "empty") in rendered


# ---------------------------------------------------------------------------
# Duplicates
# ---------------------------------------------------------------------------


def test_a_duplicate_key_raises_naming_the_registry_both_objects_and_the_fix():
    """A ``ValueError``, not an ``assert``: under ``python -O`` the second registration silently won."""
    registry = Registry("TEST")
    registry.register(type("TestClass", (), {}), name="test")

    with pytest.raises(ValueError) as exc_info:
        registry.register(type("AnotherClass", (), {}), name="test")

    msg = str(exc_info.value)
    assert "'test'" in msg and "'TEST' registry" in msg
    assert "TestClass" in msg and "AnotherClass" in msg
    assert "give one of them a different name=" in msg


def test_a_duplicate_configurable_is_told_to_rename_not_to_pass_name():
    """A configurable's key is its own name, so ``name=`` is no fix — on a term registry or a generic one."""
    term_registry, _ = _term_registry()

    class Collide(_Term):
        pass

    generic_registry = Registry("KIND")
    term_registry.register(Collide)
    generic_registry.register(Configurable)

    for registry, obj in ((term_registry, Collide), (generic_registry, Configurable)):
        with pytest.raises(ValueError) as exc_info:
            registry.register(obj)
        msg = str(exc_info.value)
        assert "Rename one of the two" in msg and "no name= to separate them" in msg
        assert "give one of them a different name=" not in msg


def test_a_variant_collision_names_the_arguments_that_built_the_key():
    registry = VariantRegistry("ASSETS")
    registry.register(type("Bottle", (), {}), base_name="bottle", variant_list=[0])

    with pytest.raises(ValueError) as exc_info:
        registry.register(type("Flask", (), {}), base_name="bottle", variant_list=[0])

    assert "base_name" in str(exc_info.value)


def test_a_duplicate_function_term_is_named_not_repr_ed():
    registry, _ = _term_registry()

    def colliding_func_term(env):
        return None

    registry.register(colliding_func_term)
    with pytest.raises(ValueError) as exc_info:
        registry.register(colliding_func_term)

    assert "colliding_func_term" in str(exc_info.value)
    assert "functools.partial" not in str(exc_info.value)


def test_a_func_wrapper_class_belongs_to_one_registry():
    class Wrapper(ManagerTermFuncWrapperBase):
        pass

    TermRegistry("REG_ONE", _Term, Wrapper)
    with pytest.raises(ValueError, match="already bound to registry"):
        TermRegistry("REG_TWO", _Term, Wrapper)


# ---------------------------------------------------------------------------
# A configurable's key is derived, never overridden
# ---------------------------------------------------------------------------


def test_the_key_a_term_registers_under_is_the_name_configure_stamps():
    registry, Wrapper = _term_registry()

    @registry.register()
    class MyClassTerm(_Term):
        pass

    @registry.register()
    def my_func_term(env):
        return None

    assert registry.get(MyClassTerm.configure().name) is MyClassTerm
    assert registry.get(Wrapper.configure(func=my_func_term).name).keywords["func"] is my_func_term


@pytest.mark.parametrize("form", ["class-decorator", "func-decorator", "class-direct", "func-direct"])
def test_a_term_registry_refuses_any_name(form):
    registry, _ = _term_registry()

    class MyClassTerm(_Term):
        pass

    def my_func_term(env):
        return None

    obj = MyClassTerm if form.startswith("class") else my_func_term
    with pytest.raises(ValueError, match="takes no name="):
        if form.endswith("decorator"):
            registry.register(name="an_alias")(obj)
        else:
            registry.register(obj, name="an_alias")

    assert "an_alias" not in registry and "my_class_term" not in registry and "my_func_term" not in registry
    # `**term_defaults` would otherwise have swallowed `name` as a class attribute shadowing the key property.
    assert not isinstance(getattr(_Term, "name", None), str)


def test_a_generic_registry_refuses_a_name_only_for_a_configurable():
    """Assets, robots and tasks keep ``name=`` (``beyondmimic``); a file handler or dataset must not alias."""
    registry = Registry("MIXED")

    with pytest.raises(ValueError, match="its key is always"):
        registry.register(Configurable, name="an_alias")

    plain = type("NotConfigurable", (), {})
    registry.register(plain, name="an_alias")
    assert registry.get("an_alias") is plain


def test_both_refusals_give_the_same_reason():
    """One rule, two entry points: the shared constant is the contract they must both quote."""
    from eden.utils.registry import _WHY_THE_KEY_IS_THE_NAME

    registry, _ = _term_registry()
    with pytest.raises(ValueError) as term_exc:
        registry.register(_Term, name="an_alias")
    with pytest.raises(ValueError) as generic_exc:
        Registry("KIND").register(Configurable, name="an_alias")

    term_msg, generic_msg = str(term_exc.value), str(generic_exc.value)
    assert _WHY_THE_KEY_IS_THE_NAME in term_msg and _WHY_THE_KEY_IS_THE_NAME in generic_msg
    assert "takes no name=" in term_msg and "takes no name=" not in generic_msg
    assert "'configurable'" in generic_msg and "'configurable'" not in term_msg


# ---------------------------------------------------------------------------
# Registration defaults
# ---------------------------------------------------------------------------


def test_function_term_defaults_land_on_its_own_wrapper():
    registry, Wrapper = _term_registry()

    @registry.register(weight=3.0)
    def metric_a(env):
        return None

    @registry.register()
    def metric_b(env):
        return None

    assert registry.get("metric_a").func.weight == 3.0
    assert registry.get("metric_b").func.weight == 1.0
    assert Wrapper.weight == 1.0


def test_registered_class_is_the_class_a_term_is_built_as():
    """For a function term that is its own wrapper subclass, the one place its defaults live."""
    registry, Wrapper = _term_registry()

    @registry.register(weight=3.0)
    def metric_a(env):
        return None

    registered = registry.register(weight=2.0)(type("TermA", (_Term,), {}))

    assert registry.registered_class("metric_a") is registry.get("metric_a").func
    assert (registry.registered_class("metric_a").weight, Wrapper.weight) == (3.0, 1.0)
    assert registry.registered_class("term_a") is registered
    with pytest.raises(KeyError, match="metric_b"):
        registry.registered_class("metric_b")


def test_class_term_defaults_reach_configure_through_an_isolated_subclass():
    """The source class is unmutated and still configures to the field default."""
    registry, _ = _term_registry()

    class TermA(_Term):
        pass

    class TermB(_Term):
        pass

    registered_a = registry.register(TermA, weight=2.0)
    registered_b = registry.register(weight=3.0)(TermB)

    assert registered_a is registry.get("term_a") and registered_b is registry.get("term_b")
    assert (registered_a.configure().weight, registered_b.configure().weight) == (2.0, 3.0)
    assert TermA.configure().weight == TermB.configure().weight == 1.0
    assert TermA.weight == TermB.weight == _Term.weight == 1.0


def test_the_defaults_subclass_keeps_the_source_identity():
    registry, _ = _term_registry()

    class Decorated(_Term):
        pass

    registered = registry.register(weight=2.0)(Decorated)

    assert registered is not Decorated and issubclass(registered, Decorated)
    assert (registered.__name__, registered.__qualname__, registered.__module__) == (
        Decorated.__name__,
        Decorated.__qualname__,
        Decorated.__module__,
    )
    assert repr(registered) == repr(Decorated)
    assert registered.configure().name == Decorated.configure().name == "decorated"


def test_a_registration_without_defaults_registers_the_class_itself():
    registry, _ = _term_registry()

    class Plain(_Term):
        pass

    registry.register(Plain)
    assert registry.get("plain") is Plain


def test_a_func_wrapper_subclass_with_defaults_is_isolated_too():
    registry, Wrapper = _term_registry()

    registered = registry.register(Wrapper, weight=4.0)

    assert issubclass(registered, Wrapper) and registered is not Wrapper
    assert (registered.weight, Wrapper.weight) == (4.0, 1.0)


# ---------------------------------------------------------------------------
# Lookup misses
# ---------------------------------------------------------------------------


def test_a_miss_names_the_key_the_registry_and_the_roster():
    registry = Registry("TEST")
    registry.register(type("Option1", (), {}), name="option1")

    with pytest.raises(KeyError) as exc_info:
        registry.get("non_existent")

    message = exc_info.value.args[0]
    assert "non_existent" in message and "TEST" in message and "option1" in message
    assert "non_existent" not in registry


def test_a_near_miss_suggests_the_closest_key():
    registry = Registry("TEST")
    registry.register(type("BaseHeight", (), {}))

    with pytest.raises(KeyError, match="Did you mean 'base_height'"):
        registry.get("base_hight")


def test_a_term_registry_miss_points_at_the_catalog_search():
    from eden.managers import OBSERVATION_TERM_REGISTRY

    with pytest.raises(KeyError) as excinfo:
        OBSERVATION_TERM_REGISTRY.get("no_such_observation")

    message = excinfo.value.args[0]
    assert "eden terms --kind observation --grep <text>" in message
    assert "\n" not in message  # KeyError.__str__ is repr(args[0]): a newline would render escaped


def test_a_plain_registry_miss_does_not_point_at_eden_terms():
    registry = Registry("TEST")
    registry.register(type("Thing", (), {}))

    with pytest.raises(KeyError) as excinfo:
        registry.get("missing")

    assert "eden terms" not in excinfo.value.args[0]


def test_a_non_string_key_still_gets_the_readable_miss():
    """A term entry with no ``name=`` arrives as ``None``; the suggestion must not raise a TypeError."""
    registry = Registry("TEST")
    registry.register(type("BaseHeight", (), {}))

    with pytest.raises(KeyError) as excinfo:
        registry.get(None)

    message = excinfo.value.args[0]
    assert "No object named 'None'" in message and "base_height" in message
    assert "Did you mean" not in message


def test_every_term_registry_cli_kind_is_an_eden_terms_kind():
    from eden.managers.catalog import KIND_NAMES, KINDS

    for kind in KINDS:
        assert kind.registry().cli_kind == kind.name
    assert set(KIND_NAMES) == {kind.registry().cli_kind for kind in KINDS}


# ---------------------------------------------------------------------------
# Lazy population (``populated_by``)
# ---------------------------------------------------------------------------


@pytest.fixture
def make_package(tmp_path, monkeypatch):
    """Write a package whose ``__init__`` declares ``REGISTRY`` populated by its ``terms`` module.

    Returns ``make(prefix, terms_body, init_extra="")`` -> the imported package. Real files on disk,
    because what is under test is the ``importlib.import_module`` the registry triggers.
    """
    monkeypatch.syspath_prepend(str(tmp_path))
    names: list[str] = []

    def make(prefix: str, terms_body: str, init_extra: str = ""):
        name = f"_eden_{prefix}_{tmp_path.name.replace('-', '_')}"
        pkg = tmp_path / name
        pkg.mkdir()
        (pkg / "__init__.py").write_text(
            f"from eden.utils.registry import Registry\n{init_extra}"
            f"REGISTRY = Registry('{prefix.upper()}', populated_by=('{name}.terms',))\n"
        )
        (pkg / "terms.py").write_text(terms_body.format(pkg=name))
        names.append(name)
        return importlib.import_module(name)

    yield make

    for module in [key for key in sys.modules if any(key == n or key.startswith(f"{n}.") for n in names)]:
        del sys.modules[module]


_PROBE_TERMS = "from {pkg} import IMPORTS, REGISTRY\nIMPORTS.append('imported')\n@REGISTRY.register()\nclass ProbeTerm:\n    pass\n"


@pytest.fixture
def probe(make_package):
    """A package whose term module appends to ``IMPORTS`` when imported, and registers ``probe_term``."""
    return make_package("probe", _PROBE_TERMS, init_extra="IMPORTS = []\n")


def test_nothing_is_imported_until_the_registry_is_read(probe):
    assert probe.IMPORTS == []


@pytest.mark.parametrize(
    "finds_the_term",
    [
        pytest.param(lambda registry: registry.get("probe_term") is not None, id="get"),
        pytest.param(lambda registry: "probe_term" in list(registry.keys()), id="keys"),
        pytest.param(lambda registry: "probe_term" in registry, id="contains"),
        pytest.param(lambda registry: "probe_term" in [key for key, _ in registry], id="iter"),
        pytest.param(lambda registry: "probe_term" in repr(registry), id="repr"),
    ],
)
def test_every_read_path_populates_once(probe, finds_the_term):
    """Parametrized, not chained: the first read populates for good, which would hide a skipped path."""
    assert finds_the_term(probe.REGISTRY)
    assert finds_the_term(probe.REGISTRY)
    assert probe.IMPORTS == ["imported"]


def test_registering_into_a_cold_registry_catches_the_duplicate_and_blames_the_newcomer(probe):
    """A write is a read of the key space; the built-in incumbent must not be named as the offender."""

    class ProbeTerm:
        pass

    with pytest.raises(ValueError, match="cannot register") as caught:
        probe.REGISTRY.register(ProbeTerm)

    offender, _, incumbent = str(caught.value).partition("that key is taken by")
    assert "test_registry" in offender
    assert f"{probe.__name__}.terms.ProbeTerm" in incumbent
    # The populating import ran to completion, so the rejection did not poison later reads.
    assert probe.REGISTRY.get("probe_term") is not None
    assert probe.IMPORTS == ["imported"]


def test_a_module_still_executing_is_not_memoized_as_complete(make_package):
    """A term module imported directly registers through ``_populate`` into itself, half-built."""
    terms = (
        "from {pkg} import REGISTRY, SNAPSHOTS\n"
        "@REGISTRY.register()\nclass First:\n    pass\n"
        "SNAPSHOTS.append((REGISTRY._populated, len(REGISTRY._obj_map)))\n"
        "@REGISTRY.register()\nclass Second:\n    pass\n"
        "SNAPSHOTS.append((REGISTRY._populated, len(REGISTRY._obj_map)))\n"
    )
    package = make_package("partial", terms, init_extra="SNAPSHOTS = []\n")
    importlib.import_module(f"{package.__name__}.terms")

    # `_populated` / `_obj_map` are what a concurrent reader would act on mid-import: no public read
    # can be made from inside the module body without itself triggering population.
    assert package.SNAPSHOTS == [(False, 1), (False, 2)]
    assert package.REGISTRY.get("second") is not None
    assert package.REGISTRY._populated


def test_a_populating_module_may_read_its_own_registry_without_recursing(make_package):
    terms = (
        "from {pkg} import REGISTRY, SEEN\n"
        "@REGISTRY.register()\nclass First:\n    pass\n"
        "SEEN.extend(REGISTRY.keys())\n"
        "@REGISTRY.register()\nclass Second:\n    pass\n"
    )
    package = make_package("reentrant", terms, init_extra="SEEN = []\n")

    assert package.REGISTRY.get("second") is not None
    assert package.SEEN == ["first"]


def test_a_failed_population_is_raised_again_on_every_read(make_package):
    package = make_package("broken", "raise RuntimeError('term module is broken')\n")

    for _ in range(2):
        with pytest.raises(RuntimeError, match="term module is broken"):
            package.REGISTRY.get("anything")


def test_a_module_that_registers_before_it_fails_keeps_reporting_its_own_error(make_package):
    """A retry re-runs the body against its own earlier registrations; that must not read as a duplicate."""
    terms = "from {pkg} import REGISTRY\n@REGISTRY.register()\nclass Alpha:\n    pass\nraise RuntimeError('boom')\n"
    package = make_package("half", terms)

    for attempt in range(3):
        with pytest.raises(RuntimeError, match="boom") as excinfo:
            package.REGISTRY.get("alpha")
        assert "that key is taken by" not in str(excinfo.value), f"read {attempt} inverted the blame"
    assert any("'HALF' registry" in note for note in excinfo.value.__notes__)


def test_replaying_the_failure_does_not_grow_its_traceback(make_package):
    def depth(traceback):
        total = 0
        while traceback:
            total, traceback = total + 1, traceback.tb_next
        return total

    package = make_package("tbgrowth", "raise RuntimeError('still broken')\n")
    depths = []
    for _ in range(12):
        with pytest.raises(RuntimeError) as excinfo:
            package.REGISTRY.get("anything")
        depths.append(depth(excinfo.value.__traceback__))

    assert len(set(depths[1:])) == 1, depths
    assert depths[-1] <= depths[0] + 2, depths
    assert any("terms.py" in str(frame.path) for frame in excinfo.traceback)


def test_a_mistyped_populated_by_entry_names_the_registry_and_the_field():
    registry = Registry("MYTHING", populated_by=("eden.managers.terms.rewardz",))

    with pytest.raises(ModuleNotFoundError) as excinfo:
        registry.get("anything")

    notes = " ".join(excinfo.value.__notes__)
    assert "'MYTHING' registry" in notes and "eden.managers.terms.rewardz" in notes and "populated_by" in notes


def test_a_repr_reports_a_broken_population_instead_of_raising(make_package):
    package = make_package("reprbroken", "raise RuntimeError('term module is broken')\n")

    rendered = repr(package.REGISTRY)

    assert "REPRBROKEN" in rendered and "RuntimeError" in rendered and "term module is broken" in rendered
    with pytest.raises(RuntimeError, match="term module is broken"):
        package.REGISTRY.get("anything")


def test_a_concurrent_first_read_does_not_see_an_empty_registry(make_package):
    """The overlap is observed, not timed: the probe module blocks until the second reader is
    provably inside a call made from ``_populate`` — the import that waits on the per-module lock.
    """
    from eden.utils.registry import Registry as _Registry

    def inside_a_call_from_populate(thread: threading.Thread) -> bool:
        frame = sys._current_frames().get(thread.ident)
        if frame is None or frame.f_code is _Registry._populate.__code__:
            return False
        while frame is not None:
            if frame.f_code is _Registry._populate.__code__:
                return True
            frame = frame.f_back
        return False

    terms = (
        "import {pkg} as pkg\n"
        "pkg.importing.set()\n"
        "assert pkg.released.wait(60), 'the test never released the probe module'\n"
        "@pkg.REGISTRY.register()\nclass SlowTerm:\n    pass\n"
    )
    package = make_package(
        "concurrent",
        terms,
        init_extra="import threading\nimporting = threading.Event()\nreleased = threading.Event()\n",
    )
    results: dict[str, object] = {}

    def read(tag):
        try:
            results[tag] = package.REGISTRY.get("slow_term")
        except Exception as exc:
            results[tag] = exc

    try:
        first, second = (threading.Thread(target=read, args=(tag,)) for tag in ("first", "second"))
        first.start()
        assert package.importing.wait(60), "the populating thread never entered the probe module"
        second.start()
        # Bounded: with the bug the second thread never enters the import, so this never turns true.
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and not inside_a_call_from_populate(second):
            time.sleep(0.001)
        package.released.set()
        for thread in (first, second):
            thread.join(60)
            assert not thread.is_alive(), "a reader thread deadlocked"
    finally:
        # A probe still blocked on `released` would hold the import lock for its full 60 s.
        package.released.set()

    assert not isinstance(results["second"], Exception), results["second"]
    assert results["first"] is results["second"], results
