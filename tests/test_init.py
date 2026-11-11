"""Tests for ``eden/__init__.py``: what ``import eden`` does (not) import, ``eden.init()``, and the run directory.

Import side effects are properties of a process, and pytest has imported half the tree before any test
runs, so those checks run in a fresh interpreter. They are structural (``X in sys.modules``), not timed.
"""

import ast
import json
import logging
import os
import pathlib
import subprocess
import sys
import textwrap

import pytest

import eden as en
from eden._logging import _EDEN_OWNED_HANDLER, LOG_THEME_ENV_VAR
from eden.utils.format import ANSI_GREEN, colorize


def _in_fresh_interpreter(body: str, *, env: dict | None = None) -> str:
    """Run ``body`` in a new process and return its stdout, failing the test on a non-zero exit.

    Parameters
    ----------
    body : str
        Python source, dedented before it is run.
    env : dict | None
        Environment for the child; the parent's when ``None``.

    Returns
    -------
    str
        The script's stdout.
    """
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(body)], capture_output=True, text=True, check=False, env=env
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def _verdict(stdout: str) -> dict:
    """The JSON object a probe printed on its last ``VERDICT=`` line."""
    lines = [line for line in stdout.splitlines() if line.startswith("VERDICT=")]
    assert lines, f"the probe printed no verdict:\n{stdout}"
    return json.loads(lines[-1].removeprefix("VERDICT="))


#: One per lazily re-exported name: the attribute ``eden`` promises, and the module behind it.
LAZY_NAMES = sorted(en._LAZY_SUBMODULES.items())


@pytest.fixture(scope="module")
def bare_import():
    """What a fresh ``import eden`` pulled in, then what ``from eden import ...`` resolves to."""
    return _verdict(
        _in_fresh_interpreter(
            """
            import json, sys
            import eden
            eager = sorted(name for name in eden._LAZY_SUBMODULES.values() if name in sys.modules)
            heavy = sorted({name.split(".")[0] for name in sys.modules} & {"genesis", "torch"})
            from eden import observations, rewards, robots, scenes
            resolved = [m.__name__ for m in (rewards, observations, robots, scenes)]
            print("VERDICT=" + json.dumps({"eager": eager, "heavy": heavy, "resolved": resolved}))
            """
        )
    )


def test_import_eden_does_not_import_the_term_or_options_modules(bare_import):
    """Driven off ``_LAZY_SUBMODULES``, so a new lazy export extends the guard instead of escaping it."""
    assert bare_import["eager"] == [], f"`import eden` eagerly imported: {bare_import['eager']}"


def test_import_eden_imports_neither_genesis_nor_torch(bare_import):
    """Any ``genesis.*`` submodule runs ``genesis/__init__.py``, so one stray name costs the whole import."""
    assert bare_import["heavy"] == [], f"`import eden` pulled in: {bare_import['heavy']}"


def test_from_eden_import_reaches_the_lazy_names(bare_import):
    assert bare_import["resolved"] == [
        "eden.managers.terms.rewards",
        "eden.managers.terms.observations",
        "eden.options.robots",
        "eden.options.scenes",
    ]


@pytest.mark.parametrize(("attribute", "module"), LAZY_NAMES, ids=[name for name, _ in LAZY_NAMES])
def test_each_lazy_name_resolves_to_its_module_and_is_listed(attribute, module):
    resolved = getattr(en, attribute)
    assert resolved.__name__ == module
    assert getattr(en, attribute) is resolved  # bound into the module globals on first access
    assert attribute in dir(en)  # tab-completion and help(eden) keep every namespace


def test_an_unknown_attribute_is_still_an_attribute_error():
    """A fallback that imported ``eden.<anything>`` would turn a typo into a surprising import."""
    with pytest.raises(AttributeError, match="no attribute 'rewrads'"):
        en.rewrads  # noqa: B018


def test_a_task_config_does_not_import_cv2():
    """``numpydantic`` imports ``cv2``; the camera matrices every task config imports no longer use it.

    ``genesis`` alone must not import ``cv2`` either, or this guard can no longer see Eden.
    """
    out = _in_fresh_interpreter(
        """
        import sys
        import genesis
        print("cv2" in sys.modules)
        import eden.tasks.benchmark.reacher.config
        print("eden.options.camera" in sys.modules, "cv2" in sys.modules, "numpydantic" in sys.modules)
        """
    )
    after_genesis, after_config = out.splitlines()
    assert after_genesis == "False", "`import genesis` imports cv2 by itself now; this guard can no longer see Eden"
    camera_imported, cv2_imported, numpydantic_imported = after_config.split()
    assert camera_imported == "True", "the reacher config no longer imports eden.options.camera; pick one that does"
    assert (cv2_imported, numpydantic_imported) == ("False", "False"), (
        f"a task config imported cv2={cv2_imported}, numpydantic={numpydantic_imported}; "
        "run `python -X importtime -c 'import eden.tasks.benchmark.reacher.config' 2>&1 | grep -B5 cv2` to find who"
    )


def test_a_config_leaves_genesis_theme_to_init():
    """Importing and printing a config sets no Genesis theme, so its repr is plain until ``eden.init()``.

    Genesis renders an unset theme plain (https://github.com/Genesis-Embodied-AI/Genesis/pull/3452), so Eden needs no
    import-time seed against literal "None" colors.
    """
    out = _in_fresh_interpreter(
        """
        import json, os
        os.environ["EDEN_LOG_THEME"] = "human"
        import eden
        from eden.tasks.registry import TASK_REGISTRY
        from genesis.styles import style

        rendered = repr(TASK_REGISTRY.get("reacher")())
        print("VERDICT=" + json.dumps({"unset": style.theme is None, "escapes": "\\x1b" in rendered}))
        """
    )
    assert _verdict(out) == {"unset": True, "escapes": False}


_TERM_REGISTRIES = {
    "REWARD_TERM_REGISTRY": "action_l2",
    "OBSERVATION_TERM_REGISTRY": "base_lin_vel",
    "ACTION_TERM_REGISTRY": "implicit_pd_controller",
    "EVENT_TERM_REGISTRY": "push_by_velocity",
    "TERMINATION_TERM_REGISTRY": "time_out",
    "COMMAND_TERM_REGISTRY": "uniform_velocity_command",
    "CURRICULUM_TERM_REGISTRY": "velocity_command_curriculum",
    "METRIC_TERM_REGISTRY": "object_lifted",
}
_MODIFIER_REGISTRIES = ("NOISE_MODEL_REGISTRY", "OBSERVATION_MODIFIER_REGISTRY", "ACTION_MODIFIER_REGISTRY")


@pytest.mark.parametrize(
    ("imports", "check"),
    [
        *((f"from eden.managers import {r}", f"{r}.get({t!r}) is not None") for r, t in _TERM_REGISTRIES.items()),
        (
            f"from eden.managers.modifiers.base import {', '.join(_MODIFIER_REGISTRIES)}",
            " and ".join(f"len(list({r}.keys())) > 0" for r in _MODIFIER_REGISTRIES),
        ),
        ("from eden.options.catalog import ROBOT_REGISTRY", "ROBOT_REGISTRY.get('franka_emika_panda') is not None"),
    ],
    ids=[*_TERM_REGISTRIES, "modifier-registries", "ROBOT_REGISTRY"],
)
def test_a_registry_fills_itself_after_a_bare_import_eden(imports, check):
    """A lookup imports the registry's ``populated_by`` itself, the situation a config loaded from JSON is in.

    One interpreter per registry: importing any one term module drags in every kind, so a shared
    interpreter would let a registry with a wrong ``populated_by`` resolve anyway.
    """
    out = _in_fresh_interpreter(
        f"""
        import eden
        {imports}
        print({check})
        """
    )
    assert out.strip() == "True"


def _utils_modules_importing_options() -> list[str]:
    """Every ``eden.utils.*`` module with a module-scope ``eden.options`` import, found by parsing.

    Returns
    -------
    list[str]
        Dotted module names, sorted.
    """
    root = pathlib.Path(en.__file__).parent
    found = []
    for path in sorted((root / "utils").rglob("*.py")):
        for node in ast.parse(path.read_text(), str(path)).body:  # module scope only
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
            else:
                continue
            if any(name == "eden.options" or name.startswith("eden.options.") for name in names):
                parts = list(path.relative_to(root.parent).with_suffix("").parts)
                found.append(".".join(parts[:-1] if parts[-1] == "__init__" else parts))
                break
    return sorted(found)


@pytest.mark.parametrize("module", _utils_modules_importing_options())
def test_a_utils_module_that_reaches_into_options_imports_first(module):
    """``eden.options`` imports back into ``eden.utils``, so such a module is half a cycle.

    The cycle closes only when that module is the first of the two imported, which is why each runs
    as the first statement of its own interpreter (regression: ``import eden.utils.file_handler`` raised).
    """
    _in_fresh_interpreter(f"import {module}")


@pytest.fixture(scope="module")
def agent_themed_run(tmp_path_factory):
    """A fresh process that runs ``eden.init(logger_theme="agent")`` for real and logs one line."""
    script = f"""
        import genesis as gs
        import eden
        eden.init(logger_theme="agent", log_root_path={str(tmp_path_factory.mktemp("logs"))!r})
        eden.logger.info("a ~~<path>~~ here")
        print("DEFAULT_BACKEND_IS_CPU", gs.backend == gs.cpu)
        """
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    return result.stdout + result.stderr


def test_an_agent_themed_run_emits_no_ansi_and_no_greeting_box(agent_themed_run):
    """The whole run's output: ``eden.init`` puts Genesis on its no-ANSI palette too.

    Genesis' own greeting box comes from inside ``gs.init``, so the box check is scoped to Eden's lines.
    """
    assert "\x1b[" not in agent_themed_run, "an ANSI escape survived the agent theme"
    eden_lines = [line for line in agent_themed_run.splitlines() if line.startswith("[Eden")]
    assert eden_lines, f"nothing was logged by Eden at all:\n{agent_themed_run}"
    assert not [line for line in eden_lines if "╭" in line or "┈┉" in line], "Eden's greeting box survived"
    assert "a path here" in agent_themed_run
    assert "Eden ver:" in agent_themed_run  # provenance is not decoration


def test_init_defaults_to_the_genesis_cpu_backend(agent_themed_run):
    """``backend=gs.cpu`` in the signature would evaluate ``gs.cpu`` at import, hence a sentinel."""
    assert "DEFAULT_BACKEND_IS_CPU True" in agent_themed_run


def test_init_imports_genesis_and_torch():
    """Deferring them must not mean nobody imports them: ``import eden`` staying clean is trivial otherwise."""
    out = _in_fresh_interpreter(
        """
        import sys
        import eden
        eden.init(initialize_genesis=False)
        print("GENESIS_AND_TORCH", "genesis" in sys.modules, "torch" in sys.modules)
        """
    )
    assert "GENESIS_AND_TORCH True True" in out


@pytest.mark.parametrize("genesis_imported_first", [False, True], ids=["fresh", "genesis-first"])
def test_a_bad_theme_variable_is_refused_at_import(genesis_imported_first):
    """The module-scope ``Logger`` resolves the theme unconditionally, whichever order genesis came in."""
    script = "import genesis; " if genesis_imported_first else ""
    result = subprocess.run(
        [sys.executable, "-c", f"{script}import eden"],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, LOG_THEME_ENV_VAR: "humam"},
    )

    assert result.returncode != 0
    # Quadrants wraps sys.stdout on import, so the traceback can land on either stream.
    assert "Invalid EDEN_LOG_THEME: 'humam'" in result.stdout + result.stderr, result.stdout + result.stderr


class TestReinit:
    """The session fixture has already initialized Eden and Genesis, so these re-inits hit the guard."""

    def test_it_keeps_a_caller_installed_handler_and_retires_its_own(self, tmp_path, eden_caplog):
        """It used to remove and close every handler on the ``eden`` logger, pytest's capture handler included."""
        en.init(debug=True, log_root_path=str(tmp_path))

        eden_logger = logging.getLogger("eden")
        assert eden_caplog.handler in eden_logger.handlers
        assert len([h for h in eden_logger.handlers if getattr(h, _EDEN_OWNED_HANDLER, False)]) == 1
        en.logger.warning("sentinel-after-reinit")
        assert "sentinel-after-reinit" in eden_caplog.text

    def test_it_warns_and_names_the_fix(self, tmp_path, eden_caplog):
        """A double init costs training attempts to an opaque CUDA "unknown error"."""
        en.init(debug=True, log_root_path=str(tmp_path))
        assert "called again" in eden_caplog.text
        assert "once per process" in eden_caplog.text


class TestLoggerThemeArgument:
    def test_it_is_keyword_only(self):
        """Inserted mid-list, it re-bound every caller passing ``log_root_path`` and beyond positionally."""
        with pytest.raises(TypeError):
            en.init(None, False, False, None, False, None, "", True, False, False, "32", 1e-8, "agent")

    @pytest.mark.parametrize("theme", ["human", "auto"])
    def test_init_exports_the_request_for_every_other_consumer(self, monkeypatch, theme):
        """``colorize``, ``--help`` and subprocesses read the variable; ``auto`` stays a per-process decision."""
        monkeypatch.setenv(LOG_THEME_ENV_VAR, "agent")  # registers a restore: init() writes os.environ itself
        monkeypatch.delenv(LOG_THEME_ENV_VAR)
        monkeypatch.setenv("FORCE_COLOR", "1")

        en.init(initialize_genesis=False, logger_theme=theme)

        assert os.environ[LOG_THEME_ENV_VAR] == theme
        assert colorize("x", ANSI_GREEN) != "x"

    @pytest.mark.parametrize(("theme", "genesis_theme"), [("agent", "raw"), ("human", "dark")])
    def test_init_points_genesis_at_the_matching_theme(self, monkeypatch, theme, genesis_theme):
        """``agent`` keeps the escapes out of Genesis' output too.

        https://github.com/Genesis-Embodied-AI/Genesis/pull/3452 renamed ``dumb`` to ``raw``.
        """
        from genesis.styles import THEME, style

        monkeypatch.setattr(style, "theme", style.theme)  # registers a restore: init() writes it itself
        monkeypatch.setenv(LOG_THEME_ENV_VAR, theme)

        en.init(initialize_genesis=False, logger_theme=theme)

        assert style.theme is THEME[genesis_theme]


# ---------------------------------------------------------------------------
# What eden.init forwards to Genesis and torch
# ---------------------------------------------------------------------------


_REPORT = """
    import json, genesis as gs, torch
    from torch.overrides import _get_current_function_mode_stack
    from torch.utils._device import DeviceContext

    print("VERDICT=" + json.dumps({
        "eps": gs.EPS,
        "float64": gs.tc_float == torch.float64,
        "deterministic": gs.use_deterministic_algorithms,
        "default_device_modes": sum(isinstance(m, DeviceContext) for m in _get_current_function_mode_stack()),
        "default_dtype_is_genesis_float": torch.get_default_dtype() == gs.tc_float,
    }))
"""


@pytest.mark.parametrize(
    ("init", "expected"),
    [
        ("en.init()", {"float64": False, "deterministic": False}),
        # Genesis clamps EPS to the float's own epsilon, so the 1e-8 default is visible at 64-bit only.
        ('en.init(precision="64")', {"eps": 1e-8, "float64": True}),
        ('en.init(precision="64", eps=1e-12, deterministic=True)', {"eps": 1e-12, "deterministic": True}),
    ],
    ids=["defaults", "default-eps", "forwarded"],
)
def test_init_forwards_precision_eps_and_determinism_to_genesis_and_torch(init, expected):
    """``eps`` defaults to 1e-8 for rigid-solver stability; the SAP coupler needs 64-bit and 1e-12."""
    report = _verdict(_in_fresh_interpreter(f"import eden as en\n{init}\n" + textwrap.dedent(_REPORT)))

    assert {k: report[k] for k in expected} == expected


@pytest.mark.parametrize("genesis_first", [False, True], ids=["eden-inits-genesis", "genesis-already-initialized"])
def test_init_leaves_no_torch_default_device_behind(genesis_first):
    """While one is set every torch call pays a ``TorchFunctionMode`` dispatch; ``gs.init`` set one before
    https://github.com/Genesis-Embodied-AI/Genesis/pull/3431.
    """
    prologue = "import genesis as gs\ngs.init(backend=gs.cpu, logging_level='warning')\n" if genesis_first else ""

    report = _verdict(_in_fresh_interpreter(prologue + "import eden as en\nen.init()\n" + textwrap.dedent(_REPORT)))

    assert report["default_device_modes"] == 0


@pytest.mark.parametrize("genesis_first", [False, True], ids=["eden-inits-genesis", "genesis-already-initialized"])
def test_init_sets_torch_default_dtype_to_the_genesis_float(genesis_first):
    """Genesis stopped setting it in https://github.com/Genesis-Embodied-AI/Genesis/pull/3431; at 64-bit a float32
    buffer can't take an indexed write of a Genesis tensor.
    """
    prologue = (
        "import genesis as gs\ngs.init(backend=gs.cpu, precision='64', logging_level='warning')\n"
        if genesis_first
        else ""
    )
    init = "en.init()" if genesis_first else 'en.init(precision="64")'

    report = _verdict(_in_fresh_interpreter(f"{prologue}import eden as en\n{init}\n" + textwrap.dedent(_REPORT)))

    assert report["float64"] and report["default_dtype_is_genesis_float"]


# ---------------------------------------------------------------------------
# The run directory: resolved at init, created on first use, unique per launch
# ---------------------------------------------------------------------------


@pytest.fixture
def restore_run_globals(monkeypatch):
    """Put back the session's run globals that ``init()`` / ``_new_run_dirname()`` rebind."""
    for name in ("_log_dir", "_run_id", "_run_timestamp", "logger"):
        monkeypatch.setattr(en, name, getattr(en, name))


def test_init_resolves_the_log_dir_and_only_first_use_creates_it(tmp_path, restore_run_globals):
    """An eval script calls ``init()`` and then globs ``logs/*`` for the newest run; an empty dir made by
    ``init()`` would be what it found.
    """
    root = tmp_path / "logs"
    previous = root / "26Aug09T110000-previous"
    previous.mkdir(parents=True)

    en.init(debug=True, initialize_genesis=False, log_root_path=str(root))

    assert list(root.iterdir()) == [previous]
    log_dir = en.get_log_dir()
    assert os.path.isdir(log_dir) and os.path.dirname(log_dir) == str(root)
    assert en.log_dir == en.get_log_dir() == log_dir  # the attribute alias; a second call is fine
    assert os.path.basename(log_dir) == f"{en.get_run_timestamp()}-{en.get_run_id()}"


def test_the_log_dir_before_init_is_a_pointed_error():
    report = _verdict(
        _in_fresh_interpreter(
            """
        import json, eden
        messages = []
        for read in (eden.get_log_dir, lambda: eden.log_dir):
            try:
                read()
            except RuntimeError as exc:
                messages.append(str(exc))
        print("VERDICT=" + json.dumps({"messages": messages}))
        """
        )
    )

    assert len(report["messages"]) == 2 and all("call `eden.init()`" in m for m in report["messages"])


@pytest.fixture
def frozen_now(monkeypatch, restore_run_globals):
    """Every run in the test resolves the same wall-clock second (the clock boundary)."""
    import eden.utils.misc

    monkeypatch.setattr(eden.utils.misc, "get_now", lambda: "26Aug15T120000")
    return "26Aug15T120000"


def test_same_second_launches_get_distinct_names_from_a_random_base36_id(frozen_now):
    names = {en._new_run_dirname() for _ in range(1000)}

    assert len(names) == 1000
    for name in names:
        timestamp, _, run_id = name.partition("-")
        assert timestamp == frozen_now
        assert len(run_id) == 8 and set(run_id) <= set(en._RUN_ID_ALPHABET)


def test_a_run_name_is_sanitized_into_the_middle_and_the_id_stays_last(frozen_now):
    """The accessors, not basename parsing, are the source of truth for the run's identity."""
    name = en._new_run_dirname("lr=3e-4/seed:0 arm")

    assert name == f"{frozen_now}-lr=3e-4-seed-0-arm-{en.get_run_id()}"  # separators and spaces become "-"
    assert en.get_run_timestamp() == frozen_now
    assert en._new_run_dirname("") == f"{frozen_now}-{en.get_run_id()}"
