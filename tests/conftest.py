"""Pytest configuration and fixtures for Eden framework tests."""

import copy
import importlib
import logging
import pathlib
import sys
import warnings

import pytest
import torch

import eden as en

# No sys.path manipulation here: with ``tests`` a regular package (see tests/__init__.py),
# pytest's default prepend import mode puts the repo root on sys.path itself. The manual
# insert this replaced could not have fixed the shadowing anyway — path order is irrelevant
# when a *regular* package of the same name exists anywhere on the path.


@pytest.fixture(scope="session", autouse=True)
def _init_eden_once():
    en.init(debug=True)
    yield
    # Genesis's atexit destroy() logs "💤 Exiting Genesis…" through a handler bound
    # to sys.stdout at init; under pytest that stream is closed by the time atexit
    # fires, raising "I/O operation on closed file" over the real pass/fail line.
    # Destroy here instead (stream still open): it logs cleanly and unregisters its
    # own atexit hook. getattr guards against an upstream rename.
    import genesis as gs

    if getattr(gs, "_initialized", False):
        gs.destroy()
    gs_logger = getattr(gs, "logger", None)
    if gs_logger is not None:
        gs_logger.removeHandler(gs_logger.handler)


@pytest.fixture
def user_assets_dir(tmp_path, monkeypatch) -> pathlib.Path:
    """The per-user asset directory (``get_assets_dir()``), moved into ``tmp_path``.

    ``EDEN_ASSETS_DIR`` is unset and ``XDG_CACHE_HOME`` points at a scratch directory, so a test that
    downloads through a default ``local_dir`` neither fills ``~/.cache`` nor depends on what it holds.
    """
    monkeypatch.delenv("EDEN_ASSETS_DIR", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    return tmp_path / "xdg" / "eden" / "assets"


@pytest.fixture
def task_registry_alias():
    """Register extra names on the global ``TASK_REGISTRY`` and remove them afterwards.

    Yields a ``register(cls, name)`` callable. Use it instead of hand-rolling teardown: a plain
    ``TASK_REGISTRY._registry.pop(name)`` is incomplete, because ``register`` writes **two** dicts
    (``TaskRegistry._do_register`` sets ``_registry`` *and* ``_modifiers``), so the popped name
    leaves a modifier entry behind for the rest of the session.

    Deliberately removes only the names it added — a wholesale snapshot-and-restore of
    ``_registry`` is **not** safe here. Entries that appeared because a task module was imported
    cannot be put back: the module stays in ``sys.modules``, so its ``@TASK_REGISTRY.register``
    decorator never runs a second time, and the next ``TASK_REGISTRY.get`` for that task raises
    "Auto-discovered task ... but importing its module did not register it"
    (``eden/tasks/registry.py``). Growing the registry is harmless; shrinking it is not.
    """
    from eden.tasks import TASK_REGISTRY

    added: list[str] = []

    def register(cls, name):
        TASK_REGISTRY.register(cls, name=name)
        added.append(name)
        return name

    yield register

    for name in added:
        TASK_REGISTRY._registry.pop(name, None)
        TASK_REGISTRY._modifiers.pop(name, None)


@pytest.fixture
def task_package(tmp_path, monkeypatch):
    """Write an importable package of task files: ``task_package("pkg", {"task.py": source})`` returns its directory.

    The packages live in a directory of their own on ``sys.path`` rather than in ``tmp_path``, where a
    test's other files (an ``examples/`` tree, say) would become importable too.
    """
    root = tmp_path / "task_packages"
    root.mkdir()
    monkeypatch.syspath_prepend(str(root))

    def write(name: str, files: dict[str, str]) -> pathlib.Path:
        package = root / name
        package.mkdir()
        (package / "__init__.py").write_text("")
        for file_name, source in files.items():
            (package / file_name).write_text(source)
        importlib.invalidate_caches()
        # Submodules too: another test may have imported one of a package by this name from its own tmp_path.
        for module in [m for m in sys.modules if m == name or m.startswith(f"{name}.")]:
            del sys.modules[module]
        return package

    return write


@pytest.fixture
def isolated_task_registry():
    """The global ``TASK_REGISTRY`` on copies of its state, rescanning on its next lookup; restored after.

    For a test that changes what the scan sees (search paths, installed entry points) or registers
    tasks by importing a module — which registers into the global registry, so a fresh
    :class:`~eden.tasks.registry.TaskRegistry` would not see it. Every attribute is copied, so a
    field added to the registry later cannot leak between tests.

    Restoring is not a plain snapshot, for the reason :func:`task_registry_alias` gives: a module
    imported during the test stays in ``sys.modules``, so its decorator never runs again. The
    registrations an ``eden.*`` module made during the test are therefore kept; the test's own
    throwaway packages are dropped with everything else.
    """
    from eden.tasks import TASK_REGISTRY

    saved = {attr: copy.copy(value) for attr, value in vars(TASK_REGISTRY).items()}
    TASK_REGISTRY._invalidate_discovery_cache()
    yield TASK_REGISTRY

    imported = {
        name: (cls, TASK_REGISTRY._modifiers[name])
        for name, cls in TASK_REGISTRY._registry.items()
        if name not in saved["_registry"] and cls.__module__.startswith("eden.")
    }
    for attr, value in saved.items():
        setattr(TASK_REGISTRY, attr, value)
    for name, (cls, entry) in imported.items():
        TASK_REGISTRY._write_registration(name, cls, entry)


@pytest.fixture
def eden_caplog(caplog):
    """``caplog``, wired up so it can actually see ``en.logger`` output.

    ``en.logger`` writes to the ``eden`` stdlib logger, which owns its own handler
    and sets ``propagate=False`` (``eden/_logging.py``) so Eden's formatting isn't
    duplicated by whatever the host app puts on the root logger. pytest's ``caplog``
    only installs a handler on the *root* logger, so a plain ``caplog`` sees nothing
    from Eden and every assertion on its records silently passes vacuously — or fails
    while the message is plainly there in captured stdout.

    Ask for this fixture instead of ``caplog`` whenever the log line under test comes
    from ``en.logger``. It also drops the ``eden`` logger to DEBUG for the test, so the
    logger's own threshold can't silently swallow the record you're asserting on —
    no ``at_level(..., logger="eden")`` dance needed at the call site.

    Survives an ``eden.init()`` inside the test: ``eden._logging.Logger`` retires only
    the handlers it installed itself, so this one stays attached (it does reset the
    logger's level, which only matters for records below the level ``init`` was given).

    Retirable once the pytest floor reaches 9.1: from that version ``caplog`` attaches to
    every non-propagating logger itself, so the hand-attachment here is redundant and plain
    ``caplog`` would do. The level-lowering would still be this fixture's own contribution.
    ``tests/test_logging.py`` asserts whichever behaviour the running pytest has.
    """
    eden_logger = logging.getLogger("eden")
    previous_level = eden_logger.level
    eden_logger.setLevel(logging.DEBUG)
    eden_logger.addHandler(caplog.handler)
    try:
        yield caplog
    finally:
        eden_logger.removeHandler(caplog.handler)
        eden_logger.setLevel(previous_level)


@pytest.fixture(autouse=True)
def reset_logging():
    """Reset logging configuration before each test."""
    # Store original handlers
    root_logger = logging.getLogger()
    original_handlers = root_logger.handlers[:]

    yield

    # Restore original handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    for handler in original_handlers:
        root_logger.addHandler(handler)


@pytest.fixture
def count_cuda_syncs():
    """Return ``count(fn)``: how many blocking GPU→CPU syncs calling ``fn()`` costs.

    For pinning a hot path's sync budget, which no functional test can see: a stray ``.item()``
    or a bool mask used as an index is correct, just slow. Counts the warnings
    ``torch.cuda.set_sync_debug_mode("warn")`` raises on every synchronizing call; PyTorch calls
    that detector a prototype, so a count can only err low. Skips without CUDA.
    """
    if not torch.cuda.is_available():
        pytest.skip("counting GPU->CPU syncs needs CUDA")

    def count(fn) -> int:
        torch.cuda.synchronize()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            torch.cuda.set_sync_debug_mode("warn")
            try:
                fn()
            finally:
                torch.cuda.set_sync_debug_mode("default")
        return sum("called a synchronizing CUDA operation" in str(w.message) for w in caught)

    return count


def pytest_configure(config):
    """Declare the markers tests apply to themselves, per test or per module (``pytestmark``).

    Nothing is marked automatically: a ``nodeid``-substring hook once did, and selected nearly the
    opposite of what it read as.
    """
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "requires_genesis: marks tests requiring Genesis simulator")
    config.addinivalue_line(
        "markers",
        "precision(bits): the Genesis precision ('32' or '64') a MochiSolver test runs at (tests/engine/solvers/mochi)",
    )
