"""Guards on the test suite's own package layout.

Without ``tests/__init__.py``, ``tests`` is only a namespace package, and any dependency shipping a
top-level ``tests/`` (``ultralytics``, via the pose-estimation extra) takes the name outright, so every
``from tests.support... import`` dies at collection. Below that, ``tests/`` mirrors ``eden/``: only
tests of a top-level ``eden`` module live directly in it.
"""

from __future__ import annotations

from pathlib import Path

import eden
import tests

TESTS = Path(tests.__file__).resolve().parent
#: Root files that mirror no module: this guard, and the tests of the repo-root ``pyproject.toml``.
ROOT_FILES_THAT_ARE_NOT_MODULE_MIRRORS = {"test_suite_layout.py", "test_pyproject.py"}


def test_tests_package_resolves_inside_the_repo():
    resolved = Path(tests.__file__).resolve()
    assert resolved == Path(__file__).resolve().parent / "__init__.py", (
        f"`import tests` resolved to {resolved}, not Eden's own test package. Is tests/__init__.py missing?"
    )


def test_tests_helper_modules_are_importable():
    """The submodule imports the collection errors used to fire on."""
    from tests.support.asserts import assert_allclose  # noqa: F401
    from tests.support.ddp import ddp_spawn  # noqa: F401
    from tests.support.envs import build_env  # noqa: F401


def test_every_root_test_file_mirrors_a_top_level_eden_module():
    modules = {path.stem.strip("_") for path in Path(eden.__file__).parent.glob("*.py")}
    allowed = {f"test_{name}.py" for name in modules} | ROOT_FILES_THAT_ARE_NOT_MODULE_MIRRORS

    strays = sorted(path.name for path in TESTS.glob("test_*.py") if path.name not in allowed)

    assert not strays, (
        f"tests/ holds {strays}, which mirror no top-level eden module. Move each to the path of the module "
        f"it tests (eden/utils/geom.py -> tests/utils/test_geom.py); see docs/guides/testing.md."
    )
