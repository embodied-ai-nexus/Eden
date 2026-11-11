"""Module-scoped clip library and mimic env shared by the ``eden.managers.motion`` tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.support.motion import build_mimic_env, write_library


@pytest.fixture(scope="module")
def library(tmp_path_factory) -> Path:
    """Three clips of 20, 30 and 40 frames at 50 Hz."""
    return write_library(tmp_path_factory.mktemp("library"), [20, 30, 40])


@pytest.fixture(scope="module")
def mimic_env(library):
    """A built, reset mimic env (4 envs) over :func:`library`."""
    return build_mimic_env(library)
