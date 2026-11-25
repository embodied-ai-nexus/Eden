"""The suite's supermimic library and env (``tests/support/supermimic.py``) under the names these tests use."""

import pytest


@pytest.fixture(scope="session")
def lafan1_library(supermimic_library):
    return supermimic_library


@pytest.fixture(scope="session")
def env(supermimic_env):
    return supermimic_env
