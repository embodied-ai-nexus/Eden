"""Fixtures shared by the task tests."""

from __future__ import annotations

from pathlib import Path

import pytest

TESTS_DATA_DIR = Path(__file__).resolve().parents[1] / "data"


@pytest.fixture(scope="session")
def tests_data_dir() -> Path:
    """``tests/data``, where the small golden ``.npz`` files live."""
    return TESTS_DATA_DIR
