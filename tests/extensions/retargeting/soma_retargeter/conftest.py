"""Fixtures over the stand-in ``eden-soma-retargeter`` installation in :mod:`tests.support.soma_retargeter`."""

from __future__ import annotations

import pytest

from tests.support.soma_retargeter import install_soma_stand_in, write_dist_info


@pytest.fixture
def soma_install(tmp_path, monkeypatch):
    """``soma_install(backend="0.4.0", py_soma_x="0.2.1")`` puts a fake install first on ``sys.path``.

    A version of ``None`` leaves that distribution's metadata out; ``NOT_INSTALLED`` leaves its package out too.
    Returns a callable listing, in order, the entries the fake library has recorded so far.
    """
    return lambda **install: install_soma_stand_in(monkeypatch, tmp_path, **install)


@pytest.fixture(scope="module")
def soma_installed(tmp_path_factory):
    """The pinned fake install for a whole module; yields its record reader like ``soma_install``."""
    with pytest.MonkeyPatch.context() as monkeypatch:
        yield install_soma_stand_in(monkeypatch, tmp_path_factory.mktemp("soma"))


@pytest.fixture
def dist_info():
    """``dist_info(site, name, version, editable_checkout=None, files=None)`` writes distribution metadata, as the
    fake install does.
    """
    return write_dist_info
