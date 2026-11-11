"""Tests for ``eden.utils.devices.metaquest.xr_bridge``: a missing televuer is named with the commands that install it."""

import importlib
import sys

import pytest

from eden.utils.distribution import pip_install_extra, uv_sync_extra

BRIDGE = "eden.utils.devices.metaquest.xr_bridge"


class _NotInstalled:
    """A meta-path finder for which ``name`` is not installed: the import fails as it does without the package (a
    ``None`` in ``sys.modules`` would report the submodule asked for, ``televuer.televuer``, instead).
    """

    def __init__(self, name: str):
        self.name = name

    def find_spec(self, fullname, path=None, target=None):
        if fullname == self.name:
            raise ModuleNotFoundError(f"No module named {fullname!r}", name=fullname)
        return None


def test_a_missing_televuer_names_both_install_commands(monkeypatch):
    """The televuer package is not on PyPI, so `pip install "eden-world[metaquest]"` alone does not bring it."""
    pytest.importorskip("zmq", reason="the metaquest extra")
    for module in [m for m in sys.modules if m == "televuer" or m.startswith("televuer.")] + [BRIDGE]:
        monkeypatch.delitem(sys.modules, module, raising=False)
    monkeypatch.setattr(sys, "meta_path", [_NotInstalled("televuer"), *sys.meta_path])
    with pytest.raises(ModuleNotFoundError) as info:
        importlib.import_module(BRIDGE)
    assert info.value.name == "televuer"
    assert pip_install_extra("metaquest") in str(info.value)
    assert uv_sync_extra("metaquest") in str(info.value)
