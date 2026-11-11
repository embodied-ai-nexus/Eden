"""Tests for ``eden.extensions.retargeting.soma_retargeter._backend.check_backend_available``.

It refuses installs whose fit is too old or silently wrong, reading distribution metadata only.
"""

import contextlib
import importlib
import importlib.machinery
import sys
from typing import get_args, get_type_hints

import pytest

from eden.extensions.retargeting.soma_retargeter import SomaRetargeter
from eden.extensions.retargeting.soma_retargeter._backend import SMPL_FAMILY, check_backend_available
from tests.support.soma_retargeter import NOT_INSTALLED


@pytest.mark.parametrize(
    ("install", "body_model", "message"),
    [
        # A pin binds only a fresh resolution; an older env keeps a backend without the kwarg, whatever the model.
        ({"backend": "0.3.0"}, "smplx", "enable_procedural_transforms"),
        ({"backend": "0.3.0", "py_soma_x": None}, "soma", "eden-soma-retargeter"),
        ({"backend": None}, "soma", "distribution metadata"),
        # 0.1.0's fit writes near-zero rotations and blames the GPU.
        ({"py_soma_x": "0.1.0"}, "smplx", "near-zero rotations"),
        ({"py_soma_x": None}, "smplh", "carries no py-soma-x distribution metadata"),
        # Not installed at all: each names the command that installs it.
        ({"backend": NOT_INSTALLED}, "smplx", r"backend is not installed\. Install it with `uv sync --inexact"),
        (
            {"py_soma_x": NOT_INSTALLED},
            "smplh",
            r"py-soma-x, which is not installed\. Install it with `uv sync --inexact",
        ),
    ],
    ids=[
        "old-backend",
        "old-backend-native-soma",
        "no-backend-metadata",
        "broken-py-soma-x",
        "py-soma-x-without-metadata",
        "no-backend",
        "no-py-soma-x-for-a-fit",
    ],
)
def test_an_unusable_install_is_refused(soma_install, install, body_model, message):
    soma_install(**install)
    with pytest.raises(ImportError, match=message):
        check_backend_available(body_model)


def _package(root, module):
    """Write an empty top-level package ``module`` under ``root``."""
    (root / module).mkdir(parents=True)
    (root / module / "__init__.py").write_text("")
    return root / module


def _append_to_sys_path(monkeypatch, *paths):
    monkeypatch.setattr(sys, "path", [*sys.path, *map(str, paths)])
    importlib.invalidate_caches()


@pytest.mark.parametrize(
    ("install", "body_model", "distribution", "module"),
    [
        ({"backend": None}, "soma", "eden-soma-retargeter", "soma_retargeter"),
        ({"py_soma_x": None}, "smplx", "py-soma-x", "soma"),
    ],
    ids=["backend", "py-soma-x"],
)
def test_another_copys_metadata_does_not_vouch_for_the_imported_one(
    soma_install, dist_info, tmp_path, monkeypatch, install, body_model, distribution, module
):
    """The imported copy has no metadata; a usable copy further down sys.path is named, with how to use it."""
    soma_install(**install)
    shadowed = tmp_path / "shadowed-site"
    dist_info(shadowed, distribution, "0.4.0", files=[f"{module}/__init__.py"])
    package = _package(shadowed, module)
    _append_to_sys_path(monkeypatch, shadowed)
    with pytest.raises(ImportError, match="distribution metadata") as refused:
        check_backend_available(body_model)
    assert f"0.4.0 at {package.resolve()}" in str(refused.value)
    assert "shadowed by this one: take this copy off sys.path" in str(refused.value)


def test_leftover_metadata_is_named_as_such_not_as_a_copy_to_switch_to(soma_install, dist_info, tmp_path, monkeypatch):
    """Metadata whose package is gone (an interrupted uninstall) gets no advice to take the imported copy away."""
    soma_install(backend=None)
    leftover = tmp_path / "leftover-site"
    dist_info(leftover, "eden-soma-retargeter", "0.4.0", files=["soma_retargeter/__init__.py"])
    _append_to_sys_path(monkeypatch, leftover)
    with pytest.raises(ImportError, match="distribution metadata") as refused:
        check_backend_available("soma")
    assert f"Leftover eden-soma-retargeter metadata (0.4.0 in {leftover.resolve()})" in str(refused.value)
    assert str(leftover.resolve() / "soma_retargeter") not in str(refused.value)


def test_stale_metadata_ahead_of_the_imported_copy_does_not_refuse_it(soma_install, dist_info, tmp_path, monkeypatch):
    """The first metadata of the name on sys.path is a leftover 0.3.0; the copy imported is a usable 0.4.0."""
    soma_install()
    stale = tmp_path / "stale-site"
    dist_info(stale, "eden-soma-retargeter", "0.3.0", files=["soma_retargeter/__init__.py"])
    monkeypatch.syspath_prepend(str(stale))
    check_backend_available("smplx")


@pytest.mark.parametrize(
    ("editable", "stale", "outcome"),
    [
        ("0.4.0", "0.3.0", contextlib.nullcontext()),
        ("0.3.0", "0.4.0", pytest.raises(ImportError, match="enable_procedural_transforms")),
    ],
    ids=["usable", "too-old"],
)
def test_an_editable_backend_is_vetted_by_its_own_metadata(dist_info, tmp_path, monkeypatch, editable, stale, outcome):
    """An editable install keeps its metadata away from its package, behind leftover metadata of another version:
    ``direct_url.json`` names the checkout, so the editable install's own version decides.
    """
    checkout, stale_site, site = tmp_path / "checkout", tmp_path / "stale-site", tmp_path / "site"
    _package(checkout, "soma_retargeter")
    dist_info(stale_site, "eden-soma-retargeter", stale, files=["soma_retargeter/__init__.py"])
    dist_info(site, "eden-soma-retargeter", editable, editable_checkout=checkout)
    sys_path_head = [checkout, stale_site, site]  # the stale metadata is found before the editable install's
    for path in reversed(sys_path_head):
        monkeypatch.syspath_prepend(str(path))
    with outcome:
        check_backend_available("soma")


def test_a_wheel_split_between_lib_and_lib64_is_found_through_its_record(dist_info, tmp_path, monkeypatch):
    """The installer put the package in ``lib64`` and the metadata in ``lib``; ``RECORD`` links the two."""
    lib, lib64 = tmp_path / "lib", tmp_path / "lib64"
    dist_info(lib, "eden-soma-retargeter", "0.4.0", files=["../lib64/soma_retargeter/__init__.py"])
    _package(lib64, "soma_retargeter")
    for path in (lib, lib64):
        monkeypatch.syspath_prepend(str(path))
    check_backend_available("soma")


def test_a_loader_that_reports_no_location_is_refused_by_name(monkeypatch):
    class NoLocationFinder:
        @staticmethod
        def find_spec(name, path=None, target=None):
            return importlib.machinery.ModuleSpec(name, loader=None) if name == "soma_retargeter" else None

    monkeypatch.setattr(sys, "meta_path", [NoLocationFinder, *sys.meta_path])
    with pytest.raises(ImportError, match="soma_retargeter is importable, but its loader .* reports no location"):
        check_backend_available("soma")


@pytest.mark.parametrize(
    ("install", "body_model"), [({}, "smplx"), ({"py_soma_x": None}, "soma")], ids=["pinned", "native-soma"]
)
def test_a_usable_install_passes(soma_install, install, body_model):
    """Native SOMA input never goes through the py-soma-x fit, so that one needs only the backend."""
    soma_install(**install)
    check_backend_available(body_model)


def test_the_smpl_family_is_every_body_model_but_native_soma():
    """It decides who needs py-soma-x, so a body model added to the option and not here would skip the check."""
    assert set(get_args(get_type_hints(SomaRetargeter)["body_model"])) == set(SMPL_FAMILY) | {"soma"}
