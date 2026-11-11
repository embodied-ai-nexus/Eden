"""``eden.utils.run_manifest``: the per-run ``run.json`` and the RL-library provenance it carries."""

from __future__ import annotations

import json
from importlib import metadata

import pytest

from eden.utils import run_manifest
from eden.utils.misc import get_eden_provenance, get_package_provenance


def _manifest(tmp_path, **overrides) -> dict:
    fields = dict(
        log_dir=str(tmp_path / "26Aug15T120000-ab12cd34"),
        run_id="ab12cd34",
        timestamp="26Aug15T120000",
        task="reacher",
        seed=7100,
        world_size=1,
        run_name="ablation-A",
        argv=["eden", "train", "--task", "reacher"],
    )
    return run_manifest.build_run_manifest(**{**fields, **overrides})


def test_the_manifest_records_the_run_its_identity_and_the_installed_code(tmp_path):
    manifest = _manifest(tmp_path, rl_backend="rsl_rl")

    assert {k: manifest[k] for k in ("run_id", "timestamp", "task", "seed", "world_size", "run_name", "argv")} == {
        "run_id": "ab12cd34",
        "timestamp": "26Aug15T120000",
        "task": "reacher",
        "seed": 7100,
        "world_size": 1,
        "run_name": "ablation-A",
        "argv": ["eden", "train", "--task", "reacher"],
    }
    assert manifest["schema_version"] == 3
    assert manifest["log_dir"] == str(tmp_path / "26Aug15T120000-ab12cd34")
    assert manifest["resumed_from"] is None
    assert manifest["eden"] == {**get_eden_provenance(), "dirty": manifest["eden"]["dirty"]}
    assert manifest["eden"]["dirty"] in (True, False, None)
    assert manifest["genesis"] == get_package_provenance("genesis-world")
    assert manifest["genesis"]["version"] == metadata.version("genesis-world")
    assert manifest["rl_library"] == {
        "backend": "rsl_rl",
        "distribution": "rsl-rl-lib",
        "version": metadata.version("rsl-rl-lib"),
    }


def test_a_resumed_run_records_its_parent_and_a_seedless_backend_records_none(tmp_path):
    manifest = _manifest(tmp_path, seed=None, resumed_from="26Aug15T110000-parent01")

    assert (manifest["seed"], manifest["resumed_from"]) == (None, "26Aug15T110000-parent01")


@pytest.mark.parametrize(
    ("backend", "distribution"), [("rsl_rl", "rsl-rl-lib"), ("rl_games", "rl-games")], ids=["rsl_rl", "rl_games"]
)
def test_the_rl_library_record_follows_the_backend(backend, distribution):
    pytest.importorskip(backend)

    assert run_manifest.get_rl_library(backend) == {
        "backend": backend,
        "distribution": distribution,
        "version": metadata.version(distribution),
    }


def test_the_rl_library_is_never_fabricated(monkeypatch):
    """An unknown backend records nothing; a known one whose distribution is absent still names the backend."""

    def not_installed(distribution):
        raise metadata.PackageNotFoundError(distribution)

    assert run_manifest.get_rl_library(None) is None
    assert run_manifest.get_rl_library("some_future_runner") is None
    monkeypatch.setattr(metadata, "version", not_installed)  # the installed-package metadata boundary
    assert run_manifest.get_rl_library("rsl_rl") == {"backend": "rsl_rl", "distribution": "rsl-rl-lib", "version": None}


def test_write_then_read_round_trips(tmp_path):
    manifest = _manifest(tmp_path, rl_backend="rsl_rl")

    path = run_manifest.write_run_manifest(str(tmp_path), manifest)

    assert path == str(tmp_path / "run.json")
    assert json.loads((tmp_path / "run.json").read_text()) == manifest
    assert run_manifest.read_run_manifest(str(tmp_path)) == manifest


@pytest.mark.parametrize("content", [None, "{not json", "[]"], ids=["missing", "corrupt", "not-a-mapping"])
def test_reading_is_best_effort(tmp_path, content):
    """A run without ``run.json`` or a truncated write must not raise."""
    if content is not None:
        (tmp_path / "run.json").write_text(content)

    assert run_manifest.read_run_manifest(str(tmp_path)) is None
