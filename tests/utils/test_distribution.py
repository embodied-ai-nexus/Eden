"""Tests for ``eden.utils.distribution``: the install commands Eden prints, in or out of a checkout, are complete."""

import json
import shlex
from importlib import metadata

import pytest
from packaging.requirements import Requirement

from eden.utils.distribution import (
    DISTRIBUTION_NAME,
    REINSTALLED_PACKAGES,
    UNPUBLISHED_REQUIREMENTS,
    UV_ONLY_EXTRAS,
    direct_url_record,
    editable_checkout,
    install_hint,
    pip_install_extra,
    uv_sync_extra,
)


def _requirements(command: str) -> list[Requirement]:
    words = shlex.split(command)
    assert words[:2] == ["pip", "install"], command
    return [Requirement(word) for word in words[2:]]


def test_an_extra_on_pypi_installs_as_the_distribution_with_that_extra():
    (req,) = _requirements(pip_install_extra("vis"))
    assert (req.name, req.extras, req.url) == (DISTRIBUTION_NAME, {"vis"}, None)


def test_an_extra_with_unpublished_requirements_names_them_in_the_same_command():
    """`pip install "eden-world[flashsac]"` alone leaves rsl-rl-flashsac out: PyPI would refuse the git pin."""
    for extra, unpublished in UNPUBLISHED_REQUIREMENTS.items():
        eden, *rest = _requirements(pip_install_extra(extra))
        assert (eden.name, eden.extras) == (DISTRIBUTION_NAME, {extra})
        assert [str(req) for req in rest] == [str(Requirement(req)) for req in unpublished]


def test_a_checkout_syncs_an_extra_alone_or_with_the_git_group_that_completes_it():
    assert uv_sync_extra("vis") == "uv sync --inexact --extra vis"
    for extra in UNPUBLISHED_REQUIREMENTS:
        assert uv_sync_extra(extra) == f"uv sync --inexact --extra {extra} --group {extra}-git"


def test_a_checkout_reinstalls_a_wheel_that_shares_its_directory_with_another_extras():
    """`--inexact` keeps onnxruntime-gpu when the reader asks for the CPU `onnx` extra; both own `onnxruntime/`, so the
    requested wheel is reinstalled on top, which also repairs a directory a half-removed wheel emptied.
    """
    assert REINSTALLED_PACKAGES == {"onnx": "onnxruntime", "onnx-gpu": "onnxruntime-gpu"}
    for extra, package in REINSTALLED_PACKAGES.items():
        assert uv_sync_extra(extra) == f"uv sync --inexact --extra {extra} --reinstall-package {package}"


def test_pip_reinstalls_an_installed_wheel_alone_at_its_installed_version(monkeypatch):
    """With pip, an installed onnxruntime stays under onnxruntime-gpu's files, so the hint reinstalls it on top; at the
    installed version, as `--force-reinstall` would otherwise fetch the newest, and `--no-deps` keeps torch out.
    """
    monkeypatch.setitem(REINSTALLED_PACKAGES, "vis", "torch")  # installed in every Eden environment
    install, reinstall = pip_install_extra("vis").split(" && ")
    (req,) = _requirements(install)
    assert (req.name, req.extras) == (DISTRIBUTION_NAME, {"vis"})
    *flags, pinned = shlex.split(reinstall)
    assert flags == ["pip", "install", "--force-reinstall", "--no-deps"]
    assert Requirement(pinned) == Requirement(f"torch=={metadata.version('torch')}")


def test_pip_installs_a_wheel_it_lacks_with_the_extra_and_reinstalls_nothing(monkeypatch):
    """The extra's install puts a wheel it lacks on top by itself; a reinstall would fail where the extra's marker
    skips the wheel (`onnxruntime-gpu` on macOS).
    """
    monkeypatch.setitem(REINSTALLED_PACKAGES, "vis", "eden-no-such-wheel")
    assert pip_install_extra("vis") == f'pip install "{DISTRIBUTION_NAME}[vis]"'


def test_a_hint_gives_both_commands_so_it_reads_right_from_pypi_or_a_clone():
    for extra in ("vis", "onnx", *UNPUBLISHED_REQUIREMENTS):
        hint = install_hint(extra)
        assert f"`{pip_install_extra(extra)}`" in hint
        assert f"`{uv_sync_extra(extra)}` in a checkout" in hint


def test_a_hint_for_an_extra_pip_cannot_install_gives_the_checkouts_command_and_says_why():
    """`pip install "eden-world[soma-retargeter]"` fails to resolve: pip ignores uv's override-dependencies."""
    assert UV_ONLY_EXTRAS
    for extra, reason in UV_ONLY_EXTRAS.items():
        hint = install_hint(extra)
        assert "pip install" not in hint
        assert f"`{uv_sync_extra(extra)}` in a checkout" in hint
        assert reason in hint


def _dist(tmp_path, direct_url: str | None) -> metadata.Distribution:
    """A distribution whose ``direct_url.json`` holds ``direct_url`` verbatim (none when ``None``)."""
    info = tmp_path / "probe-1.0.dist-info"
    info.mkdir()
    (info / "METADATA").write_text("Metadata-Version: 2.1\nName: probe\nVersion: 1.0\n")
    if direct_url is not None:
        (info / "direct_url.json").write_text(direct_url)
    return metadata.PathDistribution(info)


def test_an_editable_install_names_its_checkout(tmp_path):
    checkout = tmp_path / "my checkout"  # a space, percent-encoded in the URL
    dist = _dist(tmp_path, json.dumps({"url": checkout.as_uri(), "dir_info": {"editable": True}}))
    assert editable_checkout(dist) == checkout


@pytest.mark.parametrize(
    "direct_url",
    [
        None,  # from an index
        json.dumps({"url": "file:///src/probe", "dir_info": {}}),  # a non-editable install of a local directory
        json.dumps({"url": "https://github.com/x/y", "vcs_info": {"vcs": "git", "commit_id": "a" * 40}}),
        "{not json",
        "[]",
    ],
    ids=["index", "local-non-editable", "git", "corrupt", "not-an-object"],
)
def test_anything_but_an_editable_install_names_no_checkout(tmp_path, direct_url):
    assert editable_checkout(_dist(tmp_path, direct_url)) is None


@pytest.mark.parametrize("direct_url", [None, "{not json", "[]"], ids=["none", "corrupt", "not-an-object"])
def test_a_missing_or_unreadable_record_reads_as_empty(tmp_path, direct_url):
    assert direct_url_record(_dist(tmp_path, direct_url)) == {}
