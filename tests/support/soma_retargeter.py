"""A stand-in ``eden-soma-retargeter`` installation: real files on ``sys.path``, found the way an install is.

The library needs warp, py-soma-x, downloadable assets and a GPU, so a stand-in package takes its place:
distribution metadata for the preflight, and a ``MotionRetargeter`` for the worker subprocess that records
what it was given to the JSON-lines file named by ``EDEN_TEST_SOMA_RECORD`` and echoes each clip back.

It ships both top-level packages, ``soma_retargeter`` and py-soma-x's ``soma``, and shadows a real install of
either further down ``sys.path`` (the ``soma-retargeter`` extra): the preflight reads the metadata of the copy it
would import, so a version the stand-in leaves out is missing however the environment was synced.
"""

from __future__ import annotations

import enum
import json
import sys
import textwrap
from collections.abc import Collection
from pathlib import Path

import pytest

_MOTION_RETARGETER = """
import json, os

import numpy as np


def _record(entry):
    with open(os.environ["EDEN_TEST_SOMA_RECORD"], "a") as f:
        f.write(json.dumps(entry) + "\\n")


class MotionRetargeter:
    def __init__(self, enable_procedural_transforms=True, **kwargs):
        self._procedural = enable_procedural_transforms
        _record({"init": {"enable_procedural_transforms": enable_procedural_transforms, **kwargs}})

    def retarget_batch(self, clips):
        # The layer is built lazily on the first clip, with the flag threaded through rather than defaulted.
        from soma.soma import SOMALayer

        _record({"layer_procedural": SOMALayer("assets", enable_procedural_transforms=self._procedural).enable_procedural_transforms})
        return [
            {key: np.asarray(clip[key]) if key != "joint_names" else list(clip[key]) for key in ("joint_names", "dof_pos", "root_pos", "root_quat")}
            for clip in clips
        ]
"""

_SOMA_LAYER = '''
class SOMALayer:
    """Mirrors py-soma-x: the flag is positional-or-keyword, not keyword-only."""

    def __init__(self, data_root=None, low_lod=False, device="cuda", enable_procedural_transforms=True):
        self.enable_procedural_transforms = enable_procedural_transforms
'''


class _Absent(enum.Enum):
    NOT_INSTALLED = "not installed"


NOT_INSTALLED = _Absent.NOT_INSTALLED
"""A version for :func:`install_soma_stand_in` that leaves the package out altogether, not only its metadata."""


def write_dist_info(
    site: Path, name: str, version: str, editable_checkout: Path | None = None, files: Collection[str] | None = None
) -> None:
    """Write ``name``'s ``.dist-info`` into ``site``. With ``editable_checkout``, as an editable install of it; with
    ``files``, a ``RECORD`` listing those paths (relative to ``site``, as an installer writes them).
    """
    info = site / f"{name.replace('-', '_')}-{version}.dist-info"
    info.mkdir(parents=True)
    (info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n")
    if editable_checkout is not None:
        direct_url = {"url": editable_checkout.as_uri(), "dir_info": {"editable": True}}
        (info / "direct_url.json").write_text(json.dumps(direct_url))
    if files is not None:
        (info / "RECORD").write_text("".join(f"{path},,\n" for path in files))


def install_soma_stand_in(
    monkeypatch: pytest.MonkeyPatch,
    root: Path,
    backend: str | _Absent | None = "0.4.0",
    py_soma_x: str | _Absent | None = "0.2.1",
):
    """Write the stand-in install under ``root``, put it first on ``sys.path``, and point the record at ``root``.

    Each version argument says how that distribution is installed:

    - a version string: the package, and metadata at that version;
    - ``None``: the package is importable but carries no metadata;
    - :data:`NOT_INSTALLED`: neither, and a real copy further down ``sys.path`` is hidden from this process too
      (a ``None`` in ``sys.modules``), so the preflight finds it not installed however the environment was synced.
      The hiding does not reach the worker subprocess, which gets ``sys.path`` but not ``sys.modules``: a test that
      runs the worker would import the real copy there.

    Returns a callable listing, in order, the entries the stand-in library has recorded so far.
    """
    record = root / "soma_record.jsonl"
    monkeypatch.setenv("EDEN_TEST_SOMA_RECORD", str(record))
    site = root / "site"
    site.mkdir(parents=True)
    if backend is NOT_INSTALLED:
        _hide(monkeypatch, "soma_retargeter")
    else:
        (site / "soma_retargeter" / "retargeting").mkdir(parents=True)
        (site / "soma_retargeter" / "__init__.py").write_text("")
        (site / "soma_retargeter" / "retargeting" / "__init__.py").write_text(textwrap.dedent(_MOTION_RETARGETER))
        if backend is not None:
            write_dist_info(site, "eden-soma-retargeter", backend)
    if py_soma_x is NOT_INSTALLED:
        _hide(monkeypatch, "soma")
    else:
        (site / "soma").mkdir()
        (site / "soma" / "__init__.py").write_text("")
        (site / "soma" / "soma.py").write_text(textwrap.dedent(_SOMA_LAYER))
        if py_soma_x is not None:
            write_dist_info(site, "py-soma-x", py_soma_x)
    monkeypatch.syspath_prepend(str(site))  # the retargeter hands its sys.path on to the worker
    return lambda: [json.loads(line) for line in record.read_text().splitlines()] if record.exists() else []


def _hide(monkeypatch: pytest.MonkeyPatch, package: str) -> None:
    """Make ``package`` unimportable in this process: ``importlib.util.find_spec`` returns ``None`` for it."""
    monkeypatch.setitem(sys.modules, package, None)
