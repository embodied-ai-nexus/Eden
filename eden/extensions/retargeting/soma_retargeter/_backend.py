"""Preflight check for the installed SOMA backend.

:func:`check_backend_available` answers one question — is a *usable* backend
installed? It reads distribution metadata only, never importing the backend
(which would pull warp / newton / py-soma-x), so it is cheap enough to run before
a Genesis scene is built: a missing or known-broken install then costs a message
instead of a 30 s scene build followed by a worker crash.

The metadata it reads is the one that belongs to the copy that would be imported,
not the first of that name on ``sys.path`` (what ``importlib.metadata.version``
returns): with a checkout on ``PYTHONPATH`` ahead of an install, or a stale
install ahead of a newer one, those are two different copies.

The backend floor exists because the public SOMA-X assets cannot build
py-soma-x's default rig: Eden fits against the public 78-joint one instead, which
needs a backend new enough to take ``enable_procedural_transforms``. The py-soma-x
floor skips a release whose fit is numerically wrong (see ``_MIN_PY_SOMA_X``).
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
from dataclasses import dataclass, field
from pathlib import Path

from packaging.version import parse

from eden.utils.distribution import editable_checkout, install_hint

#: Body models that reach the SOMA rig through the py-soma-x fit. Native
#: ``"soma"`` input (arrays / BVH) skips the fit, so it does not need py-soma-x
#: installed at all. Kept in sync with the ``SomaRetargeter.body_model`` literal
#: by ``tests/extensions/retargeting/soma_retargeter/test_backend.py``.
SMPL_FAMILY = ("smpl", "smplh", "smplx")

# First release whose MotionRetargeter accepts `enable_procedural_transforms`,
# which SomaRetargeter always passes; older ones raise TypeError on it.
_MIN_BACKEND = "0.4.0"

# py-soma-x 0.1.0's fit writes near-zero rotation matrices, which the backend
# then reports as "non-orthonormal rotations ... persistent GPU issue" — a
# silently-wrong numeric result behind a misleading message. 0.2 is the first
# release that fits correctly.
_MIN_PY_SOMA_X = "0.2"

_INSTALL_HINT = install_hint("soma-retargeter")


@dataclass
class _ImportedCopy:
    """The copy of a top-level module that ``import`` would load, and what the ``sys.path`` metadata says about it."""

    location: Path
    """The package directory, or the module file."""
    version: str | None = None
    """The version in the metadata that describes this copy; ``None`` when none does."""
    others: list[str] = field(default_factory=list)
    """Other copies the metadata of that name describes, as ``"<version> at <path>"``."""
    leftovers: list[str] = field(default_factory=list)
    """Metadata of that name whose copy is gone, as ``"<version> in <dir holding the metadata>"``."""


def _imported_copy(module: str, distribution: str) -> _ImportedCopy | None:
    """Find the copy of ``module`` that ``import`` would load, and the ``distribution`` metadata that describes it;
    ``None`` when ``module`` is not importable. Nothing is imported.

    Matched by ownership, not by name: ``importlib.metadata.version`` reads the first metadata of that name on
    ``sys.path``, which describes another copy when one shadows another.
    """
    spec = importlib.util.find_spec(module)
    if spec is None:
        return None
    # A package's location is its directory (the first, for a namespace package); a module's is its file.
    locations = list(spec.submodule_search_locations or ())
    origin = locations[0] if locations else spec.origin
    if origin is None:
        raise ImportError(
            f"{module} is importable, but its loader ({spec.loader!r}) reports no location, so the {distribution} "
            "metadata that describes the copy it would load cannot be found and its version cannot be checked."
        )
    imported = _ImportedCopy(Path(origin).resolve())
    for dist in importlib.metadata.distributions(name=distribution):
        described = _described_copy(dist, module)
        if described is None:
            imported.leftovers.append(f"{dist.version} in {Path(dist.locate_file('')).resolve()}")
            continue
        if imported.location.is_relative_to(described):  # the copy itself, or inside an editable checkout
            imported.version = dist.version
            return imported
        imported.others.append(f"{dist.version} at {described}")
    return imported


def _described_copy(dist: importlib.metadata.Distribution, module: str) -> Path | None:
    """Where ``dist``'s metadata puts its copy of top-level ``module``: the checkout of an editable install, else the
    installed package directory or module file. ``None`` when that copy no longer exists.

    An installed copy is read from the file list (``RECORD``) first, which can place it outside the metadata's own
    directory (a wheel split between ``lib`` and ``lib64``), then looked for beside the metadata: an ``.egg-info``
    written in a checkout has no ``RECORD``, and its ``SOURCES.txt`` paths are relative to the project, not to it.
    """
    checkout = editable_checkout(dist)
    if checkout is not None:
        checkout = checkout.resolve()
        return checkout if checkout.exists() else None
    # A package lists its ``__init__.py``; the shortest such path is the top-level one, not a same-named subpackage.
    files = dist.files or ()  # parses RECORD on every access
    listed = [f.parent for f in files if f.name == "__init__.py" and f.parent.name == module]
    listed += [f for f in files if f.name == f"{module}.py"]
    candidates = [Path(dist.locate_file(min(listed, key=lambda f: len(f.parts))))] if listed else []
    beside = Path(dist.locate_file(""))
    candidates += [beside / module, beside / f"{module}.py"]
    return next((path.resolve() for path in candidates if path.exists()), None)


def _other_metadata(imported: _ImportedCopy, distribution: str) -> str:
    """The sentences naming the ``distribution`` metadata that does not describe ``imported``; ``""`` when there is
    none.
    """
    sentences = []
    if imported.others:
        sentences.append(
            f" Copies with {distribution} metadata on sys.path ({'; '.join(imported.others)}) are shadowed by this "
            "one: take this copy off sys.path (PYTHONPATH, for one) to use one of them, or install this one."
        )
    if imported.leftovers:
        sentences.append(
            f" Leftover {distribution} metadata ({'; '.join(imported.leftovers)}) describes no copy that still exists "
            "(left by an interrupted uninstall, or an editable install whose checkout moved), so it vouches for "
            "nothing: delete that .dist-info (or .egg-info) directory."
        )
    return "".join(sentences)


def check_backend_available(body_model: str) -> None:
    """Raise ``ImportError`` unless a usable backend is installed for ``body_model``.

    Distribution-metadata only — no backend import, no asset download — so this
    is cheap enough to run before building a scene. ``eden-soma-retargeter`` is
    checked for every body model; ``py-soma-x`` only for the :data:`SMPL_FAMILY`
    ones, which are the ones that fit through it.

    Version floors are checked here as well as pinned in ``pyproject.toml``,
    because a pin only binds a fresh resolution — an environment created before
    the pin moved keeps whatever it already had.
    """
    backend = _imported_copy("soma_retargeter", "eden-soma-retargeter")
    if backend is None:
        raise ImportError(f"The SomaRetargeter backend is not installed. Install it with {_INSTALL_HINT}.")
    backend_version = backend.version
    if backend_version is None:
        raise ImportError(
            f"soma_retargeter is importable from {backend.location} but that copy carries no eden-soma-retargeter "
            f"distribution metadata, so its version cannot be checked; Eden needs >={_MIN_BACKEND}. Install it with "
            f"{_INSTALL_HINT}. A backend installed from its own checkout records one once installed with "
            f"`pip install -e`.{_other_metadata(backend, 'eden-soma-retargeter')}"
        )
    if parse(backend_version) < parse(_MIN_BACKEND):
        raise ImportError(
            f"eden-soma-retargeter {backend_version} has no `enable_procedural_transforms`, which "
            f"SomaRetargeter passes to fit against the public 78-joint SOMA rig. Install "
            f">={_MIN_BACKEND} with {_INSTALL_HINT}."
        )
    if body_model not in SMPL_FAMILY:
        return
    soma = _imported_copy("soma", "py-soma-x")
    if soma is None:
        raise ImportError(
            f"body_model={body_model!r} is fitted onto the SOMA rig by py-soma-x, which is not installed. "
            f"Install it with {_INSTALL_HINT}."
        )
    version = soma.version
    if version is None:
        raise ImportError(
            f"body_model={body_model!r} is fitted onto the SOMA rig by py-soma-x (import name `soma`), but the `soma` "
            f"importable from {soma.location} carries no py-soma-x distribution metadata: it is another project's "
            "`soma`, or py-soma-x put on sys.path without installing it, and either way its version cannot be "
            f"checked; Eden needs py-soma-x >={_MIN_PY_SOMA_X}. Install it with {_INSTALL_HINT}."
            f"{_other_metadata(soma, 'py-soma-x')}"
        )
    if parse(version) < parse(_MIN_PY_SOMA_X):
        raise ImportError(
            f"py-soma-x {version} fits near-zero rotations and then blames the GPU for them. "
            f"Install >={_MIN_PY_SOMA_X} with {_INSTALL_HINT}."
        )
