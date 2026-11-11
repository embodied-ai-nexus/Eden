"""The per-run reproducibility manifest (``run.json``).

Eden's canonical run identity is the log-directory basename ``<timestamp>-<run id>``
(resolved in :mod:`eden.__init__`). This module writes a sibling ``run.json`` recording
everything needed to reproduce or trace a run: the *applied* seed, the eden and Genesis
versions, the eden working-tree git state, the RL library that wrote the checkpoints, the
full command line, and the task.

It is deliberately separate from the per-config ``config.json`` envelope (``_meta``),
which is re-stamped on every ``save_as_json`` — including ``deploy_config.json`` — and
describes the *task*, not the *run instance*. ``run.json`` is written exactly once, at
launch, into the (lazily created) log directory.
"""

import json
import os
from typing import Any

from eden.utils.distribution import installed_version
from eden.utils.misc import get_eden_run_provenance, get_package_provenance

RUN_MANIFEST_FILENAME = "run.json"

# Stamped into every manifest so a reader can tell which field set it is looking at. No
# reader branches on it today — they tolerate missing fields by shape — so bumping it is a
# signal, not a switch. v2 added ``rl_library``; v3 records Eden and Genesis each as
# ``{"version", "commit"}`` (Eden adds ``dirty``).
RUN_MANIFEST_SCHEMA_VERSION = 3

# Eden's runner backends and the distribution that actually writes their checkpoints. The
# backend key is Eden's own (``_is_rl_games`` picks it); the distribution name is what
# ``importlib.metadata`` and ``pip install`` know it by.
RL_LIBRARY_DISTRIBUTIONS = {"rsl_rl": "rsl-rl-lib", "rl_games": "rl-games"}


def get_rl_library(backend: str | None) -> dict[str, str | None] | None:
    """The RL library record for ``backend``, or ``None`` when there is nothing to record.

    Parameters
    ----------
    backend : str | None
        Eden's runner-backend key — a key of :data:`RL_LIBRARY_DISTRIBUTIONS`
        (``"rsl_rl"`` / ``"rl_games"``), or ``None`` when the caller doesn't know.

    Returns
    -------
    dict | None
        ``{"backend": ..., "distribution": ..., "version": ...}``, with ``version``
        ``None`` if the distribution isn't installed under that name. ``None`` for an
        unknown or missing ``backend`` — provenance is best-effort and never fabricated.
    """
    distribution = RL_LIBRARY_DISTRIBUTIONS.get(backend or "")
    if distribution is None:
        return None
    # The backend can be known (the config named it) while the distribution is absent under
    # that name — a source checkout on PYTHONPATH, say: the version is None, the backend recorded anyway.
    return {"backend": backend, "distribution": distribution, "version": installed_version(distribution)}


def read_run_manifest(log_dir: str) -> dict[str, Any] | None:
    """Read ``<log_dir>/run.json``, or ``None`` if it is missing or unreadable.

    Best-effort by contract: callers enrich diagnostics with this, so a run predating the
    manifest must degrade to "unrecorded", never to an error.

    Parameters
    ----------
    log_dir : str
        A run's log directory (the one holding ``run.json`` and the checkpoints).

    Returns
    -------
    dict | None
        The parsed manifest, or ``None``.
    """
    path = os.path.join(log_dir, RUN_MANIFEST_FILENAME)
    try:
        with open(path) as f:
            manifest = json.load(f)
    except (OSError, ValueError):  # missing/unreadable, or truncated JSON
        return None
    return manifest if isinstance(manifest, dict) else None


def build_run_manifest(
    *,
    log_dir: str,
    run_id: str | None,
    timestamp: str | None,
    task: str,
    seed: int | None,
    world_size: int,
    run_name: str,
    argv: list[str],
    resumed_from: str | None = None,
    rl_backend: str | None = None,
) -> dict[str, Any]:
    """Assemble the run-manifest dict, resolving the eden/Genesis/RL-library versions.

    Split from :func:`write_run_manifest` so the field set can be asserted without
    touching the filesystem. Not pure: resolving the versions shells out to git and reads
    installed-package metadata, which is why tests stub those three lookups.

    Parameters
    ----------
    log_dir : str
        The run's log directory (its basename is the canonical run identity).
    run_id : str | None
        This run's base-36 id, from :func:`eden.get_run_id`. Taken as an argument rather
        than parsed out of ``log_dir`` so the manifest never has to guess how the basename
        is composed (a later run_name decoration would break basename parsing).
    timestamp : str | None
        This run's creation timestamp, from :func:`eden.get_run_timestamp`.
    task : str
        The registered task name being trained.
    seed : int | None
        The base seed applied to the run's RNGs (``None`` when the backend exposes no seed
        field). Under DDP this is rank 0's seed; rank ``r`` uses ``seed + r`` (see
        ``world_size``), so the whole run reproduces from this base plus the topology.
    world_size : int
        Number of DDP ranks (1 for single-GPU). Makes ``seed`` unambiguous: the per-rank
        seeds are ``seed`` through ``seed + world_size - 1``.
    run_name : str
        The human-facing run name (``""`` when unset).
    argv : list[str]
        The full process command line (typically ``sys.argv``).
    resumed_from : str | None
        Basename of the parent run this one resumed from (``--resume``), or ``None`` for a
        fresh run. Records the fork lineage a resume creates (a resume trains in a new run
        directory rather than continuing the parent in place).
    rl_backend : str | None
        Eden's runner-backend key (``"rsl_rl"`` / ``"rl_games"``) — the library that will
        write this run's checkpoints. Its installed version is resolved and recorded under
        ``rl_library`` (see :func:`get_rl_library`); ``None`` records ``rl_library: null``.

    Returns
    -------
    dict
        The JSON-serializable manifest.
    """
    return {
        "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "run_id": run_id,
        "run_name": run_name,
        "timestamp": timestamp,
        "log_dir": os.path.abspath(log_dir),
        "task": task,
        "seed": seed,
        "world_size": world_size,
        "resumed_from": resumed_from,
        "argv": list(argv),
        # The release version, the commit (a clone's HEAD, else null) and whether that tree
        # was dirty. Genesis is a dependency with no working tree of its own, so it has no
        # dirty flag; its commit is the one a git+ or editable install records.
        "eden": get_eden_run_provenance(),
        "genesis": get_package_provenance("genesis-world"),
        # The RL library owns the checkpoint layout, so replaying a `.pt` needs a
        # compatible version of *it*, not just of eden/genesis.
        "rl_library": get_rl_library(rl_backend),
    }


def write_run_manifest(log_dir: str, manifest: dict[str, Any]) -> str:
    """Write ``manifest`` to ``<log_dir>/run.json`` and return the path.

    Overwrites any existing manifest (a run writes it exactly once, at launch). The
    directory is assumed to exist: the caller resolves ``log_dir`` through
    ``eden.log_dir``, which creates the directory on first access.

    Parameters
    ----------
    log_dir : str
        The run's (already-created) log directory.
    manifest : dict
        The manifest to write, from :func:`build_run_manifest`.

    Returns
    -------
    str
        The path to the written ``run.json``.
    """
    path = os.path.join(log_dir, RUN_MANIFEST_FILENAME)
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    return path
