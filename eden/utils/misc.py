"""Miscellaneous helpers (env-index sanitization, entity resolution, hashing, snake_case, key loading, paths)."""

import datetime
import glob
import hashlib
import os
import re
import reprlib
import socket
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from importlib import metadata
from typing import Any

import numpy as np
import torch

# Top-level self-import, safe because `en.*` attributes are only read at call time -- same
# pattern as eden.utils.configs. `import eden` does not import this module at all.
import eden as en
from eden.utils.distribution import direct_url_record, editable_checkout, get_eden_version


def resolve_named_entity(entities: dict, name: str, *, hint: str | None = None):
    """Look up ``name`` in an env's ``entities``, naming what is there when it is missing.

    Shared by ``EnvBase.get_entity``, ``ManagerTermBase.resolve_entity`` and
    ``PolicyBase.require_entity``: the name is always one someone typed, so the useful failure
    lists the names that exist, plus ``hint`` for where the caller declares its own. Takes the
    mapping, not the env, so a bare-``entities`` stub resolves too.
    """
    try:
        return entities[name]
    except KeyError:
        message = f"No scene entity named '{name}'; this env has {sorted(entities)}."
        raise KeyError(f"{message} {hint}" if hint else message) from None


def _on_host(idx) -> torch.Tensor:
    """``idx`` as a CPU tensor; a list is built there directly, whatever torch's default device is."""
    return idx.cpu() if isinstance(idx, torch.Tensor) else torch.as_tensor(idx, device="cpu")


def contiguous_range(idx) -> range | None:
    """``idx`` as the equivalent ``range`` if it counts up one at a time from its first entry, else None.

    The form to hand Genesis for a fixed DOF / q selection. Genesis turns a ``range`` into a slice,
    while an index tensor takes its advanced-indexing path: about 5x the host time per call (32 vs
    6 us for ``get_dofs_position`` on a G1), a GPU->CPU sync in every ``control_dofs_*`` call, and no
    bool-mask fast path in the setters (``RigidEntity.setters_take_every_row``). Reads ``idx`` on the
    host, so resolve it once, in ``build()``, and keep the tensor for anything that does tensor math.

    None for a selection that counts from the end: Genesis turns ``range(-2, 0)`` into ``slice(-2, 0)``,
    which selects nothing, where the index tensor ``[-2, -1]`` it would replace selects the last two.
    """
    host = _on_host(idx)
    if host.ndim != 1 or len(host) == 0:
        return None
    start = int(host[0])
    if start < 0:
        return None
    if torch.equal(host, torch.arange(start, start + len(host), dtype=host.dtype, device="cpu")):
        return range(start, start + len(host))
    return None


def as_genesis_index(idx: torch.Tensor) -> range | torch.Tensor:
    """``idx`` as its :func:`contiguous_range` when it has one, else ``idx`` itself — the selector to hand Genesis.

    Same order, same entries, so the two forms select identical DOFs; only the Genesis code path differs.
    Resolve it once, in ``build()`` (it reads ``idx`` on the host).
    """
    as_range = contiguous_range(idx)
    return idx if as_range is None else as_range


class DofRuns(tuple):
    """A non-contiguous DOF selection as its contiguous runs: ``((dofs_range, columns_slice), ...)``.

    What :func:`as_genesis_control_index` hands a ``RigidEntity.control_dofs_*`` setter for a selection
    with gaps (a parallel gripper's two driver joints, ``[0, 3]``): the setter writes each run through
    its own ``range`` — the run's DOFs from the value's ``columns_slice`` — instead of once through an
    index tensor, which Genesis' ``control_dofs_*`` turns into a GPU->CPU sync twice per call.
    Only the control setters accept it; hand a getter the index tensor or :func:`as_genesis_index`.
    """

    __slots__ = ()


class DofPermutation(tuple):
    """A DOF selection that permutes a contiguous block, as ``(dofs_range, order)``.

    What :func:`as_genesis_control_index` hands a ``RigidEntity.control_dofs_*`` setter when ``idx``
    names every DOF of ``dofs_range`` exactly once in another order (a quadruped's joints in leg-major
    order over a joint-major asset): the setter writes the whole block through the ``range`` from
    ``value.index_select(-1, order)`` — ``order[j]`` is the value column of the block's ``j``-th DOF —
    instead of through an index tensor, which Genesis' ``control_dofs_*`` syncs on twice per call.
    Same DOFs, same values. Only the control setters accept it.
    """

    __slots__ = ()

    @property
    def dofs(self) -> range:
        return self[0]

    @property
    def order(self) -> torch.Tensor:
        return self[1]


#: Past this many runs, one index-tensor call is cheaper on the host than a call per run.
_MAX_CONTROL_RUNS = 4


def as_genesis_control_index(
    idx: torch.Tensor, device: torch.device | str | None = None
) -> range | DofPermutation | DofRuns | torch.Tensor:
    """``idx`` as the selector to hand a ``RigidEntity.control_dofs_*`` setter.

    A ``range`` when ``idx`` is contiguous (as :func:`as_genesis_index`); a :class:`DofPermutation`
    when it reorders a contiguous block; a :class:`DofRuns` when it splits into at most a few
    contiguous runs; else ``idx`` itself. Every form writes the same DOFs from the same value columns.
    ``device`` is where the permutation's ``order`` lives — the values' device; ``idx``'s by default.
    Resolve it once, in ``build()`` (it reads ``idx`` on the host).
    """
    as_range = contiguous_range(idx)
    if as_range is not None:
        return as_range
    host = _on_host(idx).reshape(-1).tolist()
    if not host:
        return idx
    lo = min(host)
    if lo < 0:
        # Counted from the end: a run or block `range` would select something else (`contiguous_range`).
        return idx
    if sorted(host) == list(range(lo, lo + len(host))):
        order = sorted(range(len(host)), key=host.__getitem__)
        if device is None:
            device = idx.device if isinstance(idx, torch.Tensor) else "cpu"
        return DofPermutation((range(lo, lo + len(host)), torch.tensor(order, dtype=torch.long, device=device)))
    runs, start = [], 0
    for i in range(1, len(host) + 1):
        if i == len(host) or host[i] != host[i - 1] + 1:
            runs.append((range(host[start], host[i - 1] + 1), slice(start, i)))
            start = i
    return DofRuns(runs) if len(runs) <= _MAX_CONTROL_RUNS else idx


def is_env_mask(envs_idx) -> bool:
    """Whether ``envs_idx`` is a bool env mask (rather than a slice, ``None`` or int ids)."""
    return isinstance(envs_idx, torch.Tensor) and envs_idx.dtype == torch.bool


def _selected_rows(values: torch.Tensor, envs_idx: slice | torch.Tensor) -> torch.Tensor:
    """``values[envs_idx]``, keeping the env axis for a scalar selector (a 0-dim index tensor or an int)."""
    selected = values[envs_idx]
    return selected.unsqueeze(0) if selected.ndim < values.ndim else selected


def sum_over_envs(values: torch.Tensor, envs_idx: slice | torch.Tensor) -> torch.Tensor:
    """``values[envs_idx].sum(dim=0)``, without a GPU->CPU sync when ``envs_idx`` is a bool mask.

    Gathering with a bool mask (``values[mask]``) has to size its output, which is a sync; the
    auto-reset in ``RLEnvBase.step`` hands every manager's ``reset()`` exactly such a mask
    (``reset_buf``), so a per-term gather for a log value costs one sync per term per step.
    A mask is reduced in place instead. A bool ``values`` sums to the count of ``True`` entries.
    """
    if is_env_mask(envs_idx):
        return torch.where(envs_idx.view(-1, *([1] * (values.ndim - 1))), values, 0).sum(dim=0)
    return _selected_rows(values, envs_idx).sum(dim=0)


def mean_over_envs(values: torch.Tensor, envs_idx: slice | torch.Tensor) -> torch.Tensor:
    """``values[envs_idx].mean(dim=0)`` — zero when nothing is selected — without a sync for a bool mask.

    See :func:`sum_over_envs`.
    """
    if is_env_mask(envs_idx):
        return sum_over_envs(values, envs_idx) / envs_idx.sum().clamp(min=1)
    selected = _selected_rows(values, envs_idx)
    return selected.sum(dim=0) / max(selected.shape[0], 1)


def assign_over_envs(target: torch.Tensor, envs_idx: slice | torch.Tensor, source: torch.Tensor) -> None:
    """``target[envs_idx] = source[envs_idx]`` for a full-batch ``source``, without a sync for a bool mask.

    ``target[mask] = source[mask]`` gathers first, which is a GPU->CPU sync (see :func:`sum_over_envs`);
    a mask is applied with ``torch.where`` into ``target`` instead.
    """
    if is_env_mask(envs_idx):
        torch.where(envs_idx.view(-1, *([1] * (target.ndim - 1))), source, target, out=target)
    else:
        target[envs_idx] = source[envs_idx]


#: The ``(mask, ids)`` pair :func:`sharing_env_indices` publishes for the reset it brackets.
_SHARED_ENV_INDICES: ContextVar[tuple[torch.Tensor, torch.Tensor] | None] = ContextVar(
    "eden_shared_env_indices", default=None
)


def env_indices(envs_idx: slice | torch.Tensor) -> slice | torch.Tensor:
    """A bool env mask as the index tensor of its ``True`` entries; a slice or an index tensor unchanged.

    For code that must *gather* the selected rows (``values[ids]``): every ``values[mask]`` gather is
    a GPU->CPU sync, and the one ``nonzero()`` here pays for all of them. Keep the mask for
    scalar fills (``buf[mask] = 0.0``) and ``torch.where``: a scalar written through an *index*
    tensor is the one form that syncs the other way round. Keep it for Genesis' state setters too
    where they take one on the device (``RigidEntity.setters_take_every_row``, with a ``range``
    selection): handed ids, a setter writes its error flag through them, a sync per call.

    Inside :func:`sharing_env_indices` the mask it was opened on costs nothing here: its ids were
    taken once, for the whole reset.
    """
    if not is_env_mask(envs_idx):
        return envs_idx
    shared = _shared_ids(envs_idx)
    return envs_idx.nonzero()[:, 0] if shared is None else shared


def _shared_ids(mask: torch.Tensor) -> torch.Tensor | None:
    """The ids :func:`sharing_env_indices` took for ``mask`` (matched by identity), else ``None``."""
    shared = _SHARED_ENV_INDICES.get()
    return shared[1] if shared is not None and shared[0] is mask else None


@contextmanager
def sharing_env_indices(envs_idx) -> Iterator[None]:
    """Take a bool ``envs_idx`` mask's ids once, for every :func:`env_indices` call on it inside.

    The auto-reset hands the same ``reset_buf`` to the scene reset, every RESET event and every
    manager. Each consumer that gathers selected rows would otherwise take its own ``nonzero()`` —
    one GPU->CPU sync apiece, 3-9 a step on the shipped tasks. Bracketing the reset in this
    takes the one sync up front and hands its ids to all of them, along with the selected count
    (:func:`sanitize_envs_idx`'s ``return_n_envs``) that gates the reset.

    The ids are matched to the mask by identity, so a mask derived from it (``mask & active``)
    takes its own ``nonzero()``. The mask must not be written while the bracket is open,
    or the ids would describe the old selection; the env opens it on a ``reset_buf`` that nothing
    writes during a reset. Anything but a bool mask (``None``, a slice, an index tensor) passes
    through untouched.
    """
    if not is_env_mask(envs_idx):
        yield
        return
    token = _SHARED_ENV_INDICES.set((envs_idx, envs_idx.nonzero()[:, 0]))
    try:
        yield
    finally:
        _SHARED_ENV_INDICES.reset(token)


def _selected_env_count(idx: slice | torch.Tensor, num_envs: int) -> int:
    """Return the number of environments selected by a normalized index.

    A bool mask is counted on the host, a GPU->CPU sync — except the one :func:`sharing_env_indices`
    is bracketing, whose ids already carry the count.
    """
    if isinstance(idx, slice):
        start, stop, step = idx.indices(num_envs)
        return len(range(start, stop, step))
    if idx.dtype == torch.bool:
        shared = _shared_ids(idx)
        return int(idx.sum().item()) if shared is None else int(shared.numel())
    return int(idx.numel())


def _check_envs_idx_in_range(envs_idx, num_envs: int, name: str) -> None:
    """Range-check an integer ``envs_idx`` before it reaches a kernel.

    Genesis's ``sanitize_index`` disabled its own out-of-range check for
    performance, so an out-of-bounds index would otherwise surface as an opaque
    CUDA device-side assert far from its cause. We re-check here,
    but only for integer selectors cheap to inspect on the host (Python/numpy scalars,
    sequences, numpy arrays, CPU tensors); on-device tensors come from trusted
    internal construction (e.g. auto-reset) and are skipped to avoid a GPU sync
    on the hot reset path. Boolean masks are not integer indices, so they are
    left for ``gs_sanitize_index`` to convert (torch bool masks are length-checked
    upstream in :func:`sanitize_envs_idx`). A bare bool, or a list/tuple of them, is
    neither: ``bool`` subclasses ``int``, so Genesis would read ``True`` as the index 1
    and ``[True, False]`` as the indices ``[1, 0]``, and these raise instead.
    """
    if isinstance(envs_idx, (bool, np.bool_)):
        raise IndexError(
            f"{name}={envs_idx!r} is a bool, not an env selector: Genesis would read it as the index "
            f"{int(envs_idx)}. Fix: pass {name}=None for every env, {name}=[] for none, or a torch.bool mask of "
            f"shape ({num_envs},), e.g. torch.ones({num_envs}, dtype=torch.bool, device=env.device). Do not coerce "
            f"it with int(): that still selects env {int(envs_idx)}."
        )
    if isinstance(envs_idx, (int, np.integer)):
        values = (envs_idx,)
    elif isinstance(envs_idx, range):
        values = (envs_idx[0], envs_idx[-1]) if envs_idx else ()  # monotonic: its ends are its extremes
    elif isinstance(envs_idx, (tuple, list)):
        values = envs_idx
        pos = next((i for i, v in enumerate(values) if isinstance(v, (bool, np.bool_))), None)
        if pos is not None:
            raise IndexError(
                f"{name}={reprlib.repr(envs_idx)} holds the bool {values[pos]!r} at position {pos}: Genesis would "
                f"read it as the index {int(values[pos])}, not as a mask entry. Fix: for a mask, pass a torch.bool "
                f"tensor of shape ({num_envs},), e.g. torch.tensor({name}, dtype=torch.bool, device=env.device); "
                f"for indices, pass ints only."
            )
    elif isinstance(envs_idx, np.ndarray):
        if envs_idx.dtype == bool:
            return  # boolean mask, not integer indices
        values = envs_idx.reshape(-1).tolist()
    elif isinstance(envs_idx, torch.Tensor) and envs_idx.device.type == "cpu":
        if envs_idx.dtype == torch.bool:
            return  # boolean mask, not integer indices
        values = envs_idx.reshape(-1).tolist()
    else:
        return  # on-device tensor: trust the caller, don't pay for a sync
    for v in values:
        if not (-num_envs <= v < num_envs):
            raise IndexError(
                f"{name} contains index {v}, out of range for num_envs={num_envs} (valid: {-num_envs}..{num_envs - 1})."
            )


def _device_arange(start: int, stop: int, step: int) -> torch.Tensor:
    """The index tensor Genesis' ``sanitize_index`` makes of a ``range``, generated on the device.

    Genesis builds it with ``torch.as_tensor(range(...), device=gs.device)``, a host->device upload —
    a GPU->CPU sync — on every call, and the per-DOF / per-link info getters sanitize their default
    ``envs_idx=None`` this way every step. ``torch.arange`` on the device is the same int
    tensor with no upload.
    """
    import genesis as gs

    return torch.arange(start, stop, step, dtype=gs.tc_int, device=gs.device)


def sanitize_envs_idx(
    envs_idx: int | range | slice | tuple[int, ...] | list[int] | torch.Tensor | np.ndarray | None,
    num_envs: int,
    *,
    prefer_slice: bool = True,
    name: str = "envs_idx",
    return_n_envs: bool = False,
) -> tuple[slice | torch.Tensor, int] | slice | torch.Tensor:
    """Normalize environment indices with an optional slice fast-path.

    Parameters
    ----------
    envs_idx: int | range | slice | tuple[int, ...] | list[int] | torch.Tensor | np.ndarray | None
        Environment selector. A mask must be a bool ``torch.Tensor`` or ``np.ndarray`` of shape
        ``(num_envs,)``; a bare bool or a list/tuple of bools raises ``IndexError``.
    num_envs: int
        Total number of environments.
    prefer_slice: bool
        If True, preserve slices for faster backend indexing when possible.
    name: str
        Name used in validation errors.
    return_n_envs: bool
        If True, also return the number of environments selected by the index.

    Returns
    -------
    envs_idx: slice | torch.Tensor
        Normalized environment indices.
    n_envs: int, optional
        Number of environments selected by the index. Only returned if return_n_envs is True.
    """
    # Deferred, not module-scope: importing any ``genesis.*`` submodule runs
    # ``genesis/__init__.py``, so this one name would cost every importer of this module the full
    # ``import genesis``. The per-call price of importing it here is a ``sys.modules`` lookup:
    # measured at 59 ns against this function's own 19.8 us, i.e. 0.3%, which is why it is not
    # worth caching the name in a module global. By the time anything sanitizes an env index
    # Genesis is long since imported.
    from genesis.utils.misc import sanitize_index as gs_sanitize_index

    if num_envs < 0:
        raise ValueError(f"num_envs must be non-negative, got {num_envs}")

    if is_env_mask(envs_idx):
        if envs_idx.numel() != num_envs:
            raise IndexError(f"{name} boolean mask has length {envs_idx.numel()}, expected num_envs={num_envs}.")
        out = envs_idx
    elif envs_idx is None:
        out = slice(None) if prefer_slice else _device_arange(0, num_envs, 1)
    elif isinstance(envs_idx, slice):
        start, stop, step = envs_idx.indices(num_envs)
        out = slice(start, stop, step) if prefer_slice else _device_arange(start, stop, step)
    else:
        # Genesis's sanitize_index disabled its out-of-range check (perf), so
        # validate here before delegating the int-tensor conversion.
        _check_envs_idx_in_range(envs_idx, num_envs, name)
        out = gs_sanitize_index(envs_idx, -1, num_envs, 0, name)

    if return_n_envs:
        return out, _selected_env_count(out, num_envs)
    return out


def invert_segmentation_idx_dict(segmentation_idx_dict: dict) -> dict[int, list[int]]:
    """Invert Genesis' ``segmentation_idx_dict`` into ``entity_idx -> [segmentation ids]``.

    Genesis maps *segmentation id -> scene element*, where the element is a bare entity index at
    ``segmentation_level="entity"`` and an ``(entity_idx, link_idx, ...)`` tuple at ``"link"`` /
    ``"geom"``. Callers want the reverse, and at link/geom level an entity owns **several** ids —
    one per link/geom. Keeping only one (what a plain ``d[entity_idx] = seg_idx`` loop does, since
    the last write wins) silently reduces the entity to a single link: a masked point cloud loses
    most of the object, and a background mask stops covering most of the background.
    """
    entity_seg_ids: dict[int, list[int]] = {}
    for seg_idx, scene_elem in segmentation_idx_dict.items():
        entity_idx = scene_elem[0] if isinstance(scene_elem, (list, tuple)) else scene_elem
        entity_seg_ids.setdefault(entity_idx, []).append(seg_idx)
    return entity_seg_ids


def verify_file_hash(path: str, expected_hash: str) -> bool:
    """Verify file integrity using SHA256."""
    if not os.path.exists(path):
        return False

    sha256_hash = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256_hash.update(chunk)

    return sha256_hash.hexdigest() == expected_hash


def string_to_sha256_hex(text: str) -> str:
    """Convert a string to the hex SHA-256 hash of its UTF-8 encoding."""
    encoded_string = text.encode("utf-8")
    hasher = hashlib.sha256()
    hasher.update(encoded_string)
    hex_digest = hasher.hexdigest()
    return hex_digest


def to_snake_case(camel_case_string):
    """Convert a CamelCase string to snake_case using regular expressions."""
    if not camel_case_string:
        return ""

    # Add an underscore before each capitalized word (an uppercase letter followed by lowercase).
    s1 = re.sub("([^_])([A-Z][a-z]+)", r"\1_\2", camel_case_string)
    # Handle cases like "HTTP" by adding an underscore between consecutive uppercase letters.
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def _display_greeting(INFO_length):
    from eden import logger

    try:
        terminal_size = os.get_terminal_size()[0]
    except OSError:
        terminal_size = 80
    wave_width = int((terminal_size - INFO_length - 11) / 2)
    if wave_width % 2 == 0:
        wave_width -= 1
    wave_width = max(0, min(38, wave_width))
    bar_width = wave_width * 2 + 8
    wave = ("┈┉" * wave_width)[:wave_width]
    logger.info(f"~<╭{'─' * (bar_width)}╮>~")
    logger.info(f"~<│{wave}>~ ~~~~< Eden >~~~~ ~<{wave}│>~")
    logger.info(f"~<╰{'─' * (bar_width)}╯>~")


def get_package_provenance(package_name: str) -> dict[str, str | None]:
    """Which ``package_name`` is installed, for provenance: ``{"version": ..., "commit": ...}``, each ``None`` when
    unknown (both are, when the package is not installed).

    ``version`` is the distribution's (``1.4.0``). ``commit`` is the git commit it was installed from: the one a
    ``git+`` install recorded (PEP 610's ``direct_url.json``), or the ``HEAD`` of the checkout an editable install
    points at; an install from an index has none. Two fields, not one holding either, so whoever reads a saved record
    never has to tell a commit from a version by its shape. :func:`get_eden_provenance` is the same record for Eden.
    """
    try:
        dist = metadata.distribution(package_name)
    except metadata.PackageNotFoundError:
        return {"version": None, "commit": None}
    return {"version": dist.version or None, "commit": _installed_commit(dist)}


def _installed_commit(dist: metadata.Distribution) -> str | None:
    """The commit ``dist`` was installed from, read from its ``direct_url.json``; ``None`` when it records none."""
    if commit := direct_url_record(dist).get("vcs_info", {}).get("commit_id"):
        return str(commit)
    checkout = editable_checkout(dist)
    if checkout is None:
        return None
    try:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=checkout, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError):  # the checkout moved, or is not a git repository
        return None
    return head.decode("utf-8").strip()


def _eden_checkout() -> tuple[str, str] | None:
    """``(root, HEAD)`` of the git checkout whose ``HEAD`` holds this ``eden`` package, or ``None`` when none does (a
    wheel in site-packages, Eden staged but not yet committed, or no ``git``).

    misc.py is ``<root>/eden/utils/misc.py``. The test is whether ``HEAD`` contains this file, not whether ``<root>``
    is the work tree's top level, because two layouts share an enclosing work tree and need opposite answers: Eden
    vendored as a subdirectory of a larger repository is committed there, and that repository's ``HEAD`` is the code
    that ran; a wheel installed into a venv inside some other repository is not, and that ``HEAD`` is not Eden's.

    One ``git`` call, and the path is given relative to ``<root>`` (``HEAD:./...`` resolves against the working
    directory), so no absolute path reaches git: on Windows its drive letter's case may differ from git's own.
    """
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(os.path.abspath(__file__)))))
    try:
        # Prints HEAD, then misc.py's blob in HEAD; fails when HEAD does not contain misc.py.
        head = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD", "HEAD:./eden/utils/misc.py"], cwd=root, stderr=subprocess.DEVNULL
            )
            .decode("utf-8")
            .split()[0]
        )
    except Exception:
        return None
    return root, head


def get_eden_commit() -> str | None:
    """The ``HEAD`` of the git checkout ``eden`` is imported from, or ``None`` outside one.

    The commit is for developers working in a clone. Every install, a ``pip install "eden-world @ git+..."`` included,
    is identified by its release version alone (:func:`get_eden_version`), on purpose.

    Found by path, the way :func:`get_eden_working_tree_dirty` finds the dirty flag, so the two always describe the
    same tree. Not by distribution name through ``importlib.metadata``, which takes the first match on ``sys.path``:
    under ``python -m`` that is the repo root, where an in-tree setuptools build leaves an ``eden_world.egg-info``
    with no ``direct_url.json``, so the commit would read back as the bare version.
    """
    checkout = _eden_checkout()
    return checkout[1] if checkout is not None else None


def get_eden_provenance() -> dict[str, str | None]:
    """Which Eden this is, for provenance: ``{"version": "0.1.0", "commit": <HEAD in a clone, else None>}``, the
    record :func:`get_package_provenance` gives any other package. A run's own record adds ``dirty``
    (:func:`get_eden_run_provenance`).
    """
    return {"version": get_eden_version() or None, "commit": get_eden_commit()}


def get_eden_run_provenance() -> dict[str, str | bool | None]:
    """:func:`get_eden_provenance` plus ``dirty`` (:func:`get_eden_working_tree_dirty`), for a run's own record
    (``run.json``, an export's ``policy_spec.json``). One checkout lookup answers both, so the commit and the dirty
    flag always describe the same tree.
    """
    checkout = _eden_checkout()
    return {
        "version": get_eden_version() or None,
        "commit": checkout[1] if checkout is not None else None,
        "dirty": _working_tree_dirty(checkout[0]) if checkout is not None else None,
    }


def format_provenance(provenance: dict) -> str:
    """A provenance record for a log line: ``0.1.0 (c246ebd)``, the version and the commit cut to 7 characters, either
    alone when the other is unknown, or ``unknown``.
    """
    version, commit = provenance.get("version"), provenance.get("commit")
    if version and commit:
        return f"{version} ({commit[:7]})"
    return version or (commit[:7] if commit else "unknown")


def describe_provenance_change(saved: object, current: dict) -> str | None:
    """How ``current`` differs from the ``saved`` record, as ``saved 0.1.0 (c246ebd), running 0.1.0 (3f2a1b2)``, or
    ``None`` when the two match or cannot be compared.

    Commits are compared when both records have one, else versions: a file saved in a clone and read from a wheel of
    the same release matches. A ``saved`` that is not a record (a file written before these fields existed) cannot be
    compared.
    """
    if not isinstance(saved, dict):
        return None
    for key in ("commit", "version"):
        if saved.get(key) and current.get(key):
            if saved[key] == current[key]:
                return None
            return f"saved {format_provenance(saved)}, running {format_provenance(current)}"
    return None


# Logging Utils
def get_now():
    return datetime.datetime.now().strftime("%y%b%dT%H%M%S")


# Path-separator, URL-reserved, and whitespace characters, collapsed to a single "-" so a
# run_name is safe both as a log-dir path segment and as a wandb run name.
_RUN_NAME_UNSAFE = re.compile(r"[/\\#?%:\s]+")


def sanitize_run_name(run_name: str) -> str:
    """Return ``run_name`` made safe as a path segment / wandb name (or ``""`` if empty).

    Collapses runs of path-separator, URL-reserved, and whitespace characters to a single
    ``-``, trims leading/trailing separators, and caps the length at 64. Used both for the
    log-dir basename segment and for ``runner_options.run_name`` so the two always agree.

    Parameters
    ----------
    run_name : str
        The raw, user-supplied run name.

    Returns
    -------
    str
        The sanitized run name, empty when the input is empty or all-unsafe.
    """
    return _RUN_NAME_UNSAFE.sub("-", run_name.strip()).strip("-")[:64]


def create_symlink(src, dst):
    if not os.path.exists(src):
        raise FileNotFoundError(f"[*] {src} not found")
    if os.path.islink(dst):
        os.remove(dst)
    os.symlink(src, dst)


def get_eden_working_tree_dirty() -> bool | None:
    """Return whether the eden source tree has uncommitted changes, for run provenance.

    Complements the commit reported by :func:`get_eden_commit`: that sha is a
    lie for reproducibility if the tree it names had local edits, and a commit alone can
    never convey that. Consumed by the run manifest (:mod:`eden.utils.run_manifest`).

    Best-effort: a non-editable (wheel) install, an unavailable ``git``, or any
    subprocess failure yields ``None`` (unknown) rather than raising — writing a run
    manifest must never abort training.

    "Dirty" means the *tracked* source differs from ``HEAD`` (staged or unstaged edits).
    Untracked files are excluded (``--untracked-files=no``): a run always drops new files
    into the tree (its own logs, ``__pycache__``, scratch), so counting those would flip
    the flag to ``True`` on essentially every run and drain it of meaning — the question
    for reproducibility is whether the committed code was modified.

    For Eden vendored inside a larger repository, the flag is that repository's: an edit anywhere in its tracked files
    counts, because the commit beside it (:func:`get_eden_commit`) is that repository's ``HEAD`` and names all of them.

    Returns
    -------
    bool | None
        ``True`` if tracked files differ from ``HEAD``, ``False`` if clean, and ``None``
        when it could not be determined (no git checkout / git unavailable).
    """
    checkout = _eden_checkout()
    return _working_tree_dirty(checkout[0]) if checkout is not None else None


def _working_tree_dirty(root: str) -> bool | None:
    """Whether the work tree at ``root`` has tracked changes, or ``None`` when ``git`` cannot say."""
    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"], cwd=root, stderr=subprocess.DEVNULL
        ).decode("utf-8")
    except Exception:
        return None
    return bool(status.strip())


def set_api_keys_from_files():
    """Load API keys from ``eden/generators/keys/`` into environment variables.

    Reads all ``*.key`` files from the ``eden/generators/keys/`` directory,
    extracts the API key from each file, and sets them as environment variables.

    The environment variable name is the filename without its extension, upper-cased:
    a file named 'my_api.key' sets an environment variable named 'MY_API'.
    """
    keys_directory = os.path.join(os.path.dirname(os.path.dirname(__file__)), "generators", "keys")
    key_files_pattern = os.path.join(keys_directory, "*.key")

    if not os.path.isdir(keys_directory):
        # No keys directory is the normal state for most installs.
        en.logger.debug(f"No API keys directory at {keys_directory}; nothing to load.")
        return

    key_files = glob.glob(key_files_pattern)

    if not key_files:
        en.logger.debug(f"No API key files found in {keys_directory}")
        return

    for key_file_path in key_files:
        try:
            # Derive the API name from the filename
            # e.g., /path/to/eden/generators/keys/some_api_name.key -> some_api_name
            base_name = os.path.basename(key_file_path)
            api_name_parts = base_name.split(".")[:-1]  # Remove .key extension
            api_name = "_".join(api_name_parts)

            if not api_name:
                en.logger.warning(f"Could not derive API name from file: {key_file_path}")
                continue

            # The environment variable name (e.g., SOME_API_NAME)
            env_var_name = f"{api_name.upper()}"

            with open(key_file_path, "r") as f:
                api_key = f.read().strip()

            if api_key:
                os.environ[env_var_name] = api_key
                en.logger.info(f"Set environment variable: {env_var_name}")
            else:
                en.logger.warning(f"API key file is empty: {key_file_path}")

        except Exception as e:
            en.logger.error(f"Error processing file {key_file_path}: {e}")


def list_host_ips():
    """Best-effort list of this host's LAN IPv4 addresses, most-likely-reachable first.

    Used by the teleop bridges to print the address an XR headset should dial.
    """
    ips = []

    # Primary outbound interface (the one the headset most likely shares).
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.connect(("8.8.8.8", 80))
        ips.append(sock.getsockname()[0])
        sock.close()
    except Exception:
        pass

    # Everything else resolvable from the hostname.
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = info[4][0]
            if not ip.startswith("127.") and ip not in ips:
                ips.append(ip)
    except Exception:
        pass

    return ips


def get_src_dir():
    return os.path.dirname(os.path.dirname(__file__))


#: Environment variable that moves the root Eden downloads assets into (see :func:`get_assets_dir`).
ASSETS_DIR_ENV_VAR = "EDEN_ASSETS_DIR"


def get_assets_dir(*parts: str) -> str:
    """Root that Eden downloads Hugging Face assets into, joined with ``parts``.

    Robots, objects, scenes, policy checkpoints and benchmark-suite assets are fetched file by file
    into this directory. It is ``$EDEN_ASSETS_DIR`` when that is set and non-empty (``~`` expanded), else
    ``get_cache_dir("assets")``: ``$XDG_CACHE_HOME/eden/assets``, by default ``~/.cache/eden/assets``, which
    is per-user, writable without root, and shared by every venv and reinstall.

    A relative ``$EDEN_ASSETS_DIR`` raises ``ValueError``. Read on every call, it would name a different
    directory each time the process (or a worker a launcher starts elsewhere) changes its working directory.

    Read on every call, like :func:`get_cache_dir`. Nothing caches it: an unset ``local_dir`` names a
    subdirectory of it that is resolved at load time (:func:`resolve_local_dir`), so a saved config's
    ``local_dir`` stays unset.

    Empty ``parts`` are skipped, so ``get_assets_dir("")`` (an asset family's unset ``assets_subdir``)
    is the root itself rather than the root with a trailing separator.

    This function neither creates the directory nor checks that it is writable. The first download
    does both, and raises naming the path and ``EDEN_ASSETS_DIR`` when it cannot write there.
    """
    override = os.environ.get(ASSETS_DIR_ENV_VAR)
    if not override:
        return get_cache_dir("assets", *(part for part in parts if part))
    root = os.path.expanduser(override)
    if not os.path.isabs(root):
        raise ValueError(
            f"{ASSETS_DIR_ENV_VAR}={override!r} is a relative path, so it would move with the working directory "
            f"of whichever process reads it. Export an absolute one: export {ASSETS_DIR_ENV_VAR}={os.path.abspath(root)}"
        )
    return os.path.join(root, *(part for part in parts if part))


def resolve_local_dir(local_dir: str | None, assets_subdir: str = "") -> str:
    """Where an asset owner caches its files: ``local_dir`` when set, else ``assets_subdir`` under the asset root.

    The one statement of the rule. ``EntityOptions``, ``PCDOptions`` and ``PolicyBase`` each expose it as a
    ``resolve_local_dir()`` method over their own fields; :func:`eden.utils.assets.get_asset_path` applies it with
    no subdirectory. Resolved on each call, so ``EDEN_ASSETS_DIR`` moves every unset ``local_dir``.
    """
    return local_dir or get_assets_dir(assets_subdir)


def get_cache_dir(*parts: str) -> str:
    """Per-user Eden cache directory, joined with ``parts``.

    Everything Eden *writes* rather than ships goes here: writable without root (unlike the
    installed package), and not shared between users on a multi-tenant host (unlike ``/tmp``).
    """
    root = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return os.path.join(root, "eden", *parts)


def get_gs_assets_dir():
    # Resolve the actual Genesis package root (e.g. .../site-packages/genesis)
    # instead of matching arbitrary nested paths that merely contain "genesis".
    try:
        genesis_module = __import__("genesis")
        genesis_dir = os.path.dirname(os.path.abspath(genesis_module.__file__))
        assets_dir = os.path.join(genesis_dir, "assets")
        if os.path.isdir(assets_dir):
            return assets_dir
    except Exception:
        pass

    # Fallback: recover package root from sys.path entries that may include
    # nested paths like ".../genesis/ext/.../bin".
    for path in sys.path:
        normalized_path = os.path.abspath(path)
        marker = f"{os.sep}genesis{os.sep}"
        if marker in normalized_path:
            genesis_root = normalized_path.split(marker, 1)[0] + marker.rstrip(os.sep)
        elif os.path.basename(normalized_path) == "genesis":
            genesis_root = normalized_path
        else:
            continue

        assets_dir = os.path.join(genesis_root, "assets")
        if os.path.isdir(assets_dir):
            return assets_dir

    raise FileNotFoundError(
        "Could not locate the Genesis 'assets' directory. "
        "Tried resolving it via the 'genesis' package location and entries in sys.path."
    )


def to_jsonish(value: Any) -> Any:
    """Normalise a value so it compares equal to itself across a ``torch.save`` round-trip.

    Tuples come back as lists, and an array-like compares elementwise — comparing one directly
    raises ``Boolean value of Tensor is ambiguous``. ``tolist()`` covers torch tensors, numpy
    arrays and numpy scalars alike.
    """
    if hasattr(value, "tolist"):
        return to_jsonish(value.tolist())
    if isinstance(value, (tuple, list)):
        return [to_jsonish(v) for v in value]
    if isinstance(value, dict):
        return {k: to_jsonish(v) for k, v in value.items()}
    return value
