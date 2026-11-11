"""Base class and registry for motion datasets backed by HuggingFace repos."""

from __future__ import annotations
from typing import TYPE_CHECKING, Callable, ClassVar, Any, TypeVar
from abc import abstractmethod
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
import json
import os
import glob
import re

import numpy as np
import torch
import yaml

import eden as en
from eden.utils import distributed as eden_dist
from eden.utils.common import ConfigurableMixin
from eden.options.datasets import DatasetModifierOptions, DatasetOptions
from eden.utils.motion_reference.utils import (
    derive_body_frame_link_fields,
    estimate_motion_velocities,
    interpolate_body_model,
    interpolate_motion,
)
from eden.utils.motion_reference.motion_data import MotionSequence, BatchMotionSequence
from eden.utils.misc import describe_provenance_change, get_eden_provenance, get_package_provenance, to_jsonish
from eden.utils.registry import Registry
from eden.utils.torch import numpy_to_tensor
from eden.utils.motion_reference.filter_mixin import FilterMixin

if TYPE_CHECKING:
    from eden.options.entities import EntityOptions


# Generic types for _parallel_read_motion_files: subclasses pass file paths
# (typically str) and return per-file MotionSequence, but several datasets
# return tuples (Arctic: 4-tuple, OMOMO/HHI-Assist: 2-tuple). The TypeVars let
# the helper preserve the subclass's actual input/output types.
_TIn = TypeVar("_TIn")
_TOut = TypeVar("_TOut")


# NOTE: a registry for datasets used for MotionTerm
DATASET_REGISTRY = Registry("DATASET")


# Bumped when the on-disk cache layout changes in a backwards-incompatible way.
# v1: pickled BatchMotionSequence containing custom ConcatBatchTensor objects
#     (slow cold-load, no mmap).
# v2: plain-tensor dict with the project's _meta/data envelope convention
#     (cf. eden/utils/configs.py). Layout:
#         {"_meta": {"_format_version": 2, "eden": {"version": ..., "commit": ...}, ...},
#          "sequences": {seq_key: BatchMotionSequence.to_state_dict(), ...}}
#     mmap-friendly via torch.load(mmap=True), so multiple DDP ranks on the
#     same node share the OS page cache instead of each materialising a
#     private copy of the motion library.
# v3: same layout as v2, but ``all_motion_files`` metadata stores each file's
#     path relative to raw_data_path (posix) instead of its bare basename.
#     Directory-qualified list-file filters (``CMU/45/45_01_stageii``) re-run
#     against this metadata on cache load and can only match when the directory
#     components survive the round-trip. Loud-fails v1/v2 caches on load (they
#     carry basename-only metadata that would silently mis-filter) so they get
#     rebuilt rather than quietly training on the wrong subset.
# v4: same layout as v3, but the forward-velocity-kinematics builder now writes a
#     free joint's base angular velocity into the DOFs in the BODY frame (MuJoCo
#     qvel convention) instead of the world frame. Every pre-v4 cache carries
#     ``R(q)``-rotated (corrupted) link/base velocities, worst on tilted/rotating
#     frames. Bumped so those caches loud-fail on load and get rebuilt rather than
#     silently training on corrupted reference velocities.
_CACHE_FORMAT_VERSION = 4

# How a cache's clips were cut by ``drop_first_n_frames`` / ``drop_last_n_frames``, recorded beside the cache identity.
# Absent (caches written before this marker): OMOMO and Weave ignored both settings, and AMASS, PHUMA and HHI kept
# frames wrapped around from the clip's start when ``drop_last_n_frames`` reached past its end — yet every cache
# recorded the values as if applied. 2: every reader that accepts them cuts clips by ``DatasetBase._frame_window``.
# Checked only when either value is non-zero, so the far commoner untrimmed caches still load; a bump of
# ``_CACHE_FORMAT_VERSION`` would have thrown those away too. A reader that already cut clips that way
# (``_drops_cut_before_marker``) has its unmarked caches trusted.
_FRAME_WINDOW_SEMANTICS = 2

# A clip of one frame spans no time (``(T - 1) / framerate = 0``) and has no velocity to estimate, so every clip needs
# two. Enforced when a clip is packed (``DatasetBase._pack_motion_sequence``) and again on the loaded library.
_MIN_CLIP_FRAMES = 2


class _ShortClipError(ValueError):
    """A clip with fewer than ``_MIN_CLIP_FRAMES`` frames — a data problem in one clip, not in the configuration.

    ``DatasetBase._parallel_read_motion_files`` leaves such a clip out with a warning rather than failing the whole
    read; anywhere else it is an ordinary ``ValueError``. ``clip`` names the clip and its frame count for that
    warning, ``fix`` says how to keep it.
    """

    def __init__(self, message: str, clip: str, fix: str):
        super().__init__(message)
        self.clip = clip
        self.fix = fix


# The item ``DatasetBase._parallel_read_motion_files`` is reading on this thread, so an error raised deep in a reader
# (``_pack_motion_sequence``) can name its file in the message itself, which the caller never passes down. A context
# variable rather than an attribute: the reads run concurrently on a thread pool, and each worker sets its own.
_READING_MOTION_FILE: ContextVar[object | None] = ContextVar("_READING_MOTION_FILE", default=None)

# Written into an archive's extraction root once the archive has been fully extracted — see
# ``DatasetBase._extract_once``.
_EXTRACTED_MARKER = ".eden_extracted"


def _as_tensor(
    value: torch.Tensor | np.ndarray | None, dtype: torch.dtype | None = torch.float32
) -> torch.Tensor | None:
    """A numpy array as a ``dtype`` tensor (its own dtype for None; see :func:`numpy_to_tensor`).

    A tensor, or None, is returned as given.
    """
    if value is None or isinstance(value, torch.Tensor):
        return value
    return numpy_to_tensor(value, dtype=dtype)


def _check_hf_narrowing(cls: type[DatasetBase]) -> None:
    """Raise when ``cls``'s ``_hf_allow_patterns`` / ``_hf_data_subdir`` would not narrow the way they look.

    Both fail silently otherwise. HuggingFace filters with ``any(...)`` over the patterns, so an empty tuple matches
    *nothing* and fetches an empty snapshot rather than the whole repo, and a bare string is iterated character by
    character. The subdirectory is joined onto the snapshot root, so an absolute one replaces it outright
    (``os.path.join(root, "/etc")`` is ``"/etc"``) and a ``..`` walks out of it, globbing whatever is there.

    Raises
    ------
    TypeError
        When either has the wrong type, like the sibling ``__init_subclass__`` checks.
    ValueError
        When the patterns are empty or hold an empty pattern, or the subdirectory is empty, absolute,
        backslash-separated, or holds an empty, ``.`` or ``..`` segment.
    """
    patterns, subject = cls._hf_allow_patterns, f"{cls.__name__}._hf_allow_patterns"
    if patterns is not None:
        if not isinstance(patterns, tuple):
            raise TypeError(f"{subject} must be a tuple of glob patterns or None, got {patterns!r}.")
        if not patterns:
            raise ValueError(f"{subject} is empty, which matches no files. Use None to fetch the whole repo.")
        if non_strings := [p for p in patterns if not isinstance(p, str)]:
            raise TypeError(f"{subject} must hold glob pattern strings, got {non_strings!r}.")
        if not all(patterns):
            raise ValueError(f"{subject} must hold non-empty glob patterns, got {patterns!r}.")

    subdir, subject = cls._hf_data_subdir, f"{cls.__name__}._hf_data_subdir"
    if subdir is not None:
        if not isinstance(subdir, str):
            raise TypeError(f"{subject} must be a relative path string or None, got {subdir!r}.")
        if os.path.isabs(subdir) or "\\" in subdir:
            raise ValueError(
                f"{subject} must be a relative '/'-separated path under the snapshot root, got {subdir!r}."
            )
        if any(segment in ("", ".", "..") for segment in subdir.split("/")):
            raise ValueError(
                f"{subject} must not be empty or hold empty, '.' or '..' segments, which escape or no-op the snapshot "
                f"root — got {subdir!r}."
            )


def _check_body_model_fields(
    body_pose_quat: torch.Tensor | None, body_shape: torch.Tensor | None, body_model: str | None, frames: int
) -> None:
    """Raise unless ``_pack_motion_sequence``'s body-model arguments are all given, shaped for ``frames`` frames."""
    from eden.datasets.utils.body_model import BODY_MODELS

    if body_pose_quat is None or body_shape is None or body_model is None:
        given = [n for n, v in (("body_pose_quat", body_pose_quat), ("body_shape", body_shape)) if v is not None]
        raise ValueError(
            f"_pack_motion_sequence takes body_pose_quat, body_shape and body_model together; got "
            f"{', '.join(given) or 'neither tensor'} and body_model={body_model!r}."
        )
    if body_model not in BODY_MODELS:
        raise ValueError(f"Unknown body_model {body_model!r}; known: {sorted(BODY_MODELS)}.")
    if body_pose_quat.ndim != 3 or body_pose_quat.shape[0] != frames or body_pose_quat.shape[-1] != 4:
        raise ValueError(
            f"body_pose_quat must be (num_frames={frames}, num_body_joints, 4) wxyz; got {tuple(body_pose_quat.shape)}."
        )
    if body_shape.ndim != 2 or body_shape.shape[0] != frames:
        raise ValueError(f"body_shape must be (num_frames={frames}, num_betas); got {tuple(body_shape.shape)}.")
    joints = BODY_MODELS[body_model].num_body_joints
    if body_pose_quat.shape[1] != joints:
        raise ValueError(
            f"body_pose_quat has {body_pose_quat.shape[1]} joints but {body_model} has {joints} below the root."
        )


def _body_model_datasets() -> list[str]:
    """Registry keys of the datasets that fill the body-model fields, for an error message."""
    return sorted(
        key
        for key in DATASET_REGISTRY.keys()
        if getattr(DATASET_REGISTRY.get(key), "_supports_body_model_params", False)
    )


def _cache_identity_field_names(cls: type) -> tuple[str, ...]:
    """Every ``_cache_identity_fields`` entry declared by ``cls`` and its bases, base-first.

    Unioned over the MRO rather than inherited whole, so a subclass declares only what *it*
    adds — re-splatting the parent's fields by hand would make dropping one a silent narrowing
    of the very check it belongs to. Same trick as :meth:`ConfigurableMixin.get_parameter_names`.
    """
    names: list[str] = []
    for base in reversed(cls.__mro__):
        for name in base.__dict__.get("_cache_identity_fields", ()):
            if name not in names:
                names.append(name)
    return tuple(names)


def _select_names(
    names_by_key: dict[str, list[str]], idx_by_key: dict[str, slice | list[int] | torch.Tensor]
) -> dict[str, list[str]]:
    """``names_by_key`` with each key in ``idx_by_key`` narrowed to the names at those indices, in index order.

    Builds a new dict rather than writing into the old one, which may be the class-level ``ClassVar`` default.
    """
    selected = dict(names_by_key)
    for key, idx in idx_by_key.items():
        names = selected[key]
        if isinstance(idx, slice):
            selected[key] = list(names[idx])
        elif isinstance(idx, torch.Tensor):  # an int index or a bool mask
            selected[key] = [names[i] for i in torch.arange(len(names))[idx.cpu()].tolist()]
        else:
            selected[key] = [names[i] for i in idx]
    return selected


def cache_identity(dataset: DatasetBase) -> dict[str, Any]:
    """The prepare-shaping settings a cache was built with, for the ``_meta`` envelope.

    Public because a dataset that *derives* its own default ``processed_data_path`` has to key
    that path by the same settings :meth:`DatasetBase.load_cached` hard-checks against it.
    Reading both off this one function is what stops the two drifting: a field added to
    ``_cache_identity_fields`` then starts keying a new cache file in the same commit that
    starts rejecting the old one, rather than turning two valid configs into one contested
    path that each rejects for the other.

    Parameters
    ----------
    dataset : DatasetBase
        Dataset to read the values from. Only ever reads attributes that have class-level
        defaults, so an uninitialized instance is a valid argument.

    Returns
    -------
    dict[str, Any]
        Field name to :func:`~eden.utils.misc.to_jsonish` value, in MRO order.
    """
    return {name: to_jsonish(getattr(dataset, name)) for name in _cache_identity_field_names(type(dataset))}


def _load_motion_list_file(path: str) -> tuple[set[str], set[str]]:
    """Parse a YAML/JSON list-file into ``(relpath_stems, basename_stems)``.

    Accepted shapes:
      * flat list of strings: ``["CMU/45.npz", ...]``;
      * list of dicts each with a ``file:`` key (the ProtoMotions
        ``amass_*_train.yaml`` shape): ``[{file: "CMU/45.motion", fps: 120, ...}]``;
      * dict wrapping either of the above under a ``motions:`` key.

    Each entry has its extension stripped (so a list authored against ``.motion``
    files matches local ``.npz`` files). Entries that contain a directory
    component (``CMU/45/45_01_stageii.motion``) go into ``relpath_stems`` so
    they only match the on-disk file with that same trailing path — preventing
    a listed ``HumanEva/S3/Static_stageii`` from also pulling in the unrelated
    ``HumanEva/S1/Static_stageii``. Entries without a directory component
    (bare basename, e.g. ``45_01_stageii``) go into ``basename_stems`` and
    match any on-disk file with that basename.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in (".yaml", ".yml"):
        with open(path) as f:
            data = yaml.safe_load(f)
    elif ext == ".json":
        with open(path) as f:
            data = json.load(f)
    else:
        raise ValueError(f"List file {path!r} must have a .yaml/.yml/.json extension; got {ext!r}.")

    if isinstance(data, dict):
        if "motions" not in data:
            raise ValueError(f"List file {path!r} is a dict but has no 'motions' key. Top-level keys: {sorted(data)}.")
        data = data["motions"]

    if not isinstance(data, list):
        raise ValueError(
            f"List file {path!r} must contain a list (or a 'motions:' key wrapping a list), got {type(data).__name__}."
        )

    relpath_stems: set[str] = set()
    basename_stems: set[str] = set()
    for i, entry in enumerate(data):
        if isinstance(entry, str):
            name = entry
        elif isinstance(entry, dict):
            if "file" not in entry:
                raise ValueError(f"Entry {i} in {path!r} is a dict but has no 'file' key: {entry!r}.")
            name = entry["file"]
            if not isinstance(name, str):
                raise ValueError(f"Entry {i} 'file' in {path!r} must be a string, got {type(name).__name__}.")
        else:
            raise ValueError(
                f"Entry {i} in {path!r} must be a string or dict with 'file' key, got {type(entry).__name__}."
            )
        no_ext = os.path.splitext(name)[0].replace(os.sep, "/").strip("/")
        if not no_ext:
            raise ValueError(f"Entry {i} in {path!r} resolves to an empty path: {name!r}.")
        if "/" in no_ext:
            relpath_stems.add(no_ext)
        else:
            basename_stems.add(no_ext)

    if not relpath_stems and not basename_stems:
        raise ValueError(f"List file {path!r} resolved to zero motion entries.")
    return relpath_stems, basename_stems


def _match_against_list(
    file_path: str,
    relpath_stems: set[str],
    basename_stems: set[str],
) -> str | None:
    """Return the listed entry that matches ``file_path``, or ``None`` if no match.

    Matches against ``relpath_stems`` by checking if the file's extension-stripped
    path ends with one of the entries on a ``/`` boundary (so
    ``HumanEva/S1/Static_stageii`` does not match a listed
    ``HumanEva/S3/Static_stageii``). Falls back to a basename-only match against
    ``basename_stems`` for bare-basename entries.

    A slash-less ``file_path`` (a file at the root of ``raw_data_path``, or a
    single-file ``raw_data_path``) matches only a slash-less ``basename_stems``
    entry: a directory-qualified relpath entry deliberately does not match it,
    because the root file was never listed under that directory.
    """
    path_stem = os.path.splitext(file_path)[0].replace(os.sep, "/")
    base_stem = path_stem.rsplit("/", 1)[-1] if "/" in path_stem else path_stem
    if base_stem in basename_stems:
        return base_stem
    if relpath_stems:
        if path_stem in relpath_stems:
            return path_stem
        # Suffix-match on '/' boundaries (so a list of `CMU/45/45_01_stageii`
        # matches both `/abs/path/CMU/45/45_01_stageii.npz` and a relpath
        # `CMU/45/45_01_stageii.npz`).
        idx = path_stem.find("/")
        while idx != -1:
            suffix = path_stem[idx + 1 :]
            if suffix in relpath_stems:
                return suffix
            idx = path_stem.find("/", idx + 1)
    return None


def _assert_min_buffer_lengths(
    motion_sequences: dict[str, "BatchMotionSequence"],
    keys: tuple[str, ...] | list[str],
    all_motion_files: list[str] | None = None,
) -> None:
    """Fail fast on clips shorter than ``_MIN_CLIP_FRAMES`` anywhere in the loaded library.

    A 0-frame clip has no valid reference row: frame selection clamps to
    ``buffer_length - 1 = -1`` (see ``MotionTerm._select_frames``), which the
    masked ``ConcatBatchTensor`` path silently zero-fills and a raw row gather
    cannot absorb at all. A 1-frame clip spans no time, so a tracked episode
    completes on its first step. ``_pack_motion_sequence`` already rejects short
    clips as they are read; this catches the ones a modifier (``TrimFrames``
    etc.) shrinks after that, and any in a loaded cache, by name at load time
    instead of deep in the hot path.
    """
    for key in keys:
        lengths = motion_sequences[key].buffer_length
        bad = (lengths < _MIN_CLIP_FRAMES).nonzero(as_tuple=True)[0].tolist()
        if bad:
            names = [all_motion_files[i] for i in bad if i < len(all_motion_files)] if all_motion_files else bad
            raise ValueError(
                f"Motion sequence '{key}' contains {len(bad)} clip(s) shorter than {_MIN_CLIP_FRAMES} frames: "
                f"{names} with {[int(lengths[i]) for i in bad]} frame(s). Every clip needs at least "
                f"{_MIN_CLIP_FRAMES} frames, because a single frame spans no time. Check drop_first_n_frames / "
                f"drop_last_n_frames, the modifiers that trim or retime clips, and the source files for the clips "
                f"above."
            )


def _assert_consistent_buffer_lengths(
    motion_sequences: dict[str, "BatchMotionSequence"], keys: tuple[str, ...] | list[str]
) -> None:
    """Verify every key in ``motion_sequences`` has identical per-sequence ``buffer_length``.

    ``DatasetBase`` assumes all motion_sequence_keys share the same per-sequence
    lengths. Temporal modifiers (SpeedScale, TrimFrames) mutate ``buffer_length``,
    and applying them to only a subset of keys via ``motion_sequence_keys=...``
    filtering will desynchronize the dataset and surface as subtle indexing
    errors downstream. This helper raises a clear ValueError up front.
    """
    if len(keys) < 2:
        return
    ref_key = keys[0]
    ref = motion_sequences[ref_key].buffer_length
    for key in keys[1:]:
        other = motion_sequences[key].buffer_length
        if ref.shape != other.shape or not torch.equal(ref, other):
            raise ValueError(
                f"Dataset modifier stack left motion_sequences with mismatched "
                f"per-sequence lengths: '{ref_key}' has {ref.tolist()} but "
                f"'{key}' has {other.tolist()}. Apply temporal modifiers to all "
                f"keys, or avoid per-key filtering via motion_sequence_keys= for "
                f"those modifiers."
            )


class DatasetBase(ConfigurableMixin[DatasetOptions], FilterMixin):
    """Base class for motion datasets backed by a HuggingFace repo.

    Parameters
    ----------
    registry : str | None
        HuggingFace namespace (user or organisation) that hosts the dataset repo.
        Subclasses should set this to enable automatic downloads, e.g.
        `registry = "ember-lab-berkeley"`.
    dataset : str | None
        HuggingFace dataset repository name, e.g. `"AMASS_Retargeted_for_G1"`.
    raw_data_path : str | None
        Path to the raw data directory. When set, it is used as-is and must
        already exist — a missing explicit path raises rather than downloading.
        When None and the class defines ``registry`` / ``dataset``, the HF repo
        is downloaded to the HuggingFace global cache.
    processed_data_path : str | None
        Path to the processed data file used by this dataset.
    target_framerate : float, optional
        The target framerate of the motion. This must match the simulation frequency. Defaults to 50.0.
    drop_first_n_frames : int, optional
        Number of frames to drop from the start of each raw sequence during
        ``prepare()``. Applied by subclasses when slicing loaded motions, through
        :meth:`_frame_window`. Defaults to 0.
    drop_last_n_frames : int, optional
        Number of frames to drop from the end of each raw sequence during
        ``prepare()``. Defaults to 0.
    modifiers : tuple[DatasetModifierOptions, ...] | list[DatasetModifierOptions], optional
        Offline transforms applied after cache load or raw prepare; never baked into the cache.
    include_pattern, exclude_pattern : str | None, optional
        Regexes ``re.search``-ed against each motion file's basename (see :class:`DatasetOptions`).
    include_list_file, exclude_list_file : str | None, optional
        YAML/JSON lists of motion files to keep / drop (see :func:`_load_motion_list_file`).
    recalculate_velocities : bool, optional
        Re-estimate velocities by finite differences of the (retimed) positions instead of carrying the
        source velocities through. Defaults to True.
    strict_cache_identity : bool, optional
        Refuse a cache that predates one of ``_cache_identity_fields`` (its value is unknown); ``False``
        accepts it with a warning. A differing value always raises. Defaults to True.
    body_model_params : bool, optional
        Also load the parametric body-model pose and shape (MANO / SMPL family) of the sequences that have one, as
        ``body_pose_quat`` / ``body_shape`` / ``body_model``. Off by default: nothing extra is read, stored or
        cached, and no simulation-side buffer ever holds them. Only datasets that declare support take it
        (``_supports_body_model_params``); a cache prepared without it is refused rather than loaded short.
    body_model_dir : str | None, optional
        Directory of the body-model files (e.g. ``MANO_RIGHT.pkl`` / ``MANO_LEFT.pkl``), for datasets that need the
        model to read a release (expanding a PCA pose, locating the root joint). The files are licence-gated and
        never shipped.

    Class Attributes
    ----------------
    file_extension: str
        The extension of the motion files.
    motion_sequence_keys: tuple[str, ...]
        The distinct names of the actors present in the dataset.
    motion_dofs_name: dict[str, list[str]]
        Per actor (``motion_sequence_keys`` entry), the names of the stored ``dofs_pos`` columns, in order.
    motion_links_name: dict[str, list[str]]
        Per actor, the names of the stored link rows, in order.
    _hf_allow_patterns: tuple[str, ...] | None
        Glob patterns limiting what the automatic download fetches from the HF
        repo. ``None`` fetches the whole repo. Set this when the repo holds
        splits this loader never reads.
    _hf_data_subdir: str | None
        Directory *inside* the downloaded (and, if applicable, unpacked) repo
        that holds this loader's motion files, as a relative ``/``-separated
        path. ``None`` reads the repo root.

    Notes
    -----
    The automatic download runs three independent stages, each opt-in:
    ``_hf_allow_patterns`` decides what is fetched, ``_post_download_process``
    unpacks archives in place and returns the root, and ``_hf_data_subdir``
    narrows that root to the directory this loader reads. Only the download
    path goes through them — an explicit ``raw_data_path`` is used as given,
    so it must already point at the equivalent of the narrowed directory.

    The two layout attributes are private class attributes, not options: a
    repo's layout belongs to the loader that reads it, so a retarget for
    another robot or split is a subclass declaring its own. Both are checked
    when the class is defined.
    """

    registry: str | None = None
    dataset: str | None = None
    raw_data_path: str | None = None
    processed_data_path: str | None = None
    target_framerate: float = 50.0
    drop_first_n_frames: int = 0
    drop_last_n_frames: int = 0
    modifiers: tuple[DatasetModifierOptions, ...] | list[DatasetModifierOptions] = ()
    include_pattern: str | None = None
    exclude_pattern: str | None = None
    include_list_file: str | None = None
    exclude_list_file: str | None = None
    # Finite-difference velocities from the (possibly retimed) position series instead
    # of carrying the source velocities through ``interpolate_motion``. Source
    # velocities and the retimed positions are otherwise inconsistent at the
    # control dt — the policy then sees a mismatched velocity reference signal
    # and reward terms based on it become noise. See OpenTrack's
    # ``recalculate_traj_*_velocity`` calls for the same fix.
    #
    # Position-only datasets (LAFAN1, SEED, OMOMO retargets) have no source
    # velocities, so the flag changes nothing for them. For datasets that ship
    # velocities (AMASS), it replaces them with finite differences of the
    # (possibly resampled) positions — what you want whenever ``target_framerate``
    # differs from the source fps. ``False`` keeps the linearly interpolated
    # source velocities with no re-estimate.
    recalculate_velocities: bool = True

    # How to treat a cache whose recorded identity omits a field this class declares in
    # ``_cache_identity_fields`` (see :meth:`load_cached`). Strict reads the recording as
    # complete, so an omitted field is an unknown value, which is a mismatch.
    #
    # A field rather than a hardcoded rule because only the caller can weigh the cost: adding a
    # field to the tuple invalidates every cache on disk, and re-preparing an AMASS/PHUMA-scale
    # library costs hours — on a cluster where the raw data may not even be mounted, since
    # shipping only the processed cache is supported. False accepts an unknown with a warning.
    # A *differing* value always raises; no flag relaxes a detected conflict.
    strict_cache_identity: bool = True

    body_model_params: bool = False
    body_model_dir: str | None = None
    #: Whether this dataset can fill the body-model fields (``body_model_params``).
    _supports_body_model_params: ClassVar[bool] = False

    # Option names whose values change the *prepared bytes*, recorded in the cache envelope and
    # compared on load. ``target_framerate`` and ``motion_sequence_keys`` have their own checks;
    # these are the rest, and a subclass adding a prepare-time parameter must list it or its
    # cache silently outlives the setting. A subclass declares only what *it* adds — the names
    # are unioned over the MRO by :func:`_cache_identity_field_names`.
    #
    # Filters are absent on purpose: ``include_pattern`` and the list files are re-applied on
    # cache load, and ``modifiers`` are never baked into the cache at all.
    _cache_identity_fields: ClassVar[tuple[str, ...]] = (
        "drop_first_n_frames",
        "drop_last_n_frames",
        "recalculate_velocities",
    )
    # What the prepared bytes *mean*, for a dataset whose reader changed it (e.g. DexYCB's hand root). A cache
    # prepared under another meaning, or before the dataset declared one, is refused outright: unlike an untracked
    # identity field, no setting can vouch that its bytes are still right, so ``strict_cache_identity`` does not
    # relax it. Changing it invalidates only this dataset's caches, where a ``_CACHE_FORMAT_VERSION`` bump would
    # invalidate every dataset's.
    _cache_semantics: ClassVar[str | None] = None

    file_extension: ClassVar[str] = ".npz"
    motion_sequence_keys: ClassVar[tuple[str, ...] | None] = None
    # NOTE: will be used to resolve the indices of the motion sequence keys
    motion_dofs_name: ClassVar[dict[str, list[str]] | None] = None
    motion_links_name: ClassVar[dict[str, list[str]] | None] = None
    #: Keys of ``self.metadata`` holding one entry per clip, positionally aligned with the trajectories. They are
    #: reindexed together with the trajectories by :meth:`select_trajectories` (file filters on cache load, DDP
    #: subsets), so a per-clip label keeps naming its own clip after a subset.
    _PER_CLIP_METADATA_KEYS: ClassVar[tuple[str, ...]] = ()
    # The HF download's layout — see the class docstring. Underscored so that
    # ``get_parameter_names`` leaves them off the options: they belong to the class.
    _hf_allow_patterns: ClassVar[tuple[str, ...] | None] = None
    _hf_data_subdir: ClassVar[str | None] = None
    #: Whether this reader trims its clips by :meth:`_frame_window`. Opt-in: a reader sets it to ``True`` only once
    #: its read path applies the window, and every other reader refuses a non-zero ``drop_first_n_frames`` /
    #: ``drop_last_n_frames`` at construction rather than silently ignoring them — they are part of the cache
    #: identity, so an ignored value would also name a cache whose clips it never trimmed.
    _applies_frame_window: ClassVar[bool] = False
    #: The drop settings this reader cut exactly as :meth:`_frame_window` does before caches recorded
    #: ``frame_window``: a cache written without that marker is trusted when every non-zero drop is one of them.
    #: Empty by default, refusing such a cache — OMOMO and Weave ignored both drops, and AMASS, PHUMA and HHI cut
    #: ``drop_first_n_frames`` right but could wrap a ``drop_last_n_frames`` longer than the clip.
    _drops_cut_before_marker: ClassVar[tuple[str, ...]] = ()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls is DatasetBase:
            return
        if cls.motion_sequence_keys is None:
            raise TypeError(f"{cls.__name__} must define 'motion_sequence_keys' as a class attribute")
        if cls.motion_dofs_name is None:
            raise TypeError(f"{cls.__name__} must define 'motion_dofs_name' as a class attribute")
        if cls.motion_links_name is None:
            raise TypeError(f"{cls.__name__} must define 'motion_links_name' as a class attribute")
        _check_hf_narrowing(cls)

    if TYPE_CHECKING:
        motion_sequences: dict[str, BatchMotionSequence]
        all_motion_files: list[str]

    def __init__(self, options: DatasetOptions, device: torch.device):
        super().__init__(options)
        if not self._applies_frame_window and self._trims_clips:
            raise ValueError(
                f"{type(self).__name__} does not trim its clips at read time, so it cannot apply "
                f"drop_first_n_frames={self.drop_first_n_frames} / drop_last_n_frames={self.drop_last_n_frames}. "
                f"Leave both at 0 and trim after loading with the TrimFrames modifier instead: "
                f"modifiers=[TrimFrames.configure(trim_first={self.drop_first_n_frames}, "
                f"trim_last={self.drop_last_n_frames})]."
            )

        self._device = device
        self._initialize()

    def __len__(self) -> int:
        return len(self.motion_sequences[self.motion_sequence_keys[0]].buffer_length)

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def buffer_length(self) -> torch.Tensor:
        # NOTE: all the motion sequences should have the same buffer length
        return self.motion_sequences[self.motion_sequence_keys[0]].buffer_length

    @property
    def framerate(self) -> torch.Tensor:
        # NOTE: all the motion sequences should have the same framerate
        return self.motion_sequences[self.motion_sequence_keys[0]].framerate

    def _initialize(self):
        """Bring the dataset to a usable state on this rank.

        DDP coordination is a custom rank-0-prepare protocol rather than letting every rank
        fetch. ``eden.utils.assets._snapshot_download_locked`` — which ``_download_from_hf`` goes
        through — is safe to call concurrently, but here only rank 0 needs the raw data at all;
        non-main ranks consume the processed cache file rank 0 produces — so every other rank
        would be paying a network/cache lookup for files it never reads, on top of a raw library
        that can be hundreds of GB.
        Instead, this method gates the entire download → prepare → save chain to rank 0 and
        uses one barrier to let the other ranks proceed with a shared mmap load. A dataset
        that overrides ``_download_from_hf`` inherits that gate for free.
        """
        if self.body_model_params and not self._supports_body_model_params:
            raise ValueError(
                f"{type(self).__name__} does not provide body-model parameters (body_model_params=True); datasets "
                f"that do: {', '.join(_body_model_datasets()) or 'none registered'}. Leave body_model_params off "
                f"for this dataset."
            )
        initialized = False
        cache_path = self.processed_data_path

        is_distributed = eden_dist.is_distributed()
        is_main = eden_dist.is_main_process()

        # 1. Fast path: load pre-processed cache when present.
        if cache_path is not None and os.path.exists(cache_path):
            en.logger.info(f"Loading processed data from {cache_path}")
            self.motion_sequences, self.metadata = self.load_cached(data_path=cache_path)
            self._reconcile_cached_body_model(cache_path)
            # Move loaded data to the target device. Cache tensors are mmap'd
            # CPU tensors; .to(device) copies to GPU once per field (cheap
            # compared to gathering per-frame at training time).
            self.motion_sequences = {k: v.to(self.device) for k, v in self.motion_sequences.items()}
            self.all_motion_files = self.metadata.get("all_motion_files", [])
            # Filter trajectories in the loaded cache against the configured
            # include/exclude/list-file filters. Useful for re-using a full-
            # library cache to train on a subset without rebuilding the cache.
            # The relpaths stored in the metadata make directory-qualified list
            # entries survive the round-trip.
            # select_trajectories reindexes all_motion_files and every key in
            # _PER_CLIP_METADATA_KEYS; a subclass that stashes positionally-aligned
            # per-clip metadata must declare it there to stay aligned.
            _, keep_idx = self._apply_file_filter(self.all_motion_files)
            if keep_idx is not None:
                self.select_trajectories(keep_idx)
            initialized = True

        # 2. Cache miss. On non-main DDP ranks, wait for rank 0 to prepare the
        # cache, then load it. This avoids N copies of the same prepare() pass.
        if not initialized and is_distributed and not is_main and cache_path is not None:
            en.logger.info(f"[rank {eden_dist.get_global_rank()}] waiting for rank 0 to prepare cache at {cache_path}")
            eden_dist.barrier()  # rank 0 finishes prepare()+save
            if not os.path.exists(cache_path):
                raise RuntimeError(f"Rank 0 did not produce the cache at {cache_path}. Check rank 0 logs.")
            en.logger.info(f"[rank {eden_dist.get_global_rank()}] loading prepared cache from {cache_path}")
            self.motion_sequences, self.metadata = self.load_cached(data_path=cache_path)
            self._reconcile_cached_body_model(cache_path)
            self.motion_sequences = {k: v.to(self.device) for k, v in self.motion_sequences.items()}
            self.all_motion_files = self.metadata.get("all_motion_files", [])
            _, keep_idx = self._apply_file_filter(self.all_motion_files)
            if keep_idx is not None:
                self.select_trajectories(keep_idx)
            initialized = True

        # 3. Rank 0 (or single-process) downloads + prepares + saves. Wrapping
        # the whole thing in try/finally guarantees the rank-0 release barrier
        # fires even if download or prepare raises — without that, the ranks
        # waiting in step 2 hang forever and the rank-0 failure looks like a
        # cluster-wide stall. The barrier is only fired when non-main ranks
        # actually wait (i.e. cache_path is not None); otherwise rank 0 would
        # be calling barrier() with no peers and hang alone.
        if not initialized:
            needs_release_barrier = is_distributed and is_main and cache_path is not None
            try:
                # 3a. Resolve the raw data location. Download only when no path
                # was given; an explicit path is used as-is and must exist, so a
                # typo doesn't silently pull several GB into the wrong place.
                if self.raw_data_path is None:
                    if not self._can_auto_download():
                        raise ValueError(
                            "No data source found. Provide `raw_data_path` / "
                            "`processed_data_path`, or set `registry` and `dataset` "
                            "on the dataset class to enable automatic HuggingFace "
                            "downloads."
                        )
                    self.raw_data_path = self._download_from_hf()
                elif not os.path.exists(self.raw_data_path):
                    raise FileNotFoundError(self._missing_raw_data_message(self.raw_data_path))

                # Backstop: a download that returned a bad path.
                if not os.path.exists(self.raw_data_path):
                    raise FileNotFoundError(f"Raw data directory {self.raw_data_path} not found")

                # 3b. Prepare from raw data (pre-existing or just downloaded).
                en.logger.info(f"Loading raw data from {self.raw_data_path}")
                self.all_motion_files = self._get_all_motion_files(self.raw_data_path, self.file_extension)
                # Surface the empty-glob case here so it doesn't get masked by
                # ``_apply_file_filter``'s "matched no files out of 0" error,
                # which conflates "filter rejected everything" with "the path
                # has no files to begin with"
                if not self.all_motion_files:
                    raise FileNotFoundError(
                        f"Found 0 {self.file_extension!r} files under {self.raw_data_path!r}. "
                        f"The directory exists but contains no matching files — check the "
                        f"path's mount on this node (e.g. SLURM workers sometimes only "
                        f"resolve user-home symlinks), permissions on the subdirectories, "
                        f"and ``file_extension`` on the dataset class."
                    )
                # Filter file list before prepare() so excluded clips are never
                # read/interpolated/velocity-estimated and never enter the cache.
                self.all_motion_files, _ = self._apply_file_filter(self.all_motion_files)
                self.motion_sequences, self.metadata = self.prepare()
                # Store paths relative to raw_data_path (not basenames): list-file
                # filtering re-runs against this metadata on cache load, and
                # directory-qualified list entries (``CMU/45/45_01_stageii``) can
                # only match when the directory components survive the round-trip.
                rel_root = (
                    self.raw_data_path if os.path.isdir(self.raw_data_path) else os.path.dirname(self.raw_data_path)
                )
                self.metadata["all_motion_files"] = [
                    os.path.relpath(f, rel_root).replace(os.sep, "/") for f in self.all_motion_files
                ]
                # NOTE: ensure all the motion lengths are the same
                sample_length = self.motion_sequences[self.motion_sequence_keys[0]].buffer_length
                for key in self.motion_sequence_keys:
                    assert (self.motion_sequences[key].buffer_length == sample_length).all(), (
                        f"Motion sequence {key} has different length than the sample length {sample_length}"
                    )

                # 3c. Save cache so subsequent runs (and other ranks waiting in
                # step 2) can load it.
                if cache_path is not None:
                    self.save_cached(cache_path)
                initialized = True
            finally:
                if needs_release_barrier:
                    eden_dist.barrier()

        # Modifiers run after cache save so the cache stays agnostic to the
        # modifier stack — changing the config re-applies them on next load.
        self._apply_modifiers()
        # Runs after BOTH the cache-load and fresh-prepare branches, and after
        # modifiers (TrimFrames etc. can only shrink clips further).
        _assert_min_buffer_lengths(
            self.motion_sequences, self.motion_sequence_keys, self.metadata.get("all_motion_files")
        )

    def _reconcile_cached_body_model(self, cache_path: str) -> None:
        """Match a cache's body-model fields to ``body_model_params``: refuse a cache prepared without them when the
        flag asks for them, and drop them from one prepared with them when it does not (so the loaded data is what
        the flag says, for the modifiers that refuse those fields too).
        """
        if not self.body_model_params:
            for seq in self.motion_sequences.values():
                seq.body_pose_quat = seq.body_shape = seq.body_model = None
            return
        if not any(seq.body_pose_quat is not None for seq in self.motion_sequences.values()):
            raise ValueError(
                f"{type(self).__name__}: the cache at {cache_path} was prepared without body-model parameters, but "
                f"body_model_params=True. Delete it to rebuild, or point processed_data_path at another file."
            )

    def _apply_modifiers(self) -> None:
        if not self.modifiers:
            return
        # Modifier ops are not idempotent: Translate uses add_() on the concat buffer,
        # Jitter adds fresh RNG noise, SpeedScale / TrimFrames re-resample, etc.
        # Calling this twice would silently compound. Reload the dataset from its
        # cache or raw source if you need to re-run the modifier stack.
        if getattr(self, "_modifiers_applied", False):
            raise RuntimeError(
                f"{type(self).__name__}._apply_modifiers() was called twice. "
                "Modifier operations mutate motion_sequences in place and are not "
                "idempotent (Translate would double-offset, Jitter would re-add "
                "noise, SpeedScale would re-resample). Reload the dataset from "
                "processed_data_path or raw_data_path to re-run the modifier stack."
            )
        from eden.datasets.modifiers.base import build_modifier

        for mod_options in self.modifiers:
            self.motion_sequences = build_modifier(mod_options).apply(self.motion_sequences)

        _assert_consistent_buffer_lengths(self.motion_sequences, self.motion_sequence_keys)
        self._modifiers_applied = True

    def _download_from_hf(self) -> str:
        """Download the dataset repository from HuggingFace Hub.

        Fetches :attr:`_hf_allow_patterns` (or the whole repo when it is None),
        hands the snapshot to :meth:`_post_download_process` to unpack any
        archives, then narrows the result to :attr:`_hf_data_subdir`.

        Only reached with ``raw_data_path`` unset (see :meth:`_initialize`), so the
        snapshot lands in the HuggingFace global cache — under a revision-keyed
        directory, which is also what :meth:`_post_download_process` unpacks into.

        Returns
        -------
        str
            Local path to the directory holding this dataset's motion files.
        """
        from eden.utils.assets import _snapshot_download_locked

        repo_id = f"{self.registry}/{self.dataset}"
        scope = "" if self._hf_allow_patterns is None else f" (files matching {list(self._hf_allow_patterns)})"
        en.logger.info(f"Downloading dataset '{repo_id}'{scope} from HuggingFace Hub...")
        local_path = _snapshot_download_locked(
            repo_type="dataset",
            repo_id=repo_id,
            local_dir=self.raw_data_path,
            allow_patterns=None if self._hf_allow_patterns is None else list(self._hf_allow_patterns),
        )
        en.logger.info(f"Dataset downloaded to {local_path}")
        return self._resolve_data_subdir(self._post_download_process(local_path))

    def _post_download_process(self, local_dir: str) -> str:
        """Unpack archives in the downloaded snapshot, in place.

        Override to extract a ``.zip`` / ``.tar.gz`` the repo ships. Return the
        root the extraction produced (usually ``local_dir`` itself) — narrowing
        to a subdirectory is :attr:`_hf_data_subdir`'s job, not this hook's, so
        that the two compose instead of each re-implementing the other.
        """
        return local_dir

    def _extract_once(self, local_dir: str, archive_name: str, extracted_root: str) -> None:
        """Extract ``local_dir/archive_name`` (a ``.zip`` or a tarball) into ``local_dir``, unless already finished.

        A finished extraction is marked by an empty ``_EXTRACTED_MARKER`` file written into
        ``local_dir/extracted_root`` (the archive's top-level directory) once the extraction has returned. No
        directory can stand in for it: an archive's directory entries precede their files, so the tree exists
        moments into an extraction, and one interrupted after that would be skipped on every later run — loading a
        silently partial library. The marker lives inside ``extracted_root`` so deleting that directory also
        re-arms the extraction. A tree extracted before the marker existed is re-extracted once, or, where it cannot
        be, refused by name — see :meth:`_check_extractable`.
        """
        archive = os.path.join(local_dir, archive_name)
        marker = os.path.join(local_dir, extracted_root, _EXTRACTED_MARKER)
        if os.path.exists(marker):
            return
        self._check_extractable(local_dir, archive_name, extracted_root)
        en.logger.info(f"Extracting {archive} to {local_dir}...")
        if archive_name.endswith(".zip"):
            import zipfile

            with zipfile.ZipFile(archive, "r") as zip_ref:
                zip_ref.extractall(local_dir)
        else:
            import tarfile

            with tarfile.open(archive, "r:*") as tar:
                tar.extractall(local_dir)
        # An archive without ``extracted_root`` is a layout change; leave it to ``_resolve_data_subdir`` to say so
        # rather than fail here writing the marker.
        if os.path.isdir(os.path.dirname(marker)):
            with open(marker, "w"):
                pass

    def _check_extractable(self, local_dir: str, archive_name: str, extracted_root: str) -> None:
        """Raise when ``archive_name`` cannot be (re-)extracted into ``local_dir``, naming the path that needs none.

        For a read-only copy — a shared mirror, a read-only HF cache — holding an ``extracted_root`` extracted by
        hand or before the marker existed: without the marker that tree cannot be told from a partial one, and
        without write access it cannot be re-extracted, so the extraction would fail with a bare ``PermissionError``
        on every run.

        Raises
        ------
        PermissionError
            When ``local_dir``, or the ``extracted_root`` already in it, is not writable.
        """
        extracted = os.path.join(local_dir, extracted_root)
        has_tree = os.path.isdir(extracted)
        targets = [local_dir, extracted] if has_tree else [local_dir]
        read_only = [target for target in targets if not os.access(target, os.W_OK)]
        if not read_only:
            return
        name = type(self).__name__
        data_subdir = os.path.join(*(self._hf_data_subdir or extracted_root).split("/"))
        if has_tree:
            raise PermissionError(
                f"{name} cannot confirm that {extracted!r} is a complete extraction of {archive_name} (it has no "
                f"{_EXTRACTED_MARKER!r} marker), and cannot re-extract it because {read_only[0]!r} is not writable. "
                f"If that tree is complete, set raw_data_path={os.path.join(local_dir, data_subdir)!r}: a path to the "
                f"extracted tree is used as is, with no extraction. Creating the marker by hand makes the same claim, "
                f"and making it about a partial tree loads a silently short motion library."
            )
        raise PermissionError(
            f"{name} cannot extract {archive_name} into {local_dir!r}: it is not writable. Extract the archive into a "
            f"writable directory and set raw_data_path to the {data_subdir} directory it produces."
        )

    def _resolve_data_subdir(self, root: str) -> str:
        """Narrow ``root`` to :attr:`_hf_data_subdir`, or return it unchanged.

        Raises
        ------
        FileNotFoundError
            When the subdirectory is absent — an interrupted download or
            extraction, or a layout that is not the one this loader reads.
            Failing here names the cause; letting the glob run on ``root``
            instead either mixes in splits this loader cannot parse or silently
            finds nothing.

        Notes
        -----
        Reached from the download path and from a dataset resolving an explicit
        ``raw_data_path`` (see ``SEEDRetargetedG1Dataset._get_all_motion_files``),
        so the message names only causes that hold for both. When the top of the
        subdirectory exists but not the rest, the tree is partial, and the message
        names that top directory to delete — never ``root``, which holds the
        downloaded archive (and, on the explicit path, is the caller's own data).
        When not even the top exists, the layout is not the one this loader reads;
        the allow-patterns clause is added only when the class declares them.
        """
        if self._hf_data_subdir is None:
            return root
        segments = self._hf_data_subdir.split("/")
        resolved = os.path.join(root, *segments)
        if os.path.isdir(resolved):
            return resolved

        def contents(path: str) -> str:
            entries = sorted(os.listdir(path)) if os.path.isdir(path) else []
            more = "" if len(entries) <= 20 else f", ... (+{len(entries) - 20} more)"
            return (", ".join(entries[:20]) or "(empty)") + more

        top = os.path.join(root, segments[0])
        expects = f"{type(self).__name__} expects {self._hf_data_subdir!r} under {root!r}"
        if os.path.isdir(top):
            raise FileNotFoundError(
                f"{expects}, but {top!r} stops short of it (contents: {contents(top)}). That is a partial "
                f"extraction or download. Delete {top!r} and rerun to rebuild it — only that directory: "
                f"{root!r} also holds what it is rebuilt from."
            )
        source = (
            f"the '{self.registry}/{self.dataset}' repo"
            if self.registry is not None and self.dataset is not None
            else "the source"
        )
        patterns = "" if self._hf_allow_patterns is None else ", or `_hf_allow_patterns` does not fetch it"
        raise FileNotFoundError(
            f"{expects}, and there is no {segments[0]!r} there (contents of {root!r}: {contents(root)}). The "
            f"layout of {source} is not the one this loader reads{patterns}. Setting _hf_data_subdir to None is "
            f"not the fix: it globs that whole directory, mixing in files this loader cannot parse."
        )

    def _can_auto_download(self) -> bool:
        """Whether ``raw_data_path=None`` can resolve this dataset's raw data.

        True for a dataset that owns a HuggingFace repo, and for one that resolves its own
        source by overriding :meth:`_download_from_hf` — a task-owned dataset fetching the one
        sample clip its task defaults to, which owns no corpus and so declares no repo.

        Read by :meth:`_initialize` before it takes the download branch, and by
        :meth:`_missing_raw_data_message` before it suggests that branch as a remedy.
        """
        if type(self)._download_from_hf is not DatasetBase._download_from_hf:
            return True
        return self.registry is not None and self.dataset is not None

    def _missing_raw_data_message(self, path: str) -> str:
        """Error text for a missing explicit ``raw_data_path``, listing sibling
        directories so a typo (``s_0000`` vs ``subset_0000``) is obvious.
        """
        parent = os.path.dirname(os.path.abspath(path.rstrip("/"))) or "/"
        hint = ""
        if os.path.isdir(parent):
            try:
                siblings = sorted(e for e in os.listdir(parent) if os.path.isdir(os.path.join(parent, e)))
            except OSError:
                siblings = []
            if siblings:
                preview = ", ".join(siblings[:20])
                more = "" if len(siblings) <= 20 else f", ... (+{len(siblings) - 20} more)"
                hint = f" Existing directories in {parent!r}: {preview}{more}."
        # Only offered when it would actually work: a dataset that resolves no source of its
        # own answers `raw_data_path=None` with "No data source found", so suggesting it there
        # would send the caller from one error to a different one.
        remedy = (
            " Pass raw_data_path=None to download the dataset instead."
            if self._can_auto_download()
            else f" {type(self).__name__} has no source of its own to fall back to — it reads "
            f"whatever file or directory it is given."
        )
        return (
            f"raw_data_path {path!r} does not exist. An explicitly provided "
            f"raw_data_path must point at data already on disk — it is not "
            f"auto-downloaded.{remedy}{hint}"
        )

    @abstractmethod
    def prepare(self) -> tuple[dict[str, BatchMotionSequence], dict[str, Any]]:
        pass

    def _get_all_motion_files(self, data_path: str, extension: str = ".npz") -> list[str]:
        if os.path.isdir(data_path):
            return glob.glob(os.path.join(data_path, f"**/*{extension}"), recursive=True)
        else:
            assert os.path.exists(data_path), f"Data path {data_path} not found"
            assert data_path.endswith(extension), f"Data path {data_path} does not end with {extension}"
            return [data_path]

    def _apply_file_filter(self, files: list[str]) -> tuple[list[str], list[int] | None]:
        """Filter ``files`` by include/exclude pattern and list-file rules.

        Returns ``(kept_files, kept_indices)``. ``kept_indices`` is ``None`` when
        no filter is active — callers can skip any index-based follow-up
        (e.g. ``select_trajectories``) in that case. Otherwise it's the list of
        positions into the original ``files`` list, so cache-load callers can
        slice ``motion_sequences`` accordingly.

        Regex filters match on basename (with extension). List-file filters match
        on the extension-stripped path stem (so YAML/JSON lists authored against
        the original ``.motion`` filenames work against local ``.npz`` files):
        directory-qualified entries match by exact-or-``/``-boundary-suffix on the
        relpath, and bare-basename entries match any file with that basename.

        Empty results raise so a misspelled pattern or wrong list fails loudly
        instead of producing a zero-length dataset that crashes later as a
        confusing shape error.
        """
        inc_list = _load_motion_list_file(self.include_list_file) if self.include_list_file is not None else None
        exc_list = _load_motion_list_file(self.exclude_list_file) if self.exclude_list_file is not None else None
        if self.include_pattern is None and self.exclude_pattern is None and inc_list is None and exc_list is None:
            return files, None
        inc = re.compile(self.include_pattern) if self.include_pattern is not None else None
        exc = re.compile(self.exclude_pattern) if self.exclude_pattern is not None else None
        kept_files: list[str] = []
        kept_indices: list[int] = []
        matched_list_entries: set[str] = set()
        for i, f in enumerate(files):
            base = os.path.basename(f)
            if inc is not None and not inc.search(base):
                continue
            if exc is not None and exc.search(base):
                continue
            if inc_list is not None:
                match = _match_against_list(f, *inc_list)
                if match is None:
                    continue
                matched_list_entries.add(match)
            if exc_list is not None and _match_against_list(f, *exc_list) is not None:
                continue
            kept_files.append(f)
            kept_indices.append(i)
        if not kept_files:
            raise ValueError(
                f"file filter (include_pattern={self.include_pattern!r}, "
                f"exclude_pattern={self.exclude_pattern!r}, "
                f"include_list_file={self.include_list_file!r}, "
                f"exclude_list_file={self.exclude_list_file!r}) matched no files out "
                f"of {len(files)}. Check the patterns / list file."
            )
        if inc_list is not None:
            total = len(inc_list[0]) + len(inc_list[1])
            missing = total - len(matched_list_entries)
            if missing > 0:
                en.logger.warning(
                    f"include_list_file {self.include_list_file!r} listed "
                    f"{total} motions but {missing} had no matching file on "
                    f"disk (paths not found under raw_data_path). The dataset "
                    f"will load the {len(kept_files)} that were found."
                )
        en.logger.info(
            f"File filter kept {len(kept_files)}/{len(files)} motion files "
            f"(include_pattern={self.include_pattern!r}, exclude_pattern={self.exclude_pattern!r}, "
            f"include_list_file={self.include_list_file!r}, exclude_list_file={self.exclude_list_file!r})"
        )
        return kept_files, kept_indices

    def get_motion_sequence(self, key: str) -> BatchMotionSequence:
        return self.motion_sequences[key]

    def select_trajectories(self, trajectories_idx: torch.Tensor | list[int] | slice) -> None:
        """Select a subset of trajectories from the loaded dataset, discarding the rest.

        Parameters
        ----------
        trajectories_idx : torch.Tensor | list[int] | slice
            The trajectory indices to keep.
        """
        for key in self.motion_sequence_keys:
            self.motion_sequences[key] = self.motion_sequences[key].select(trajectories_idx)

        if isinstance(trajectories_idx, torch.Tensor):
            trajectories_idx = trajectories_idx.tolist()
        elif isinstance(trajectories_idx, slice):
            trajectories_idx = list(range(len(self.all_motion_files)))[trajectories_idx]

        self.metadata["all_motion_files"] = [self.metadata["all_motion_files"][i] for i in trajectories_idx]
        self.all_motion_files = [self.all_motion_files[i] for i in trajectories_idx]
        for key in self._PER_CLIP_METADATA_KEYS:
            value = self.metadata.get(key)
            if value is None:
                continue
            self.metadata[key] = (
                value[trajectories_idx] if isinstance(value, np.ndarray) else [value[i] for i in trajectories_idx]
            )

    def motion_labels(self, key: str) -> list[str]:
        """The per-clip label list ``self.metadata[key]``, one entry per loaded trajectory.

        Parameters
        ----------
        key : str
            A per-clip metadata key the dataset declares in ``_PER_CLIP_METADATA_KEYS`` (e.g. ``"objects"``).
        """
        if key not in self._PER_CLIP_METADATA_KEYS or key not in self.metadata:
            raise KeyError(
                f"{type(self).__name__} has no per-clip metadata {key!r}; its per-clip keys are "
                f"{list(self._PER_CLIP_METADATA_KEYS)} (metadata present: {sorted(self.metadata)})"
            )
        labels = [str(v) for v in self.metadata[key]]
        if len(labels) != len(self):
            raise ValueError(
                f"{type(self).__name__}.metadata[{key!r}] has {len(labels)} entries but the dataset holds {len(self)} "
                f"trajectories; per-clip metadata must stay aligned with the trajectories."
            )
        return labels

    def select_dofs(self, dofs_idx_dict: dict[str, slice | list[int] | torch.Tensor]) -> None:
        """Select a subset of dofs from the loaded dataset, discarding the rest.

        ``motion_dofs_name`` is selected with the same indices, so a name keeps labelling its own
        column after a subset or a reorder.

        Parameters
        ----------
        dofs_idx_dict : dict[str, slice | list[int] | torch.Tensor]
            The dofs indices to keep, in the order to keep them.
        """
        for key, dofs_idx in dofs_idx_dict.items():
            self.motion_sequences[key] = self.motion_sequences[key].select_dofs(dofs_idx)
        self.motion_dofs_name = _select_names(self.motion_dofs_name, dofs_idx_dict)

    def select_links(self, links_idx_dict: dict[str, slice | list[int] | torch.Tensor]) -> None:
        """Select a subset of links from the loaded dataset, discarding the rest.

        ``motion_links_name`` is selected with the same indices, so a name keeps labelling its own
        row after a subset or a reorder.

        Parameters
        ----------
        links_idx_dict : dict[str, slice | list[int] | torch.Tensor]
            The links indices to keep, in the order to keep them.
        """
        for key, links_idx in links_idx_dict.items():
            self.motion_sequences[key] = self.motion_sequences[key].select_links(links_idx)
        self.motion_links_name = _select_names(self.motion_links_name, links_idx_dict)

    def compute_dofs_pos_range(
        self,
        entity_name: str,
        *,
        margin: float = 0.15,
        margin_overrides: dict[str, float] | None = None,
    ) -> dict[str, tuple[float, float]]:
        """Per-DOF ``(min - m, max + m)`` over every loaded demo frame.

        Mirrors dexmachina's ``demo_data.get_joint_init_limits`` (the ``limits`` half).
        Output keys come from ``self.motion_dofs_name[entity_name]``; values are
        plain Python floats so the dict can be passed to
        ``EntityOptions.soft_dofs_pos_limits`` without binding to a device at
        config time.

        Configuration-time only: forces a GPU→CPU sync via ``.cpu().tolist()``,
        so callers should resolve the dict once during config build (not per
        step).

        Parameters
        ----------
        entity_name:
            Key into ``self.motion_sequences``.
        margin:
            Default pad applied to both ends. Dexmachina's default is 0.15.
        margin_overrides:
            Substring → margin. If a DOF name *contains* a key, its value
            replaces ``margin`` for that DOF. When more than one substring
            matches a single DOF the last one wins, and a config-time warning
            is logged. Pass
            ``{"tx": 0.2, "ty": 0.2, "tz": 0.2, "roll": 0.5, "pitch": 0.5, "yaw": 0.5}``
            to mirror dexmachina exactly.

        Raises
        ------
        KeyError if ``entity_name`` is not a registered motion sequence key.
        ValueError if the entity has no ``dofs_pos`` data, zero DOFs, or
        ``motion_dofs_name`` desynchronized from ``dofs_pos`` (a name list
        that does not name every column).
        """
        bms, names = self._validate_dofs_motion(entity_name)
        flat = bms.dofs_pos.concatenated_tensor  # (total_frames, num_dofs)
        lo = flat.amin(dim=0).cpu().tolist()
        hi = flat.amax(dim=0).cpu().tolist()
        margins = self._resolve_dof_margins(names, margin, margin_overrides)
        return {n: (lo_i - m, hi_i + m) for n, lo_i, hi_i, m in zip(names, lo, hi, margins, strict=True)}

    def compute_dofs_pos_at(self, entity_name: str, *, frame: int = 0, sequence_idx: int = 0) -> dict[str, float]:
        """Per-DOF qpos at ``frame`` of demo sequence ``sequence_idx``.

        Companion to :meth:`compute_dofs_pos_range` for the per-clip
        ``default_dofs_pos`` override (dexmachina ``set_custom_init_qpos``).
        Defaults to the first frame of the first loaded sequence — the
        dexmachina convention is one clip per task, so this is the common
        case. When the dataset holds multiple clips, pass ``sequence_idx`` to
        pick the one this task config targets; ``TaskMod`` configs select a clip
        explicitly.

        Configuration-time only: forces a GPU→CPU sync via ``.cpu().tolist()``,
        so callers should resolve the dict once during config build (not per
        step).

        Returns ``{dof_name: q}`` as plain Python floats.

        Raises
        ------
        KeyError if ``entity_name`` is not a registered motion sequence key.
        ValueError on any of: no ``dofs_pos`` data, ``motion_dofs_name``
        desynchronized from ``dofs_pos`` (a name list that does not name every column),
        or ``frame`` / ``sequence_idx`` negative or out of range. Dataset-level
        desync is checked **before** ``frame`` / ``sequence_idx`` bounds so the
        deeper config error surfaces first.
        """
        if frame < 0:
            raise ValueError(f"frame must be >= 0, got {frame}")
        if sequence_idx < 0:
            raise ValueError(f"sequence_idx must be >= 0, got {sequence_idx}")
        bms, names = self._validate_dofs_motion(entity_name)
        num_seqs = int(bms.buffer_length.numel())
        if sequence_idx >= num_seqs:
            raise ValueError(
                f"sequence_idx {sequence_idx} out of range; entity '{entity_name}' has {num_seqs} sequence(s)."
            )
        seq_len = int(bms.buffer_length[sequence_idx].item())
        if frame >= seq_len:
            raise ValueError(
                f"frame {frame} is out of range for sequence {sequence_idx} of "
                f"entity '{entity_name}' (buffer_length={seq_len})."
            )
        # (num_dofs,) at the chosen (sequence, frame); mask is scalar True after the bounds check.
        qpos_t, _ = bms.dofs_pos.get_with_mask((sequence_idx, frame))
        qpos = qpos_t.cpu().tolist()
        return dict(zip(names, qpos, strict=True))

    def _validate_dofs_motion(self, entity_name: str) -> tuple[BatchMotionSequence, list[str]]:
        """Validate and return ``(bms, names)`` for a dofs-bearing entity.

        Three guards run in order; the first failure raises:
            (1) entity is a registered motion-sequence key — ``KeyError``;
            (2) ``dofs_pos`` is populated and non-empty — ``ValueError``;
            (3) ``motion_dofs_name`` length matches ``dofs_pos.shape[-1]`` —
                ``ValueError``.

        Guard (3) catches a name list that does not name every qpos column
        (``select_dofs`` keeps the two in step, so the usual cause is a
        ``motion_dofs_name`` option written for another data layout); callers
        would otherwise see an ``IndexError`` or a silently mis-keyed dict. It
        cannot see a same-length list in the wrong order. Public helpers run this
        validation before any ``frame`` / ``sequence_idx`` bounds checks so
        the deeper dataset-level error surfaces first.
        """
        if entity_name not in self.motion_sequences:
            raise KeyError(f"unknown entity '{entity_name}'. Known: {list(self.motion_sequences)}")
        bms = self.motion_sequences[entity_name]
        if bms.dofs_pos is None or bms.dofs_pos.concatenated_tensor.shape[-1] == 0:
            raise ValueError(f"entity '{entity_name}' has no dofs_pos data")
        names = self.motion_dofs_name[entity_name]
        num_dofs = bms.dofs_pos.concatenated_tensor.shape[-1]
        if len(names) != num_dofs:
            raise ValueError(
                f"motion_dofs_name['{entity_name}'] has {len(names)} entries but the "
                f"corresponding dofs_pos tensor has {num_dofs}. The dataset's DOF "
                f"name list and qpos buffer have desynchronized. Pass a motion_dofs_name "
                f"that names every column of the stored dofs_pos, in its order; to keep "
                f"only some DOFs, call select_dofs(), which narrows the names with the data."
            )
        return bms, names

    @staticmethod
    def _resolve_dof_margins(dof_names: list[str], default: float, overrides: dict[str, float] | None) -> list[float]:
        if not overrides:
            return [default] * len(dof_names)
        out: list[float] = []
        for n in dof_names:
            matches = [(sub, m) for sub, m in overrides.items() if sub in n]
            if len(matches) > 1:
                en.logger.warning(
                    f"compute_dofs_pos_range: DOF '{n}' matched margin overrides "
                    f"{[s for s, _ in matches]}; last wins ({matches[-1][0]}={matches[-1][1]})."
                )
            out.append(matches[-1][1] if matches else default)
        return out

    def save_cached(self, data_path: str | None = None):
        """Save the processed dataset to ``data_path`` in the current cache format (``_CACHE_FORMAT_VERSION``).

        The on-disk layout is a single ``torch.save`` call over a plain dict
        with a ``_meta`` envelope (matching the project's config-serialisation
        convention in ``eden/utils/configs.py``)::

            {
                "_meta": {
                    "_format_version": 4,
                    "eden": {"version": ..., "commit": ...}, "genesis": {...},
                    "target_framerate": ..., "motion_sequence_keys": [...],
                    "motion_dofs_name": ..., "motion_links_name": ...,
                    "cache_identity": {...}, "frame_window": ..., "cache_semantics": ...,
                    "all_motion_files": [...], ...,  # plus the dataset's own metadata keys
                },
                "sequences": {key: state_dict, ...},
            }

        Each per-key ``state_dict`` is produced by
        ``BatchMotionSequence.to_state_dict()`` and contains only tensors and
        primitive Python types — no custom objects — so the file is
        ``torch.load(mmap=True)``-loadable. Multiple DDP ranks on the same
        node share the OS page cache instead of each materialising a private
        copy of the motion library.
        """
        data_path = data_path or self.processed_data_path
        if data_path is None:
            raise ValueError("No data path provided")

        data_dir = os.path.dirname(data_path)
        if data_dir:
            os.makedirs(data_dir, exist_ok=True)

        # Spread self.metadata first so subclass-supplied keys (e.g. Arctic's
        # "objects", OMOMO's "human_height") survive, then layer the canonical
        # envelope keys on top so a load→modify→save round-trip can't carry
        # over stale version markers.
        meta = {
            **self.metadata,
            "_format_version": _CACHE_FORMAT_VERSION,
            "eden": get_eden_provenance(),
            "genesis": get_package_provenance("genesis-world"),
            "target_framerate": self.target_framerate,
            "motion_sequence_keys": list(self.motion_sequence_keys),
            "motion_dofs_name": self.motion_dofs_name,
            "motion_links_name": self.motion_links_name,
            "cache_identity": cache_identity(self),
            "frame_window": _FRAME_WINDOW_SEMANTICS,
            "cache_semantics": type(self)._cache_semantics,
        }
        sequences_sd = {
            # Move tensors to CPU before saving so the file is portable across
            # different GPU layouts (DDP rank 0 may have written from cuda:0
            # but a future job loading the cache may run on a different GPU).
            key: {k: (v.cpu() if isinstance(v, torch.Tensor) else v) for k, v in seq.to_state_dict().items()}
            for key, seq in self.motion_sequences.items()
        }

        payload = {"_meta": meta, "sequences": sequences_sd}
        en.logger.info(f"Saving processed data to {data_path}")
        # Write-then-rename, never straight to data_path. ``load_cached`` mmaps the cache, so
        # ``sequences_sd`` can be backed by the very file being written: saving in place
        # truncates it out from under its own payload and leaves 0 bytes, which every later run
        # then fails to open with torch's "mmap can only be used with files saved with
        # _use_new_zipfile_serialization" — a message about the wrong thing entirely. Reachable
        # by just re-running ``eden preprocess`` once a cache exists. The rename is atomic
        # within the directory, so a crash or a second writer mid-save also cannot leave a
        # partial cache behind.
        tmp_path = f"{data_path}.{os.getpid()}.tmp"
        try:
            torch.save(payload, tmp_path)
            os.replace(tmp_path, data_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def load_cached(self, data_path: str) -> tuple[dict[str, BatchMotionSequence], dict[str, Any]]:
        """Load the processed dataset from a cache file in the current format (``_CACHE_FORMAT_VERSION``).

        Uses ``torch.load(mmap=True)`` so tensors are page-cached at the OS
        level instead of materialised into per-process buffers — repeated
        loads (e.g. across DDP ranks on the same node) avoid re-reading the
        file from disk and avoid duplicating it in RAM.
        """
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Processed data file {data_path} not found")

        # weights_only=False is required because the _meta envelope carries
        # str/list/dict primitives alongside tensors. Caches are produced by
        # trusted Eden code on the machine that consumes them; weights_only=True
        # would require moving _meta to a sidecar.
        payload = torch.load(data_path, mmap=True, weights_only=False)

        meta = payload.get("_meta") if isinstance(payload, dict) else None
        if meta is None or meta.get("_format_version") != _CACHE_FORMAT_VERSION:
            raise RuntimeError(
                f"Cache at {data_path} is in an unsupported format "
                f"(expected v{_CACHE_FORMAT_VERSION}). Delete it to rebuild — "
                f"it is not an Eden cache, or an Eden with a different cache layout wrote it."
            )

        sequences_sd = payload["sequences"]

        if not isinstance(meta.get("eden"), dict):
            en.logger.warning(
                f"Cache at {data_path} records no Eden provenance (it predates the record), so it cannot be checked "
                "against this Eden. If it may be stale, delete it to rebuild."
            )
        for key, name, current in (
            ("eden", "Eden", get_eden_provenance()),
            ("genesis", "Genesis", get_package_provenance("genesis-world")),
        ):
            if change := describe_provenance_change(meta.get(key), current):
                en.logger.warning(
                    f"{name} differs from the one that wrote {data_path} ({change}): the loaded data may be "
                    f"incompatible with this {name}."
                )
        if meta["target_framerate"] != self.target_framerate:
            raise ValueError(f"Target framerate mismatch: {meta['target_framerate']} != {self.target_framerate}")
        if (semantics := type(self)._cache_semantics) is not None and meta.get("cache_semantics") != semantics:
            raise ValueError(
                f"Cache at {data_path} was prepared by an earlier {type(self).__name__} "
                f"(cache_semantics {meta.get('cache_semantics')!r}); this one prepares {semantics!r}, so the cached "
                f"motion means something else. Delete it to rebuild, or point processed_data_path at a different "
                f"file. strict_cache_identity=False does not relax this: no setting makes the old motion right."
            )
        # Settings that shaped the cached bytes. A cache written before the key existed carries
        # none and loads untouched (the Eden warnings above, a mismatch or no record, are the signal for an old
        # cache); one that carries the key is read strictly, because what this catches is
        # silent — a library prepared under other settings loads and trains perfectly well.
        if (cached_identity := meta.get("cache_identity")) is not None:
            identity = cache_identity(self)
            differing = {
                name: (cached_identity[name], configured)
                for name, configured in identity.items()
                if name in cached_identity and cached_identity[name] != configured
            }
            untracked = [name for name in identity if name not in cached_identity]
            # Which unknowns are fatal, resolved before anything is logged or raised — else the
            # "accepting it" warning below can land above an exception rejecting the cache
            # anyway, since a differing value is fatal whatever the flag says.
            strict_failure = untracked if self.strict_cache_identity else []
            if differing or strict_failure:
                reasons = [f"{name}: cached {was!r} != configured {now!r}" for name, (was, now) in differing.items()]
                if strict_failure:
                    reasons.append(f"written before {', '.join(strict_failure)} was tracked, so its value is unknown")
                # Offered only when it would actually get the caller past this, i.e. every
                # reason is an unknown — naming it beside a differing value would read as "this
                # is how you proceed" for a conflict no flag relaxes.
                remedy = (
                    " Or, if you know those settings have not changed since the cache was built, "
                    "accept it with a warning by setting strict_cache_identity=False on the "
                    "dataset options — from the CLI that is a nested --override, e.g. "
                    "--override 'motion_options: {default: {dataset_options: "
                    "{strict_cache_identity: false}}}' for a motion-tracking task (dotted keys "
                    "are not supported)."
                    if strict_failure and not differing
                    else ""
                )
                raise ValueError(
                    f"Cache at {data_path} does not match the current settings "
                    f"({'; '.join(reasons)}). Delete it to rebuild, or point processed_data_path "
                    f"at a different file — these settings change the prepared motion, and "
                    f"reusing the cache would train on data the current config did not ask for."
                    f"{remedy}"
                )
            # Reached only when the load goes ahead, so non-empty here implies the opt-out.
            if untracked:
                en.logger.warning(
                    f"Cache at {data_path} was written before {', '.join(untracked)} was tracked, "
                    f"so it cannot be checked against the current config "
                    f"({', '.join(f'{name}={identity[name]!r}' for name in untracked)}). Accepting "
                    f"it because strict_cache_identity=False — if any of those settings has "
                    f"changed since the cache was built, this run trains on the older motion."
                )
        cached_window = meta.get("frame_window")
        trusted = cached_window is None and all(
            name in self._drops_cut_before_marker
            for name in ("drop_first_n_frames", "drop_last_n_frames")
            if getattr(self, name)
        )
        if self._trims_clips and cached_window != _FRAME_WINDOW_SEMANTICS and not trusted:
            raise ValueError(
                f"Cache at {data_path} was prepared with drop_first_n_frames={self.drop_first_n_frames} / "
                f"drop_last_n_frames={self.drop_last_n_frames} before every reader cut clips the same way (OMOMO and "
                f"Weave ignored both; AMASS, PHUMA and HHI wrapped a drop_last_n_frames longer than a clip around to "
                f"its start), so its clips may not be trimmed as configured. Delete it to rebuild, or point "
                f"processed_data_path at a different file."
            )
        # Saved as list (JSON-friendly), compared against tuple class attribute.
        if tuple(meta["motion_sequence_keys"]) != tuple(self.motion_sequence_keys):
            raise ValueError(
                f"Motion sequence keys mismatch: {meta['motion_sequence_keys']} != {self.motion_sequence_keys}"
            )

        sequences = {key: BatchMotionSequence.from_state_dict(sd) for key, sd in sequences_sd.items()}
        return sequences, meta

    def _parallel_read_motion_files(
        self,
        files: list[_TIn],
        read_fn: Callable[[_TIn], _TOut],
        max_workers: int | None = None,
        desc: str | None = None,
    ) -> list[_TOut]:
        """Read a list of raw motion items in parallel through ``read_fn``.

        Generic over the per-item input and output: most subclasses pass file
        paths (``str``) and return ``MotionSequence``, but Arctic returns a
        4-tuple per file, OMOMO/HHI-Assist a 2-tuple, and HHI-Assist passes
        ``(cg_path, cr_path)`` pairs. The TypeVars keep static-analysis
        accurate for whichever shape the caller chooses.

        ``np.load`` releases the GIL during file I/O, so a thread pool gives
        near-linear scaling for I/O-bound reads. For PHUMA-scale datasets
        (~73k tiny .npy files) this turns a multi-minute sequential parse
        into seconds.

        Subclasses opt in by replacing the ``list(map(...))`` call in their
        ``prepare()`` with this helper. One bad file among thousands must be
        findable from the error alone: a message built during the read can name
        the item from ``_READING_MOTION_FILE`` (the short-clip error does), and
        any other exception ``read_fn`` raises gets a note naming it.

        A clip too short to use (``_ShortClipError``) is left out with one
        warning for all of them (:meth:`_skip_short_clips`) — of the result and
        of :attr:`all_motion_files` alike, so ``files`` must line up 1:1 with
        :attr:`all_motion_files` (it is that list, or one item per entry of it).
        """
        if max_workers is None:
            # os.cpu_count() can be very high on slurm nodes; cap at 32 to
            # avoid swamping shared filesystems.
            max_workers = min(32, (os.cpu_count() or 4))

        try:
            from tqdm import tqdm  # type: ignore

            iterator = tqdm(
                files,
                desc=desc or f"Reading {len(files)} motion files",
                disable=eden_dist.is_distributed() and not eden_dist.is_main_process(),
            )
        except ImportError:
            iterator = files

        name = type(self).__name__

        def read(item: _TIn) -> _TOut | _ShortClipError:
            token = _READING_MOTION_FILE.set(item)
            try:
                return read_fn(item)
            except _ShortClipError as e:
                return e  # left out below: one short clip costs that clip, not the read
            except Exception as e:
                # A note reaches a traceback but not ``str(e)``; skip it when the message already names the file.
                if repr(item) not in str(e):
                    e.add_note(f"while reading motion file {item!r} ({name})")
                raise
            finally:
                _READING_MOTION_FILE.reset(token)

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            # ThreadPoolExecutor.map preserves input order, so the resulting
            # list aligns 1:1 with `files` — important because downstream code
            # (e.g. all_motion_files metadata) uses positional indexing.
            results = list(pool.map(read, iterator))
        short = [i for i, result in enumerate(results) if isinstance(result, _ShortClipError)]
        if not short:
            return results
        if len(files) != len(self.all_motion_files):
            raise RuntimeError(
                f"{name}: _parallel_read_motion_files left out {len(short)} short clip(s), but its {len(files)} items "
                f"do not line up with the {len(self.all_motion_files)} all_motion_files it must drop them from. Pass "
                f"items in all_motion_files order (one per file)."
            )
        self._skip_short_clips([results[i] for i in short], len(results))
        dropped = set(short)
        self.all_motion_files = [f for i, f in enumerate(self.all_motion_files) if i not in dropped]
        return [result for i, result in enumerate(results) if i not in dropped]

    def _prepare_single_key(
        self,
        read_fn: Callable[[str], MotionSequence],
        desc: str,
        robot: EntityOptions | None = None,
    ) -> tuple[dict[str, BatchMotionSequence], dict[str, Any]]:
        """``prepare()`` for a dataset of one actor, one clip per motion file.

        Reads :attr:`all_motion_files` through ``read_fn`` in parallel on the CPU (see
        :meth:`_parallel_read_motion_files`), batches the clips with room for every name in
        ``motion_links_name``, and moves the batch to the dataset device in one copy rather than one per clip.
        With ``robot``, the link fields are then filled by forward kinematics of that robot
        (:meth:`_forward_velocity_kinematics`) in ``motion_links_name`` order; without, they are what the reader
        packed. Returns no metadata.
        """
        (key,) = self.motion_sequence_keys
        sequences = self._parallel_read_motion_files(self.all_motion_files, read_fn, desc=desc)
        batch = BatchMotionSequence.from_motion_sequences(sequences, num_links=len(self.motion_links_name[key])).to(
            self.device
        )
        if robot is not None:
            self._forward_velocity_kinematics(batch, robot, link_names=self.motion_links_name[key])
        return {key: batch}, {}

    @property
    def _trims_clips(self) -> bool:
        """Whether ``drop_first_n_frames`` / ``drop_last_n_frames`` cut anything from a clip."""
        return bool(self.drop_first_n_frames or self.drop_last_n_frames)

    def _frame_window(self) -> slice:
        """The frames of a raw clip that ``drop_first_n_frames`` / ``drop_last_n_frames`` keep, as a frame-axis slice.

        A clip no longer than the two together comes out empty, which :meth:`_pack_motion_sequence` refuses as too
        short — never wrapped around from the other end.
        """
        return slice(self.drop_first_n_frames, -self.drop_last_n_frames or None)

    def _pack_motion_sequence(
        self,
        framerate: float | int,
        base_pos_w: torch.Tensor | np.ndarray,
        base_quat_w: torch.Tensor | np.ndarray,
        dofs_pos: torch.Tensor | np.ndarray | None = None,
        links_pos_w: torch.Tensor | np.ndarray | None = None,
        links_quat_w: torch.Tensor | np.ndarray | None = None,
        base_lin_vel_w: torch.Tensor | np.ndarray | None = None,
        base_ang_vel_w: torch.Tensor | np.ndarray | None = None,
        dofs_vel: torch.Tensor | np.ndarray | None = None,
        links_lin_vel_w: torch.Tensor | np.ndarray | None = None,
        links_ang_vel_w: torch.Tensor | np.ndarray | None = None,
        contact_points_w: torch.Tensor | np.ndarray | None = None,
        contact_points_valid: torch.Tensor | np.ndarray | None = None,
        contact_label: torch.Tensor | np.ndarray | None = None,
        derive_body_frame: bool = True,
        body_pose_quat: torch.Tensor | None = None,
        body_shape: torch.Tensor | None = None,
        body_model: str | None = None,
    ) -> MotionSequence:
        """
        Fill the MotionSequence object with the given motion data.

        Every field may be a tensor or a numpy array. A tensor is used as given; an array becomes a CPU tensor —
        a float32 one for every field but the bool ``contact_points_valid`` and the int8 ``contact_label``, which
        keep their dtype — sharing its memory where it can (see :func:`eden.utils.torch.numpy_to_tensor`).

        A clip needs at least 2 frames, both as read and after retiming to ``target_framerate``: fewer spans no time
        and has no velocity, so it raises a ``ValueError`` quoting the frame counts and the settings that shrink a
        clip, and names the file when it runs inside ``_parallel_read_motion_files``.

        Parameters
        ----------
        framerate : float | int
            The framerate of the motion sequence.
        base_pos_w : torch.Tensor
            The translation of the root of the motion sequence.
        base_quat_w : torch.Tensor
            The quaternion of the root of the motion sequence.
        dofs_pos : torch.Tensor | None
            The positions of the joints of the motion sequence.
        links_pos_w : torch.Tensor | None
            The positions of the links of the motion sequence.
        links_quat_w : torch.Tensor | None
            The quaternions of the links of the motion sequence.
        base_lin_vel_w : torch.Tensor | None
            The linear velocity of the root. If None, estimated from positions.
        base_ang_vel_w : torch.Tensor | None
            The angular velocity of the root. If None, estimated from quaternions.
        dofs_vel : torch.Tensor | None
            The velocities of the joints. If None, estimated from positions.
        links_lin_vel_w : torch.Tensor | None
            The linear velocities of the links. If None, estimated from positions.
        links_ang_vel_w : torch.Tensor | None
            The angular velocities of the links. If None, estimated from quaternions.
        contact_points_w : torch.Tensor | None
            World-frame demo contact points, (num_frames, num_links, num_partners, 3).
            Rows must follow the dataset's ``motion_links_name`` order. Resampled
            with nearest-neighbor (never blended) on framerate change.
        contact_points_valid : torch.Tensor | None
            Bool contact flags, (num_frames, num_links, num_partners). Nearest-neighbor
            resampled together with ``contact_points_w``.
        contact_label : torch.Tensor | None
            Tri-state per-link contact labels, int8 in {-1, 0, +1}, (num_frames, num_links), rows in the dataset's
            ``motion_links_name`` order. Nearest-neighbor resampled.
        derive_body_frame : bool
            Whether to derive the body-frame ``links_*_b`` fields from the world-frame ones. ``False`` leaves them
            ``None`` — for a large library whose consumers read only world-frame link fields (the tracking terms
            do), which halves the per-link memory. Defaults to True.
        body_pose_quat : torch.Tensor | None
            Body-model joint rotations below the root, wxyz, (num_frames, num_body_joints, 4); the root is
            ``base_pos_w`` (the root joint's position) and ``base_quat_w``. Slerped per joint on retiming. Given
            together with ``body_shape`` and ``body_model``, only when ``body_model_params`` is on.
        body_shape : torch.Tensor | None
            Body-model shape coefficients, (num_frames, num_betas). Linear on retiming.
        body_model : str | None
            The body model they parameterise, a key of :data:`eden.datasets.utils.body_model.BODY_MODELS`.

        Returns
        -------
        MotionSequence
            The motion sequence object.
        """
        # A numpy field becomes a CPU tensor: float32, but for the bool contact flags and the int8 contact labels.
        base_pos_w, base_quat_w, base_lin_vel_w, base_ang_vel_w = map(
            _as_tensor, (base_pos_w, base_quat_w, base_lin_vel_w, base_ang_vel_w)
        )
        links_pos_w, links_quat_w, links_lin_vel_w, links_ang_vel_w = map(
            _as_tensor, (links_pos_w, links_quat_w, links_lin_vel_w, links_ang_vel_w)
        )
        dofs_pos, dofs_vel, contact_points_w = map(_as_tensor, (dofs_pos, dofs_vel, contact_points_w))
        contact_points_valid = _as_tensor(contact_points_valid, dtype=None)
        contact_label = _as_tensor(contact_label, dtype=None)

        source_framerate, source_frames = framerate, base_pos_w.shape[0]
        if body_pose_quat is not None or body_shape is not None or body_model is not None:
            _check_body_model_fields(body_pose_quat, body_shape, body_model, source_frames)
        if source_frames < _MIN_CLIP_FRAMES:
            raise self._short_clip_error(source_frames, source_framerate, None)
        if framerate != self.target_framerate:
            (
                base_pos_w,
                base_quat_w,
                dofs_pos,
                links_pos_w,
                links_quat_w,
                base_lin_vel_w,
                base_ang_vel_w,
                dofs_vel,
                links_lin_vel_w,
                links_ang_vel_w,
                contact_points_w,
                contact_points_valid,
                contact_label,
            ) = interpolate_motion(
                base_pos_w,
                base_quat_w,
                dofs_pos,
                links_pos_w,
                links_quat_w,
                framerate,
                self.target_framerate,
                base_lin_vel_w=base_lin_vel_w,
                base_ang_vel_w=base_ang_vel_w,
                dofs_vel=dofs_vel,
                links_lin_vel_w=links_lin_vel_w,
                links_ang_vel_w=links_ang_vel_w,
                contact_points_w=contact_points_w,
                contact_points_valid=contact_points_valid,
                contact_label=contact_label,
            )
            if body_pose_quat is not None:
                body_pose_quat, body_shape = interpolate_body_model(
                    body_pose_quat, body_shape, framerate, self.target_framerate
                )
            framerate = self.target_framerate
            if base_pos_w.shape[0] < _MIN_CLIP_FRAMES:
                raise self._short_clip_error(source_frames, source_framerate, base_pos_w.shape[0])

        # Drop carried-through velocities so the FD step below treats them as
        # missing and re-estimates them from the (now retimed) position series.
        # Without this, ``interpolate_motion`` produces velocities that are
        # linearly resampled rather than consistent with the resampled qpos —
        # a small mismatch on retimed clips and a large one when the source
        # framerate differs noticeably from ``target_framerate``.
        if self.recalculate_velocities:
            base_lin_vel_w = None
            base_ang_vel_w = None
            if dofs_pos is not None:
                dofs_vel = None
            if links_pos_w is not None:
                links_lin_vel_w = None
            if links_quat_w is not None:
                links_ang_vel_w = None

        # Estimate any missing velocities from positions
        has_all_velocities = (
            base_lin_vel_w is not None
            and base_ang_vel_w is not None
            and (dofs_vel is not None or dofs_pos is None)
            and (links_lin_vel_w is not None or links_pos_w is None)
            and (links_ang_vel_w is not None or links_quat_w is None)
        )
        if not has_all_velocities:
            (
                est_base_lin_vel_w,
                est_base_ang_vel_w,
                est_dofs_vel,
                est_links_lin_vel_w,
                est_links_ang_vel_w,
            ) = estimate_motion_velocities(
                base_pos_w,
                base_quat_w,
                dofs_pos,
                links_pos_w,
                links_quat_w,
                1 / framerate,
            )
            if base_lin_vel_w is None:
                base_lin_vel_w = est_base_lin_vel_w
            if base_ang_vel_w is None:
                base_ang_vel_w = est_base_ang_vel_w
            if dofs_vel is None:
                dofs_vel = est_dofs_vel
            if links_lin_vel_w is None:
                links_lin_vel_w = est_links_lin_vel_w
            if links_ang_vel_w is None:
                links_ang_vel_w = est_links_ang_vel_w

        links_pos_b = links_quat_b = links_lin_vel_b = links_ang_vel_b = None
        if derive_body_frame:
            links_pos_b, links_quat_b, links_lin_vel_b, links_ang_vel_b = derive_body_frame_link_fields(
                base_pos_w=base_pos_w,
                base_quat_w=base_quat_w,
                base_lin_vel_w=base_lin_vel_w,
                base_ang_vel_w=base_ang_vel_w,
                links_pos_w=links_pos_w,
                links_quat_w=links_quat_w,
                links_lin_vel_w=links_lin_vel_w,
                links_ang_vel_w=links_ang_vel_w,
            )

        # Pin scalar metadata to the same device as the per-frame tensors so a
        # MotionSequence is internally device-consistent. Callers (e.g. PHUMA's
        # parallel CPU read) then pack a list of CPU MotionSequences and bulk-
        # move the resulting BatchMotionSequence in one call.
        seq_device = base_pos_w.device
        return MotionSequence(
            framerate=torch.as_tensor(framerate, device=seq_device, dtype=torch.float32),
            buffer_length=torch.as_tensor(base_pos_w.shape[0], device=seq_device, dtype=torch.int32),
            dofs_pos=dofs_pos,
            dofs_vel=dofs_vel,
            base_pos_w=base_pos_w,
            base_quat_w=base_quat_w,
            base_lin_vel_w=base_lin_vel_w,
            base_ang_vel_w=base_ang_vel_w,
            links_pos_w=links_pos_w,
            links_quat_w=links_quat_w,
            links_pos_b=links_pos_b,
            links_quat_b=links_quat_b,
            links_lin_vel_w=links_lin_vel_w,
            links_ang_vel_w=links_ang_vel_w,
            links_lin_vel_b=links_lin_vel_b,
            links_ang_vel_b=links_ang_vel_b,
            contact_points_w=contact_points_w,
            contact_points_valid=contact_points_valid,
            contact_label=contact_label,
            body_pose_quat=body_pose_quat,
            body_shape=body_shape,
            body_model=body_model,
        )

    def _short_clip_error(
        self, source_frames: int, source_framerate: float | int, retimed_frames: int | None
    ) -> _ShortClipError:
        """The ``_pack_motion_sequence`` error for a clip shorter than ``_MIN_CLIP_FRAMES``."""
        fix = (
            f"lower drop_first_n_frames / drop_last_n_frames (now {self.drop_first_n_frames} / "
            f"{self.drop_last_n_frames}) or the reader's own window (e.g. frame_range)"
        )
        if retimed_frames is None:
            frames = f"{source_frames} frame(s) at {source_framerate:g} Hz"
        else:
            frames = (
                f"{retimed_frames} frame(s) after retiming from {source_framerate:g} Hz to target_framerate="
                f"{self.target_framerate:g} Hz ({source_frames} at {source_framerate:g} Hz)"
            )
            fix += ", raise target_framerate"
        source = _READING_MOTION_FILE.get()
        clip = "a clip" if source is None else f"motion file {source!r}"
        message = (
            f"{type(self).__name__}: {clip} has {frames}; every clip needs at least {_MIN_CLIP_FRAMES}, because a "
            f"single frame spans no time and has no velocity.\n"
            f"Fix: {fix}, or leave the file out with exclude_pattern / exclude_list_file.\n"
            f"recalculate_velocities=False is not the fix: it skips only the velocity estimate, and the clip still "
            f"spans no time."
        )
        return _ShortClipError(message, clip=f"{clip} ({frames})", fix=fix)

    def _skip_short_clips(self, errors: list[_ShortClipError], total: int) -> None:
        """Warn about the ``errors`` clips left out of ``total`` for being too short; raise when that was every clip.

        A short clip is a problem in that clip's data, so it costs the clip, not the run. With nothing left there is
        no dataset to build, and the first clip's own message says why.
        """
        if not errors:
            return
        name = type(self).__name__
        if len(errors) == total:
            raise ValueError(
                f"{name}: all {total} clip(s) have fewer than {_MIN_CLIP_FRAMES} frames, so there is nothing left "
                f"to load. The first:\n{errors[0]}"
            )
        shown = ", ".join(e.clip for e in errors[:5]) + (f" and {len(errors) - 5} more" if len(errors) > 5 else "")
        fix = max((e.fix for e in errors), key=len)  # the retimed form also names target_framerate
        en.logger.warning(
            f"{name}: skipped {len(errors)} of {total} clip(s) with fewer than {_MIN_CLIP_FRAMES} frames (a single "
            f"frame spans no time): {shown}. To keep them, {fix}; to stop this warning, leave them out with "
            f"exclude_pattern / exclude_list_file."
        )

    def _forward_velocity_kinematics(
        self,
        motion_seq: BatchMotionSequence,
        entity_options: EntityOptions,
        link_names: list[str] | None = None,
    ):
        """Given the partially filled MotionSequence, fills the links velocities and positions.

        ``link_names`` is the desired column order of the link buffers (i.e. the dataset's
        ``motion_links_name``). When provided, link world-frame fields are extracted in that
        exact order, making the filled buffers independent of Genesis's internal kinematic-tree
        ordering (which changed from breadth-first to depth-first in genesis 1.2.0). When None,
        links are read in the robot's internal Genesis order, which silently mismatches
        ``motion_links_name`` whenever the two orders differ.
        """
        from eden.utils.genesis_runtime import transient_robot_env

        with transient_robot_env(
            entity_options,
            num_envs=len(motion_seq.buffer_length),
            sim_dt=float((1.0 / motion_seq.framerate[0]).item()),
        ) as (_env, robot):
            self._fill_link_velocities(motion_seq, robot, link_names=link_names)

    def _fill_link_velocities(self, motion_seq, robot, link_names=None) -> None:
        """Per-frame FK loop body for :meth:`_forward_velocity_kinematics`.

        Extracted so the env-lifecycle wrapper stays a small ``with`` block.
        Reads frame-by-frame qpos / qvel from ``motion_seq``, drives the
        robot's state, and writes the resolved link world-frame fields
        back into ``motion_seq``.

        ``link_names``, when given, pins the link-buffer column order to those names (resolved
        once up front) so the output is independent of Genesis's internal link ordering.
        """
        import torch  # local import to keep _forward_velocity_kinematics' lazy-import shape

        from genesis.utils.geom import inv_transform_by_quat

        # Resolve the link column order from names so the buffers are not tied to Genesis's
        # internal (BFS/DFS) link ordering. None -> Genesis' full internal order.
        ls_idx_local = None
        if link_names is not None:
            ls_idx_local = robot.find_named_links_idx_local(link_names, preserve_order=True)[1]

        # Built once, not per frame: the entity resolves a selector by identity, so a fresh tensor each
        # frame would be read back to the host each frame.
        base_and_joint_dofs_idx = torch.cat(
            [
                torch.arange(6, device=robot.dofs_idx_local.device, dtype=robot.dofs_idx_local.dtype),
                robot.dofs_idx_local,
            ],
            dim=0,
        )

        for i in range(motion_seq.buffer_length.max()):
            base_pos, valid_mask = motion_seq.base_pos_w.with_mask[:, i]
            base_quat = motion_seq.base_quat_w[:, i]
            dofs_pos = motion_seq.dofs_pos[:, i]
            base_lin_vel = motion_seq.base_lin_vel_w[:, i]
            base_ang_vel = motion_seq.base_ang_vel_w[:, i]
            dofs_vel = motion_seq.dofs_vel[:, i]

            robot.set_qpos(
                torch.cat(
                    [base_pos[valid_mask], base_quat[valid_mask], dofs_pos[valid_mask]],
                    dim=1,
                ),
                envs_idx=valid_mask,
            )
            # Genesis stores a free joint's rotational DOF velocity in the base BODY frame
            # (MuJoCo qvel convention), but base_ang_vel is cached in the WORLD frame. Rotate
            # world->body with the same base_quat written to qpos above before set_dofs_vel;
            # the linear DOFs are already world-frame and left untouched.
            base_ang_vel_b = inv_transform_by_quat(base_ang_vel[valid_mask], base_quat[valid_mask])
            robot.set_dofs_vel(
                torch.cat(
                    [
                        base_lin_vel[valid_mask],
                        base_ang_vel_b,
                        dofs_vel[valid_mask],
                    ],
                    dim=1,
                ),
                dofs_idx_local=base_and_joint_dofs_idx,
                envs_idx=valid_mask,
            )

            envs_idx = valid_mask.nonzero(as_tuple=False).squeeze(-1)
            # Only fill link buffers when the batch was allocated with link dim > 0.
            # When source motions have no link data, from_motion_sequences infers num_links=0;
            # callers that run forward kinematics should pass num_links (e.g. from motion_links_name).
            num_links_buf = motion_seq.links_pos_w.shape[1]
            if num_links_buf > 0:
                links_pos_w = robot.get_links_pos(ls_idx_local=ls_idx_local, envs_idx=valid_mask)
                links_quat_w = robot.get_links_quat(ls_idx_local=ls_idx_local, envs_idx=valid_mask)
                links_lin_vel_w = robot.get_links_vel(ls_idx_local=ls_idx_local, envs_idx=valid_mask)
                links_ang_vel_w = robot.get_links_ang(ls_idx_local=ls_idx_local, envs_idx=valid_mask)
                motion_seq.links_pos_w[envs_idx, i] = links_pos_w
                motion_seq.links_quat_w[envs_idx, i] = links_quat_w
                motion_seq.links_lin_vel_w[envs_idx, i] = links_lin_vel_w
                motion_seq.links_ang_vel_w[envs_idx, i] = links_ang_vel_w
            else:
                en.logger.warning(f"No link data for batch {envs_idx}, frame {i}")
