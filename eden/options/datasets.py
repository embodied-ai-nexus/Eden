"""Dataset and dataset-modifier configuration options."""

from __future__ import annotations

from pydantic import NonNegativeInt

from eden.options.options import DynamicConfigurableOptions


class DatasetModifierOptions(DynamicConfigurableOptions):
    """
    Options for dataset modifiers.

    Parameters
    ----------
    motion_sequence_keys: tuple[str, ...] | None
        Keys in ``DatasetBase.motion_sequences`` this modifier applies to. When
        ``None`` the modifier is applied to every key.
    """

    motion_sequence_keys: tuple[str, ...] | None = None


class DatasetOptions(DynamicConfigurableOptions):
    """
    Dataset options.

    Parameters
    ----------
    registry: str | None
        HuggingFace namespace (user or organisation) that hosts the dataset repo.
        Subclasses should set this to enable automatic downloads, e.g.
        `registry = "ember-lab-berkeley"`.
    dataset: str | None
        HuggingFace dataset repository name, e.g. `"AMASS_Retargeted_for_G1"`.
    raw_data_path: str | None
        Path to the raw data directory. When provided it is used as-is and must
        already exist — a missing explicit path raises rather than downloading.
        When None and the dataset class defines `registry` / `dataset`, the
        dataset is downloaded to the HuggingFace global cache.
    processed_data_path: str | None
        Path to the processed (cached) data file.
    target_framerate: float
        The target framerate of the motion.
    drop_first_n_frames: int
        Frames dropped from the start of each raw clip during ``prepare()``. Must be >= 0. A dataset that
        cannot trim its clips at read time refuses a non-zero value; trim with the ``TrimFrames`` modifier.
    drop_last_n_frames: int
        Frames dropped from the end of each raw clip during ``prepare()``. Must be >= 0; see
        ``drop_first_n_frames``.
    modifiers: tuple[DatasetModifierOptions, ...] | list[DatasetModifierOptions]
        Ordered sequence of offline transforms applied to the loaded motion
        sequences after cache load or raw prepare. Modifiers are not baked into
        the processed cache — changing the config re-applies them on next load.
    include_pattern: str | None
        Regex applied to each motion file's basename via ``re.search``. When set,
        only files matching are kept. ``None`` keeps every file. Applied before
        ``prepare()`` so unmatched files are never read or interpolated. Plain
        substrings work as regexes ("walk" keeps any file whose basename
        contains "walk").
    exclude_pattern: str | None
        Regex applied to each motion file's basename via ``re.search``. Files
        matching are dropped after ``include_pattern`` filtering. ``None``
        disables exclusion.
    include_list_file: str | None
        Path to a YAML or JSON file enumerating motion files to keep. The file
        may be (a) a flat list of paths/basenames, (b) a list of dicts each
        with a ``file:`` key (ProtoMotions ``amass_*_train.yaml`` shape), or
        (c) a dict wrapping either form under a ``motions:`` key. Extensions are
        stripped. An entry with a directory component matches only a file with
        the same trailing path (``CMU/45/45_01_stageii.motion`` matches
        ``.../CMU/45/45_01_stageii.npz``); a bare basename matches any file of
        that name. Applied alongside ``include_pattern``.
    exclude_list_file: str | None
        Path to a YAML or JSON file enumerating motion files to drop. Same
        accepted shapes and matching rules as ``include_list_file``. Applied
        alongside ``exclude_pattern``.
    """

    registry: str | None = None
    dataset: str | None = None
    raw_data_path: str | None = None
    processed_data_path: str | None = None
    target_framerate: float = 50.0
    drop_first_n_frames: NonNegativeInt = 0
    drop_last_n_frames: NonNegativeInt = 0
    modifiers: tuple[DatasetModifierOptions, ...] | list[DatasetModifierOptions] = ()
    include_pattern: str | None = None
    exclude_pattern: str | None = None
    include_list_file: str | None = None
    exclude_list_file: str | None = None
