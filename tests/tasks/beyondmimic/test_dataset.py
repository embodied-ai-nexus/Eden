"""``BvhCsvG1Dataset``: where the default clip and its processed cache live, and what keys the cache."""

import pathlib

import pytest

from eden.datasets.base import cache_identity
from eden.tasks import TASK_REGISTRY
from eden.tasks.beyondmimic.dataset import BvhCsvG1Dataset, _default_processed_path
from eden.utils.assets import SAMPLE_DATA_REPO, sample_data_path
from eden.utils.configs import overrides_from_str
from eden.utils.misc import get_cache_dir


def _dataset_from(dataset_options) -> BvhCsvG1Dataset:
    """An uninitialized dataset carrying ``dataset_options``' fields.

    Constructing it for real would download and prepare the clip; the cache-path derivation only reads
    fields with class-level defaults, which ``cache_identity`` documents as valid on an uninitialized instance.
    """
    dataset = object.__new__(BvhCsvG1Dataset)
    for name in BvhCsvG1Dataset.get_parameter_names():
        if hasattr(dataset_options, name):
            setattr(dataset, name, getattr(dataset_options, name))
    return dataset


def _derived_cache(**mod_kwargs) -> str:
    """The processed-cache path the task config built with ``mod_kwargs`` resolves to."""
    options = TASK_REGISTRY.build("beyondmimic", **mod_kwargs).motion_options.default.dataset_options
    dataset = _dataset_from(options)
    return dataset.processed_data_path or _default_processed_path(dataset)


def test_default_paths_stay_unset_in_the_config_and_resolve_under_the_user_cache():
    """Deriving at load time keeps a saved config portable and puts the key after ``--override``."""
    options = TASK_REGISTRY.build("beyondmimic").motion_options.default.dataset_options
    assert options.raw_data_path is None
    assert options.processed_data_path is None
    assert pathlib.Path(_derived_cache()).is_relative_to(get_cache_dir())


def test_both_cache_paths_follow_xdg_cache_home(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    assert pathlib.Path(_derived_cache()).is_relative_to(tmp_path / "eden")

    # A pre-seeded sample-data cache: the clip resolves there without touching the network.
    clip = tmp_path / "eden" / SAMPLE_DATA_REPO / "beyondmimic" / "clip.csv"
    clip.parent.mkdir(parents=True)
    clip.write_text("")
    assert sample_data_path("beyondmimic", "clip.csv") == str(clip)


def test_processed_cache_is_keyed_by_raw_source_and_target_framerate():
    """A run with its own clips, or at another rate, must not load (or be refused) the default's cache."""
    default = _derived_cache()
    mine = _derived_cache(raw_data_path="/data/my_clips")
    assert mine != default
    assert _derived_cache(raw_data_path="/data/my_clips/") == mine  # normalized, and stable across builds
    assert _derived_cache(target_framerate=30.0) != default
    assert _derived_cache(raw_data_path="/data/my_clips", target_framerate=30.0) != mine
    # An explicit path is used verbatim: the derivation is a default, not a policy.
    assert _derived_cache(processed_data_path="/tmp/mine.pt") == "/tmp/mine.pt"


def test_every_hard_checked_setting_keys_its_own_cache_file():
    """Everything ``load_cached`` refuses a cache over must also change the cache path.

    Walks ``cache_identity`` rather than a hand-written list, so a field added to ``_cache_identity_fields``
    is in the key from the same commit.
    """
    baseline = object.__new__(BvhCsvG1Dataset)
    identity = cache_identity(baseline)
    assert "frame_range" in identity, identity
    for name, value in identity.items():
        changed = object.__new__(BvhCsvG1Dataset)
        setattr(changed, name, (7, 9) if isinstance(value, (list, tuple)) else (value or 0) + 1)
        assert _default_processed_path(changed) != _default_processed_path(baseline), name


def test_an_override_the_mod_cannot_see_still_reaches_the_cache_key():
    """``frame_range`` is reachable only through ``--override``, which runs after the task mod."""
    cfg = TASK_REGISTRY.build("beyondmimic").with_overrides_from_dict(
        overrides_from_str("motion_options: {default: {dataset_options: {frame_range: [2, 1942]}}}")
    )
    dataset = _dataset_from(cfg.motion_options.default.dataset_options)
    assert cache_identity(dataset)["frame_range"] == [2, 1942]
    assert _default_processed_path(dataset) != _derived_cache()


def test_an_empty_raw_path_does_not_share_the_default_key():
    empty, omitted = object.__new__(BvhCsvG1Dataset), object.__new__(BvhCsvG1Dataset)
    empty.raw_data_path = ""
    assert _default_processed_path(empty) != _default_processed_path(omitted)


def test_a_missing_raw_path_suggests_the_download_this_dataset_can_do(tmp_path):
    options = BvhCsvG1Dataset.configure(raw_data_path="/no/such/dir", processed_data_path=str(tmp_path / "x.pt"))
    with pytest.raises(FileNotFoundError, match="'/no/such/dir' does not exist.*Pass raw_data_path=None"):
        BvhCsvG1Dataset(options=options, device="cpu")


def test_a_frame_window_is_refused_since_frame_range_is_how_this_reader_trims(tmp_path):
    options = BvhCsvG1Dataset.configure(
        raw_data_path="/no/such/dir", processed_data_path=str(tmp_path / "x.pt"), drop_last_n_frames=3
    )
    with pytest.raises(ValueError, match="BvhCsvG1Dataset does not trim its clips"):
        BvhCsvG1Dataset(options=options, device="cpu")
