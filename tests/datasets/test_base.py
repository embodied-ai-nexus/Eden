"""Tests for ``eden.datasets.base.DatasetBase``, driven through a real subclass over clips written to ``tmp_path``.

Covers the processed-cache format and its identity check, the include/exclude list-file filters, the
raw-path diagnostics, the modifier pass, the config-time DOF helpers, and the HuggingFace download's
fetch → unpack → narrow stages.
"""

from __future__ import annotations

import gc
import json
import re
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest
import torch
import yaml

from eden.datasets.base import (
    _CACHE_FORMAT_VERSION,
    DATASET_REGISTRY,
    _assert_min_buffer_lengths,
    _cache_identity_field_names,
)
from eden.datasets.modifiers import SpeedScale, Translate
from eden.utils.motion_reference.motion_data import BatchMotionSequence
from eden.utils.misc import get_eden_provenance
from tests.datasets.conftest import DOFS_NAME, ClipDataset, write_clip
from tests.support import motion

BASE_IDENTITY_FIELDS = ("drop_first_n_frames", "drop_last_n_frames", "recalculate_velocities")


@pytest.fixture
def raw(tmp_path):
    """A four-clip library: two subjects' directories plus the DFaust one."""
    root = tmp_path / "raw"
    for rel, frames in (
        ("CMU/45/45_01_stageii.npz", 4),
        ("CMU/138/138_18_stageii.npz", 5),
        ("CMU/138/138_03_stageii.npz", 6),
        ("DFaust/50009/50009_hips_stageii.npz", 7),
    ):
        write_clip(root / rel, num_frames=frames)
    return root


def _write_list(path, entries):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(entries, f) if path.suffix == ".json" else yaml.safe_dump(entries, f)
    return str(path)


def _files(dataset) -> list[str]:
    return sorted(dataset.metadata["all_motion_files"])


def _payload(path):
    return torch.load(path, weights_only=False)


def _forget_identity(path, *names):
    """Rewrite the cache at ``path`` as one written before ``names`` joined the tracked fields."""
    payload = _payload(path)
    for name in names:
        del payload["_meta"]["cache_identity"][name]
    torch.save(payload, path)


def _forget_frame_window(path):
    """Rewrite the cache at ``path`` as one written before ``_FRAME_WINDOW_SEMANTICS`` was recorded."""
    payload = _payload(path)
    del payload["_meta"]["frame_window"]
    torch.save(payload, path)


class TestProcessedCache:
    def test_a_second_construction_loads_what_the_first_prepared(self, raw, tmp_path, make_dataset):
        cache = tmp_path / "cache.pt"
        prepared = make_dataset(raw_data_path=str(raw), processed_data_path=str(cache))

        (tmp_path / "raw").rename(tmp_path / "gone")  # the second one cannot have read raw data
        loaded = make_dataset(raw_data_path=str(raw), processed_data_path=str(cache))

        assert torch.equal(loaded.buffer_length, prepared.buffer_length)
        assert torch.equal(
            loaded.motion_sequences["g1"].dofs_pos.contiguous(), prepared.motion_sequences["g1"].dofs_pos.contiguous()
        )
        assert _files(loaded) == _files(prepared)

    def test_the_file_is_a_meta_envelope_around_the_sequences(self, raw, tmp_path, make_dataset):
        """``_format_version`` lives in ``_meta`` only, matching ``eden/utils/configs.py``'s envelope."""
        make_dataset(raw_data_path=str(raw), processed_data_path=str(tmp_path / "cache.pt"))
        payload = _payload(tmp_path / "cache.pt")

        assert set(payload) == {"_meta", "sequences"}
        assert {"_format_version", "eden", "genesis", "motion_sequence_keys"} <= set(payload["_meta"])
        assert payload["_meta"]["eden"] == get_eden_provenance()

    @pytest.mark.parametrize(
        ("saved", "warning"),
        [
            ({"version": "0.0.1", "commit": None}, "Eden differs from the one that wrote"),
            # A cache older than the record cannot be checked, which the cache-identity check relies on being said.
            (None, "records no Eden provenance"),
        ],
    )
    def test_a_cache_from_another_or_an_unknown_eden_loads_with_a_warning(
        self, raw, tmp_path, make_dataset, eden_caplog, saved, warning
    ):
        cache = tmp_path / "cache.pt"
        make_dataset(raw_data_path=str(raw), processed_data_path=str(cache))
        payload = _payload(cache)
        if saved is None:
            del payload["_meta"]["eden"]
        else:
            payload["_meta"]["eden"] = saved
        torch.save(payload, cache)

        eden_caplog.clear()
        make_dataset(raw_data_path=str(raw), processed_data_path=str(cache))
        assert [r.levelname for r in eden_caplog.records if warning in r.message] == ["WARNING"]

    def test_resaving_a_cache_this_process_loaded_keeps_it_intact(self, raw, tmp_path, make_dataset):
        """The loaded sequences are mmap'd from that file; saving in place used to truncate it to 0 bytes."""
        cache = tmp_path / "cache.pt"
        make_dataset(raw_data_path=str(raw), processed_data_path=str(cache))
        loaded = make_dataset(raw_data_path=str(raw), processed_data_path=str(cache))
        expected = loaded.motion_sequences["g1"].dofs_pos.contiguous().clone()

        loaded.save_cached(str(cache))

        reloaded = make_dataset(raw_data_path=str(raw), processed_data_path=str(cache))
        assert torch.equal(reloaded.motion_sequences["g1"].dofs_pos.contiguous(), expected)
        # `loaded` still maps the file the save replaced, which an NFS client keeps as a `.nfs*` entry until unmapped
        del loaded
        gc.collect()
        assert sorted(p.name for p in tmp_path.iterdir()) == ["cache.pt", "raw"]  # no temp file left behind

    def test_a_save_after_a_load_writes_the_current_format_version(self, raw, tmp_path, make_dataset):
        dataset = make_dataset(raw_data_path=str(raw))
        dataset.metadata["_format_version"] = 999  # what a loaded older envelope would carry
        dataset.save_cached(str(tmp_path / "cache.pt"))

        assert _payload(tmp_path / "cache.pt")["_meta"]["_format_version"] == _CACHE_FORMAT_VERSION

    @pytest.mark.parametrize("layout", ["tuple-root", "older-version"])
    def test_an_unsupported_cache_is_refused_with_the_fix(self, raw, tmp_path, make_dataset, layout):
        """Accepting it would hand the consumer a corrupted library, or basename-only file metadata."""
        cache = tmp_path / "cache.pt"
        if layout == "tuple-root":
            torch.save(({"g1": None}, {"target_framerate": 50.0}), cache)
        else:
            torch.save(
                {"_meta": {"_format_version": 2, "all_motion_files": ["45_01_stageii.npz"]}, "sequences": {}}, cache
            )

        with pytest.raises(RuntimeError, match=r"unsupported format.*Delete it to rebuild"):
            make_dataset(raw_data_path=str(raw), processed_data_path=str(cache))

    def test_the_include_list_is_reapplied_to_a_loaded_cache(self, raw, tmp_path, make_dataset):
        """File metadata is stored as relpaths, so a directory-qualified list entry still matches after a load."""
        options = {
            "raw_data_path": str(raw),
            "processed_data_path": str(tmp_path / "cache.pt"),
            "include_list_file": _write_list(tmp_path / "keep.yaml", ["CMU/45/45_01_stageii.npz"]),
        }
        assert _files(make_dataset(**options)) == ["CMU/45/45_01_stageii.npz"]

        loaded = make_dataset(**options)
        assert _files(loaded) == ["CMU/45/45_01_stageii.npz"]
        assert len(loaded) == 1


class TestCacheIdentity:
    """A cache must not outlive the prepare-time settings that shaped it."""

    @pytest.fixture
    def cache(self, raw, tmp_path, make_dataset):
        """Prepare with ``frame_range=None`` and return ``(path, make_reader)``."""
        path = tmp_path / "cache.pt"
        make_dataset(raw_data_path=str(raw), processed_data_path=str(path))
        return path, lambda **options: make_dataset(raw_data_path=str(raw), processed_data_path=str(path), **options)

    def test_every_declared_field_is_recorded_base_fields_first(self, raw, tmp_path, make_dataset):
        """A subclass declares only what it adds; the base's fields cannot be dropped by re-declaring."""
        make_dataset(raw_data_path=str(raw), processed_data_path=str(tmp_path / "cache.pt"), frame_range=(1, 5))
        identity = _payload(tmp_path / "cache.pt")["_meta"]["cache_identity"]

        assert list(identity) == [*BASE_IDENTITY_FIELDS, "frame_range", "input_fps"]
        assert identity["frame_range"] == [1, 5]  # normalised to how it comes back off disk

    def test_the_shipped_subclasses_extend_the_base_fields(self):
        """Read off the class: building these needs their raw formats (BVH, ConTrack h5)."""
        from eden.tasks.beyondmimic.dataset import BvhCsvG1Dataset
        from eden.tasks.contrack.dataset import DexterHandDataset

        assert set(_cache_identity_field_names(BvhCsvG1Dataset)) == {*BASE_IDENTITY_FIELDS, "input_fps", "frame_range"}
        assert set(_cache_identity_field_names(DexterHandDataset)) == {*BASE_IDENTITY_FIELDS, "side"}

    def test_the_same_settings_load(self, cache):
        _, reader = cache
        assert len(reader(frame_range=None)) == 4

    def test_a_changed_setting_raises_naming_the_difference_without_offering_the_opt_out(self, cache):
        """The opt-out would not get past a conflict, so naming it would read as how to proceed."""
        _, reader = cache
        with pytest.raises(ValueError, match=r"frame_range: cached None != configured \[1, 5\]") as caught:
            reader(frame_range=(1, 5))
        assert "strict_cache_identity" not in str(caught.value)

    def test_a_field_added_after_the_cache_was_written_is_unknown_not_fine(self, cache):
        """Otherwise extending ``_cache_identity_fields`` would protect no cache already on disk."""
        path, reader = cache
        _forget_identity(path, "frame_range")

        with pytest.raises(ValueError, match="written before frame_range was tracked") as caught:
            reader(frame_range=None)
        assert "strict_cache_identity=False" in str(caught.value)  # re-preparing can cost hours

    def test_the_opt_out_accepts_an_unknown_field_with_a_warning_naming_it(self, cache, eden_caplog):
        path, reader = cache
        _forget_identity(path, "frame_range")

        assert len(reader(frame_range=None, strict_cache_identity=False)) == 4
        assert any("frame_range" in r.message and r.levelname == "WARNING" for r in eden_caplog.records)

    def test_the_opt_out_does_not_relax_a_differing_value(self, cache):
        _, reader = cache
        with pytest.raises(ValueError, match=r"frame_range: cached None != configured \[1, 5\]"):
            reader(frame_range=(1, 5), strict_cache_identity=False)

    def test_a_cache_written_before_the_identity_key_existed_still_loads(self, cache):
        """The key is additive: no format bump, so no existing AMASS cache is invalidated by it."""
        path, reader = cache
        payload = _payload(path)
        del payload["_meta"]["cache_identity"]
        torch.save(payload, path)

        assert len(reader(frame_range=(1, 5))) == 4

    def test_a_tensor_valued_setting_compares_instead_of_raising(self, raw, tmp_path, make_dataset):
        """``cached != configured`` on two tensors is elementwise, so the truth test used to raise."""
        writer = make_dataset(raw_data_path=str(raw))
        writer.frame_range = torch.tensor([1, 5])
        writer.save_cached(str(tmp_path / "cache.pt"))
        assert _payload(tmp_path / "cache.pt")["_meta"]["cache_identity"]["frame_range"] == [1, 5]

        reader = make_dataset(raw_data_path=str(raw))
        reader.frame_range = torch.tensor([1, 5])
        sequences, _ = reader.load_cached(str(tmp_path / "cache.pt"))
        assert set(sequences) == {"g1"}

        reader.frame_range = torch.tensor([2, 6])
        with pytest.raises(ValueError, match=r"frame_range: cached \[1, 5\] != configured \[2, 6\]"):
            reader.load_cached(str(tmp_path / "cache.pt"))

    class TestConflictAndUnknownTogether:
        """The opt-out waives unknowns, never a conflict; with both, the load fails and never claims acceptance."""

        @pytest.fixture
        def mixed(self, cache):
            path, reader = cache
            _forget_identity(path, "input_fps")
            return lambda **options: reader(frame_range=(1, 5), input_fps=30, **options)

        def test_the_opt_out_logs_no_acceptance_of_a_cache_it_then_rejects(self, mixed, eden_caplog):
            with pytest.raises(ValueError, match=r"frame_range: cached None != configured \[1, 5\]") as caught:
                mixed(strict_cache_identity=False)
            assert not [r for r in eden_caplog.records if "Accepting it" in r.message]
            assert "input_fps" not in str(caught.value)  # the waived unknown is not a stated reason

        def test_strict_mode_reports_both_reasons_but_offers_no_remedy(self, mixed):
            with pytest.raises(ValueError) as caught:
                mixed()
            message = str(caught.value)
            assert "frame_range: cached None != configured [1, 5]" in message
            assert "written before input_fps was tracked" in message
            assert "strict_cache_identity" not in message


class TestListFiles:
    """``include_list_file`` / ``exclude_list_file``: YAML or JSON lists of clips, matched on the extension-less relpath."""

    @pytest.mark.parametrize(
        ("name", "entries", "expected"),
        [
            (
                "list.yaml",
                ["CMU/45/45_01_stageii.npz", "DFaust/50009/50009_hips_stageii.npz"],
                ["CMU/45/45_01_stageii.npz", "DFaust/50009/50009_hips_stageii.npz"],
            ),
            (
                "amass_train.yaml",  # ProtoMotions: dicts naming the original .motion files
                [
                    {"file": "CMU/45/45_01_stageii.motion", "fps": 120.0, "idx": 0},
                    {"file": "CMU/138/138_18_stageii.motion"},
                ],
                ["CMU/138/138_18_stageii.npz", "CMU/45/45_01_stageii.npz"],
            ),
            (
                "list.yaml",
                {"motions": [{"file": "CMU/45/45_01_stageii.motion"}, "DFaust/50009/50009_hips_stageii.npz"]},
                ["CMU/45/45_01_stageii.npz", "DFaust/50009/50009_hips_stageii.npz"],
            ),
            (
                "list.json",
                ["138_03_stageii.npz", "45_01_stageii"],
                ["CMU/138/138_03_stageii.npz", "CMU/45/45_01_stageii.npz"],
            ),
            (
                "list.yaml",
                ["CMU/138/138_18_stageii.npz", "45_01_stageii"],
                ["CMU/138/138_18_stageii.npz", "CMU/45/45_01_stageii.npz"],
            ),
        ],
        ids=["flat-relpaths", "protomotions-dicts", "motions-wrapper", "json-basenames", "mixed"],
    )
    def test_every_accepted_shape_selects_its_clips(self, raw, tmp_path, make_dataset, name, entries, expected):
        dataset = make_dataset(raw_data_path=str(raw), include_list_file=_write_list(tmp_path / name, entries))
        assert _files(dataset) == expected

    @pytest.mark.parametrize(
        ("name", "entries", "message"),
        [
            ("list.txt", None, "must have a .yaml/.yml/.json extension"),
            ("list.yaml", {"clips": ["x.npz"]}, "no 'motions' key"),
            ("list.yaml", 42, "must contain a list"),
            ("list.yaml", [{"fps": 120.0}], "no 'file' key"),
            ("list.yaml", [{"file": 123}], "must be a string"),
            ("list.yaml", [["nested"]], "must be a string or dict"),
            ("list.yaml", [], "zero motion entries"),
        ],
        ids=["extension", "no-motions-key", "not-a-list", "no-file-key", "non-string-file", "nested-list", "empty"],
    )
    def test_a_malformed_list_is_refused_naming_the_problem(self, raw, tmp_path, make_dataset, name, entries, message):
        path = tmp_path / name
        if entries is None:
            path.write_text("anything\n")
        else:
            _write_list(path, entries)
        with pytest.raises(ValueError, match=message):
            make_dataset(raw_data_path=str(raw), include_list_file=str(path))

    def test_an_exclude_list_drops_its_clips(self, raw, tmp_path, make_dataset):
        dataset = make_dataset(
            raw_data_path=str(raw),
            exclude_list_file=_write_list(tmp_path / "drop.yaml", ["CMU/138/138_03_stageii.npz"]),
        )
        assert _files(dataset) == [
            "CMU/138/138_18_stageii.npz",
            "CMU/45/45_01_stageii.npz",
            "DFaust/50009/50009_hips_stageii.npz",
        ]

    def test_a_list_and_a_regex_combine(self, raw, tmp_path, make_dataset):
        """The regex filters match the basename, so ``^138_`` drops the listed 138_18 clip."""
        keep = _write_list(tmp_path / "keep.yaml", ["CMU/45/45_01_stageii.motion", "CMU/138/138_18_stageii.motion"])
        dataset = make_dataset(raw_data_path=str(raw), include_list_file=keep, exclude_pattern=r"^138_")
        assert _files(dataset) == ["CMU/45/45_01_stageii.npz"]

    def test_a_filter_that_keeps_nothing_raises(self, raw, tmp_path, make_dataset):
        with pytest.raises(ValueError, match="matched no files"):
            make_dataset(
                raw_data_path=str(raw), include_list_file=_write_list(tmp_path / "keep.yaml", ["no_such_motion.npz"])
            )

    def test_listed_clips_missing_on_disk_are_counted_in_a_warning(self, raw, tmp_path, make_dataset, eden_caplog):
        keep = _write_list(
            tmp_path / "keep.yaml",
            ["CMU/45/45_01_stageii.motion", "Missing/one_stageii.motion", "Missing/two_stageii.motion"],
        )
        dataset = make_dataset(raw_data_path=str(raw), include_list_file=keep)
        assert _files(dataset) == ["CMU/45/45_01_stageii.npz"]
        assert "2 had no matching file on disk" in eden_caplog.text

    @pytest.fixture
    def same_named(self, tmp_path):
        """HumanEva reuses one clip name across subjects, plus a clip at the root of the library."""
        root = tmp_path / "same_named"
        for rel in (
            "HumanEva/S1/Static_stageii.npz",
            "HumanEva/S2/Static_stageii.npz",
            "HumanEva/S3/Static_stageii.npz",
            "Static_stageii.npz",
            "45_01_stageii.npz",
        ):
            write_clip(root / rel)
        return root

    @pytest.mark.parametrize(
        ("entry", "expected"),
        [
            ("HumanEva/S3/Static_stageii.motion", ["HumanEva/S3/Static_stageii.npz"]),
            (
                "Static_stageii",
                [
                    "HumanEva/S1/Static_stageii.npz",
                    "HumanEva/S2/Static_stageii.npz",
                    "HumanEva/S3/Static_stageii.npz",
                    "Static_stageii.npz",
                ],
            ),
            ("45_01_stageii", ["45_01_stageii.npz"]),
        ],
        ids=["relpath-is-exact", "basename-matches-all", "root-level-file"],
    )
    def test_a_relpath_entry_matches_one_clip_and_a_basename_entry_every_same_named_one(
        self, same_named, tmp_path, make_dataset, entry, expected
    ):
        dataset = make_dataset(
            raw_data_path=str(same_named), include_list_file=_write_list(tmp_path / "keep.yaml", [entry])
        )
        assert _files(dataset) == expected

    def test_a_directory_entry_does_not_match_a_same_named_root_level_clip(self, same_named, tmp_path, make_dataset):
        """That clip was never listed under that directory; pulling it in would be silently wrong data."""
        with pytest.raises(ValueError, match="matched no files"):
            make_dataset(
                raw_data_path=str(same_named),
                include_list_file=_write_list(tmp_path / "keep.yaml", ["CMU/45/45_01_stageii.npz"]),
            )


class TestRawDataPath:
    def test_an_empty_directory_says_the_glob_found_nothing(self, tmp_path, make_dataset):
        """Not "matched no files out of 0", which reads as a filter problem when the mount is the problem."""
        (tmp_path / "empty").mkdir()
        with pytest.raises(FileNotFoundError, match=r"Found 0 '\.npz' files under"):
            make_dataset(raw_data_path=str(tmp_path / "empty"))

    def test_a_missing_explicit_path_raises_listing_siblings_instead_of_downloading(self, tmp_path, make_dataset):
        class HubBackedDataset(ClipDataset):
            registry = "some-org"
            dataset = "SomeDataset"

            def _download_from_hf(self):  # the network boundary: reaching it is the failure
                raise AssertionError("an explicit raw_data_path must never trigger a download")

        (tmp_path / "subset_0000").mkdir()
        (tmp_path / "subset_0001").mkdir()

        with pytest.raises(FileNotFoundError) as caught:
            make_dataset(HubBackedDataset, raw_data_path=str(tmp_path / "s_0000"))

        message = str(caught.value)
        assert "'" + str(tmp_path / "s_0000") + "' does not exist" in message
        assert "subset_0000" in message and "subset_0001" in message


class _TwoKeyDataset(ClipDataset):
    """The same clips as two motion keys, the way a robot + object dataset carries them."""

    motion_sequence_keys: ClassVar[tuple[str, ...]] = ("robot", "object")
    motion_dofs_name: ClassVar[dict[str, list[str]]] = {"robot": DOFS_NAME, "object": DOFS_NAME}
    motion_links_name: ClassVar[dict[str, list[str]]] = {"robot": [], "object": []}

    def prepare(self):
        sequences, metadata = super().prepare()
        return {"robot": sequences["g1"], "object": ClipDataset.prepare(self)[0]["g1"]}, metadata


class TestModifiers:
    def test_they_run_once_at_construction_and_refuse_a_second_pass(self, raw, make_dataset):
        """Translate adds in place and Jitter draws fresh noise, so a second pass would compound silently."""
        plain = make_dataset(raw_data_path=str(raw))
        shifted = make_dataset(raw_data_path=str(raw), modifiers=[Translate.configure(offset=(0.0, 0.0, 1.0))])

        before = plain.motion_sequences["g1"].base_pos_w.contiguous()
        after = shifted.motion_sequences["g1"].base_pos_w.contiguous()
        torch.testing.assert_close(after, before + torch.tensor([0.0, 0.0, 1.0]))
        with pytest.raises(RuntimeError, match="called twice"):
            shifted._apply_modifiers()  # the guard's only entry point is a repeated call

    def test_a_modifier_that_desyncs_the_motion_keys_is_refused(self, tmp_path, make_dataset):
        """Retiming only the robot key leaves it a different length from the object key it is paired with."""
        write_clip(tmp_path / "raw" / "0.npz", num_frames=10)
        write_clip(tmp_path / "raw" / "1.npz", num_frames=6)
        speed_up_robot = SpeedScale.configure(speed=2.0, motion_sequence_keys=("robot",))
        with pytest.raises(ValueError, match="mismatched per-sequence lengths"):
            make_dataset(_TwoKeyDataset, raw_data_path=str(tmp_path / "raw"), modifiers=[speed_up_robot])

    def test_a_two_frame_clip_packs(self, tmp_path, make_dataset):
        write_clip(tmp_path / "raw" / "short.npz", num_frames=2)
        dataset = make_dataset(raw_data_path=str(tmp_path / "raw"))

        assert len(dataset) == 1
        assert dataset.motion_sequences["g1"].base_ang_vel_w.contiguous().shape == (2, 3)


class TestShortClips:
    """A clip needs 2 frames. A short one is a problem in that clip's data: it is left out with a warning.

    Only a dataset with nothing else left is refused, and either way the file is named in the message itself.
    """

    def test_a_one_frame_clip_is_left_out_with_a_warning_naming_the_file_its_frame_count_and_the_fix(
        self, tmp_path, make_dataset, eden_caplog
    ):
        write_clip(tmp_path / "raw" / "long.npz", num_frames=8)
        short = write_clip(tmp_path / "raw" / "short.npz", num_frames=1)

        dataset = make_dataset(raw_data_path=str(tmp_path / "raw"))

        assert len(dataset) == 1
        assert dataset.metadata["all_motion_files"] == ["long.npz"]  # per-clip metadata stays aligned with the clips
        (message,) = [r.getMessage() for r in eden_caplog.records if "skipped" in r.getMessage()]
        assert "ClipDataset: skipped 1 of 2 clip(s) with fewer than 2 frames" in message
        assert f"motion file {str(short)!r} (1 frame(s) at 50 Hz)" in message
        assert "lower drop_first_n_frames / drop_last_n_frames (now 0 / 0)" in message
        assert "exclude_pattern / exclude_list_file" in message

    def test_a_dataset_of_only_short_clips_is_refused_naming_the_file_its_frame_count_and_the_fix(
        self, tmp_path, make_dataset
    ):
        short = write_clip(tmp_path / "raw" / "short.npz", num_frames=1)

        with pytest.raises(ValueError) as caught:
            make_dataset(raw_data_path=str(tmp_path / "raw"))

        message = str(caught.value)  # the file is in the message itself, which a caller printing str(exc) keeps
        assert "ClipDataset: all 1 clip(s) have fewer than 2 frames, so there is nothing left to load." in message
        assert (
            f"ClipDataset: motion file {str(short)!r} has 1 frame(s) at 50 Hz; every clip needs at least 2" in message
        )
        assert "drop_first_n_frames / drop_last_n_frames (now 0 / 0)" in message
        assert "recalculate_velocities=False is not the fix" in message
        assert "target_framerate" not in message  # the clip was not retimed, so the rate is not the cause
        assert not hasattr(caught.value, "__notes__")  # already named, so no note repeats it

    def test_a_one_frame_clip_that_needs_retiming_is_rejected_before_the_retime(self, tmp_path, make_dataset):
        """``interpolate_motion`` refuses one frame too, with an error that does not say why."""
        write_clip(tmp_path / "raw" / "short.npz", num_frames=1, framerate=60.0)
        with pytest.raises(ValueError, match=re.escape("has 1 frame(s) at 60 Hz;")):
            make_dataset(raw_data_path=str(tmp_path / "raw"))

    def test_a_clip_retimed_below_two_frames_quotes_both_rates_and_both_counts(self, tmp_path, make_dataset):
        """Downsampling runs before the velocity estimate, so a clip long enough at its source rate can come out short."""
        short = write_clip(tmp_path / "raw" / "fast.npz", num_frames=3, framerate=200.0)

        with pytest.raises(ValueError) as caught:
            make_dataset(raw_data_path=str(tmp_path / "raw"))

        message = str(caught.value)
        assert (
            f"motion file {str(short)!r} has 1 frame(s) after retiming from 200 Hz to target_framerate=50 Hz "
            "(3 at 200 Hz)" in message
        )
        assert "raise target_framerate" in message

    def test_recalculate_velocities_off_does_not_let_a_one_frame_clip_through(self, tmp_path, make_dataset):
        """Skipping the estimate would pack it, as a reference that spans no time."""
        write_clip(tmp_path / "raw" / "short.npz", num_frames=1)
        with pytest.raises(ValueError, match=re.escape("has 1 frame(s)")):
            make_dataset(raw_data_path=str(tmp_path / "raw"), recalculate_velocities=False)

    def test_any_other_error_a_reader_raises_gets_a_note_naming_its_file(self, tmp_path, make_dataset):
        """The short-clip message names its file itself; an error from anywhere else in the read still points at it."""
        write_clip(tmp_path / "raw" / "good.npz", num_frames=8)
        bad = tmp_path / "raw" / "truncated.npz"
        bad.write_bytes(b"not an npz archive")

        with pytest.raises(Exception) as caught:
            make_dataset(raw_data_path=str(tmp_path / "raw"))

        assert caught.value.__notes__ == [f"while reading motion file {str(bad)!r} (ClipDataset)"]

    def test_a_short_clip_packed_outside_the_parallel_reader_still_raises_without_a_file(self, make_dataset, tmp_path):
        """A reader that packs by hand gets the frame counts and the fix; the file is named only by the shared reader."""
        write_clip(tmp_path / "raw" / "long.npz", num_frames=8)
        dataset = make_dataset(raw_data_path=str(tmp_path / "raw"))
        with pytest.raises(ValueError, match=re.escape("ClipDataset: a clip has 1 frame(s) at 50 Hz;")):
            dataset._pack_motion_sequence(
                framerate=50.0, base_pos_w=torch.zeros(1, 3), base_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]])
            )


class _WindowedDataset(ClipDataset):
    """``ClipDataset`` trimming its clips by ``drop_first_n_frames`` / ``drop_last_n_frames``, as the shipped readers do.

    It packs the npz arrays as they are, the way the readers hand ``_pack_motion_sequence`` numpy.
    """

    _applies_frame_window: ClassVar[bool] = True

    def _read_single_sequence(self, path: str):
        data = np.load(path)
        window = self._frame_window()
        return self._pack_motion_sequence(
            framerate=float(data["framerate"]),
            base_pos_w=data["base_pos_w"][window].astype(np.float64),
            base_quat_w=data["base_quat_w"][window],
            dofs_pos=data["dofs_pos"][window],
        )


class _TrustedWindowedDataset(_WindowedDataset):
    """``_WindowedDataset`` as a reader that already cut its clips by ``_frame_window`` before caches recorded it."""

    _drops_cut_before_marker: ClassVar[tuple[str, ...]] = ("drop_first_n_frames", "drop_last_n_frames")


class _FirstDropTrustedWindowedDataset(_WindowedDataset):
    """As AMASS, PHUMA and HHI: ``drop_first_n_frames`` was cut as now, a long ``drop_last_n_frames`` could wrap."""

    _drops_cut_before_marker: ClassVar[tuple[str, ...]] = ("drop_first_n_frames",)


class TestFrameWindow:
    def test_the_window_keeps_the_frames_between_the_drops(self, tmp_path, make_dataset):
        write_clip(tmp_path / "raw" / "0.npz", num_frames=8)
        dataset = make_dataset(
            _WindowedDataset, raw_data_path=str(tmp_path / "raw"), drop_first_n_frames=2, drop_last_n_frames=1
        )

        dofs_pos = dataset.motion_sequences["g1"].dofs_pos[0]
        expected = torch.arange(24, dtype=torch.float32).view(8, 3)[2:7] * 0.01
        torch.testing.assert_close(dofs_pos, expected)
        # numpy fields pack as float32 tensors, a float64 one included
        assert dataset.motion_sequences["g1"].base_pos_w.dtype == torch.float32

    def test_a_window_longer_than_the_clip_empties_it_rather_than_wrapping_around(self, tmp_path, make_dataset):
        write_clip(tmp_path / "raw" / "0.npz", num_frames=8)
        with pytest.raises(ValueError, match=re.escape("has 0 frame(s)")):
            make_dataset(_WindowedDataset, raw_data_path=str(tmp_path / "raw"), drop_last_n_frames=9)

    @pytest.mark.parametrize("field", ["drop_first_n_frames", "drop_last_n_frames"])
    def test_a_negative_drop_is_refused_by_the_options(self, field):
        with pytest.raises(ValueError, match=f"{field}\n  Input should be greater than or equal to 0"):
            ClipDataset.configure(**{field: -1})

    def test_a_reader_that_does_not_opt_in_refuses_a_window_and_names_the_modifier_that_can(self, raw, make_dataset):
        """Opt-in, so a reader whose read path never calls ``_frame_window`` cannot ignore the window by omission."""
        with pytest.raises(ValueError, match=re.escape("modifiers=[TrimFrames.configure(trim_first=0, trim_last=3)]")):
            make_dataset(ClipDataset, raw_data_path=str(raw), drop_last_n_frames=3)
        assert len(make_dataset(ClipDataset, raw_data_path=str(raw))) > 0

    def test_arctic_refuses_a_window_before_reading_anything(self, tmp_path):
        from eden.datasets.arctic_dataset import ArcticDataset

        options = ArcticDataset.configure(raw_data_path=str(tmp_path / "absent"), drop_first_n_frames=1)
        with pytest.raises(ValueError, match="ArcticDataset does not trim its clips"):
            ArcticDataset(options=options, device=torch.device("cpu"))

    class TestCachesFromBeforeEveryReaderTrimmed:
        """Caches record the drop values whether or not the reader applied them, so the identity alone can't tell."""

        @pytest.fixture
        def trimmed_cache(self, tmp_path, make_dataset):
            write_clip(tmp_path / "raw" / "0.npz", num_frames=8)
            path = tmp_path / "cache.pt"

            def prepare(cls=_WindowedDataset, **window):
                return make_dataset(cls, raw_data_path=str(tmp_path / "raw"), processed_data_path=str(path), **window)

            return path, prepare

        def test_one_that_trimmed_under_the_current_rules_loads(self, trimmed_cache):
            path, prepare = trimmed_cache
            prepare(drop_first_n_frames=2)
            (path.parent / "raw").rename(path.parent / "gone")  # the second one cannot have read raw data
            assert prepare(drop_first_n_frames=2).motion_sequences["g1"].dofs_pos[0].shape[0] == 6

        def test_one_that_recorded_a_window_without_the_marker_is_refused(self, trimmed_cache):
            path, prepare = trimmed_cache
            prepare(drop_first_n_frames=2)
            _forget_frame_window(path)

            with pytest.raises(ValueError, match=r"prepared with drop_first_n_frames=2 / drop_last_n_frames=0 before"):
                prepare(drop_first_n_frames=2)

        def test_an_untrimmed_one_without_the_marker_still_loads(self, trimmed_cache):
            """No format bump: a cache with no window is the same bytes under either rule."""
            path, prepare = trimmed_cache
            prepare()
            _forget_frame_window(path)

            assert len(prepare()) == 1

        def test_one_without_the_marker_loads_from_a_reader_that_already_trimmed_this_way(self, trimmed_cache):
            """LAFAN1, SEED, DexYCB and the hand-object readers cut clips by the same window before the marker."""
            path, prepare = trimmed_cache
            prepare(_TrustedWindowedDataset, drop_first_n_frames=2)
            _forget_frame_window(path)
            (path.parent / "raw").rename(path.parent / "gone")  # the second one cannot have read raw data

            assert (
                prepare(_TrustedWindowedDataset, drop_first_n_frames=2).motion_sequences["g1"].dofs_pos[0].shape[0] == 6
            )

        def test_one_trimmed_only_at_the_start_loads_from_a_reader_whose_start_cut_was_right(self, trimmed_cache):
            """AMASS / PHUMA / HHI: only a ``drop_last_n_frames`` longer than a clip was cut wrong."""
            path, prepare = trimmed_cache
            prepare(_FirstDropTrustedWindowedDataset, drop_first_n_frames=2)
            _forget_frame_window(path)
            (path.parent / "raw").rename(path.parent / "gone")  # the second one cannot have read raw data

            dataset = prepare(_FirstDropTrustedWindowedDataset, drop_first_n_frames=2)
            assert dataset.motion_sequences["g1"].dofs_pos[0].shape[0] == 6

        def test_that_reader_refuses_one_trimmed_at_the_end(self, trimmed_cache):
            path, prepare = trimmed_cache
            prepare(_FirstDropTrustedWindowedDataset, drop_first_n_frames=2, drop_last_n_frames=1)
            _forget_frame_window(path)

            with pytest.raises(ValueError, match=r"prepared with drop_first_n_frames=2 / drop_last_n_frames=1 before"):
                prepare(_FirstDropTrustedWindowedDataset, drop_first_n_frames=2, drop_last_n_frames=1)

        def test_that_reader_still_refuses_a_cache_marked_under_other_rules(self, trimmed_cache):
            """The trust covers only a cache written before the marker existed, not one under another window rule."""
            path, prepare = trimmed_cache
            prepare(_TrustedWindowedDataset, drop_first_n_frames=2)
            payload = _payload(path)
            payload["_meta"]["frame_window"] = 1
            torch.save(payload, path)

            with pytest.raises(ValueError, match=r"prepared with drop_first_n_frames=2 / drop_last_n_frames=0 before"):
                prepare(_TrustedWindowedDataset, drop_first_n_frames=2)


class TestDofsPosHelpers:
    """``compute_dofs_pos_range`` / ``compute_dofs_pos_at`` feed ``soft_dofs_pos_limits`` / ``default_dofs_pos``."""

    @pytest.fixture
    def two_clips(self, tmp_path, make_dataset):
        """``wrist_tx`` spans 0.1..0.9 and 0.0..0.5; ``palm_yaw`` spans -1..1 and -0.5..0.5."""
        write_clip(tmp_path / "raw" / "0.npz", np.stack([np.linspace(0.1, 0.9, 10), np.linspace(-1.0, 1.0, 10)], -1))
        write_clip(tmp_path / "raw" / "1.npz", np.stack([np.linspace(0.0, 0.5, 5), np.linspace(-0.5, 0.5, 5)], -1))
        dataset = make_dataset(raw_data_path=str(tmp_path / "raw"), motion_dofs_name={"g1": ["wrist_tx", "palm_yaw"]})
        first = dataset.all_motion_files[0].endswith("0.npz")
        return dataset, (0, 1) if first else (1, 0)

    def test_the_range_spans_every_loaded_frame_plus_the_margin_as_plain_floats(self, two_clips):
        dataset, _ = two_clips
        limits = dataset.compute_dofs_pos_range("g1", margin=0.1)

        assert limits["wrist_tx"] == pytest.approx((-0.1, 1.0), abs=1e-5)
        assert limits["palm_yaw"] == pytest.approx((-1.1, 1.1), abs=1e-5)
        assert all(type(v) is float for pair in limits.values() for v in pair)  # no device binding at config time

    def test_a_substring_override_replaces_the_margin_and_a_double_match_warns(
        self, tmp_path, make_dataset, eden_caplog
    ):
        write_clip(tmp_path / "raw" / "0.npz", np.zeros((3, 3)))
        dataset = make_dataset(
            raw_data_path=str(tmp_path / "raw"), motion_dofs_name={"g1": ["wrist_tx", "wrist_ty", "palm_yaw_joint"]}
        )

        limits = dataset.compute_dofs_pos_range(
            "g1", margin=0.15, margin_overrides={"tx": 0.2, "palm": 0.3, "yaw": 0.5}
        )

        assert limits["wrist_tx"] == pytest.approx((-0.2, 0.2))
        assert limits["wrist_ty"] == pytest.approx((-0.15, 0.15))
        assert limits["palm_yaw_joint"] == pytest.approx((-0.5, 0.5))  # the last matching substring wins...
        assert "palm_yaw_joint" in eden_caplog.text and "matched margin overrides" in eden_caplog.text  # ...loudly

    @pytest.mark.parametrize(
        ("frame", "clip", "expected"), [(0, 0, (0.1, -1.0)), (0, 1, (0.0, -0.5)), (9, 0, (0.9, 1.0))]
    )
    def test_the_pose_at_a_frame_of_a_clip(self, two_clips, frame, clip, expected):
        dataset, order = two_clips
        # sequence_idx counts loaded clips, which follow the file listing's order
        pose = dataset.compute_dofs_pos_at("g1", frame=frame, sequence_idx=order.index(clip))
        assert (pose["wrist_tx"], pose["palm_yaw"]) == pytest.approx(expected, abs=1e-5)
        assert all(type(v) is float for v in pose.values())

    def test_the_default_pose_is_the_first_frame_of_the_first_loaded_clip(self, two_clips):
        dataset, _ = two_clips
        assert dataset.compute_dofs_pos_at("g1") == dataset.compute_dofs_pos_at("g1", frame=0, sequence_idx=0)

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"frame": -1}, "must be >= 0"),
            ({"sequence_idx": -1}, "must be >= 0"),
            ({"sequence_idx": 2}, "has 2 sequence"),
        ],
    )
    def test_a_bad_frame_or_clip_index_is_refused(self, two_clips, kwargs, message):
        dataset, _ = two_clips
        with pytest.raises(ValueError, match=message):
            dataset.compute_dofs_pos_at("g1", **kwargs)

    def test_a_frame_past_the_chosen_clip_names_its_length(self, two_clips):
        dataset, order = two_clips
        with pytest.raises(ValueError, match="buffer_length=5"):
            dataset.compute_dofs_pos_at("g1", frame=5, sequence_idx=order.index(1))

    def test_an_unknown_entity_is_a_key_error(self, two_clips):
        with pytest.raises(KeyError):
            two_clips[0].compute_dofs_pos_range("not_an_entity")

    def test_an_entity_without_dofs_data_is_refused(self, tmp_path, make_dataset):
        write_clip(tmp_path / "raw" / "0.npz", np.zeros((3, 0)))  # base pose only
        dataset = make_dataset(raw_data_path=str(tmp_path / "raw"), motion_dofs_name={"g1": []})
        with pytest.raises(ValueError, match="no dofs_pos data"):
            dataset.compute_dofs_pos_range("g1")

    @pytest.mark.parametrize("helper", ["compute_dofs_pos_range", "compute_dofs_pos_at"])
    def test_fewer_names_than_dof_columns_is_a_desync_and_both_helpers_say_so(self, raw, make_dataset, helper):
        """A bad ``frame`` must not mask the desync."""
        dataset = make_dataset(raw_data_path=str(raw), motion_dofs_name={"g1": ["only_one"]})
        kwargs = {"frame": 999} if helper == "compute_dofs_pos_at" else {}
        with pytest.raises(ValueError, match="desynchronized"):
            getattr(dataset, helper)("g1", **kwargs)

    @pytest.mark.parametrize(
        ("dofs_idx", "expected_names"),
        [
            ([1, 0], ["palm_yaw", "wrist_tx"]),
            (torch.tensor([1, 0]), ["palm_yaw", "wrist_tx"]),
            (slice(1, None), ["palm_yaw"]),
            ([1], ["palm_yaw"]),
        ],
        ids=["list", "tensor", "slice", "subset"],
    )
    def test_select_dofs_selects_the_names_with_the_columns(self, two_clips, dofs_idx, expected_names):
        """A subset or a reorder keeps every name on its own column."""
        dataset, _ = two_clips
        before = dataset.compute_dofs_pos_range("g1", margin=0.1)

        dataset.select_dofs({"g1": dofs_idx})

        assert dataset.motion_dofs_name["g1"] == expected_names
        assert dataset.compute_dofs_pos_range("g1", margin=0.1) == {name: before[name] for name in expected_names}
        assert type(dataset).motion_dofs_name == {"g1": DOFS_NAME}  # the class default is left alone


@pytest.mark.parametrize(("recalculate", "bogus_survives"), [(True, False), (False, True)])
def test_retimed_velocities_are_finite_differenced_unless_disabled(tmp_path, recalculate, bogus_survives):
    """Source velocities are inconsistent with retimed positions; the default re-estimates them."""
    bogus = np.full((60, len(motion.ROBOT_DOFS)), 99.0, dtype=np.float32)
    motion.write_clip(tmp_path / "clip_00.npz", 60, framerate=30.0, overrides={"robot.dofs_vel": bogus})

    dataset = motion.load_dataset(tmp_path, target_framerate=50.0, recalculate_velocities=recalculate)

    dofs_vel = dataset.motion_sequences["robot"].dofs_vel.concatenated_tensor
    # The default clip moves every joint 0.001 rad per source frame: 0.03 rad/s at 30 Hz.
    assert (dofs_vel.abs().max().item() > 50.0) is bogus_survives
    if not bogus_survives:
        np.testing.assert_allclose(dofs_vel.numpy(), 0.03, atol=1e-4)


@pytest.mark.parametrize("length", [0, 1])
def test_a_clip_shorter_than_two_frames_is_rejected_by_name(length):
    """Called directly: such a clip cannot be packed from raw data, but a modifier can still shrink one this far."""
    seq = BatchMotionSequence.make_empty(buffer_lengths=[10, length, 5], num_dofs=2, num_links=1)

    with pytest.raises(ValueError, match=rf"1 clip\(s\) shorter than 2 frames: \['b.npz'\] with \[{length}\] frame"):
        _assert_min_buffer_lengths({"robot": seq}, ("robot",), ["a.npz", "b.npz", "c.npz"])
    _assert_min_buffer_lengths({"robot": seq.select([0, 2])}, ("robot",), ["a.npz", "c.npz"])


class TestPerClipMetadata:
    """``_PER_CLIP_METADATA_KEYS`` stay aligned with the trajectories through every subset a dataset takes."""

    LABELS = ["box", "chair", "box", "lamp"]

    @pytest.fixture
    def labelled(self, tmp_path):
        return motion.write_library(tmp_path / "raw", [5, 6, 7, 8], labels=self.LABELS)

    def test_motion_labels_are_one_per_clip_in_clip_order(self, labelled):
        assert motion.load_dataset(labelled).motion_labels("objects") == self.LABELS

    def test_select_trajectories_reindexes_the_labels_with_the_clips(self, labelled):
        dataset = motion.load_dataset(labelled)
        dataset.select_trajectories(torch.tensor([3, 0]))
        assert dataset.motion_labels("objects") == ["lamp", "box"]
        dataset.select_trajectories(slice(1, None))
        assert dataset.motion_labels("objects") == ["box"]
        assert len(dataset) == 1

    def test_a_file_filter_on_a_cache_load_drops_the_filtered_clips_labels_too(self, labelled, tmp_path):
        cache = str(tmp_path / "cache.pt")
        motion.load_dataset(labelled, processed_data_path=cache)  # writes the full-library cache
        dataset = motion.load_dataset(labelled, processed_data_path=cache, exclude_pattern="clip_01")
        assert dataset.motion_labels("objects") == ["box", "box", "lamp"]
        assert dataset.buffer_length.tolist() == [5, 7, 8]

    def test_an_undeclared_key_names_the_declared_ones(self, labelled):
        with pytest.raises(KeyError, match=r"no per-clip metadata 'subjects'.*\['objects'\]"):
            motion.load_dataset(labelled).motion_labels("subjects")

    def test_labels_out_of_step_with_the_trajectories_are_refused(self, labelled):
        dataset = motion.load_dataset(labelled)
        dataset.metadata["objects"] = dataset.metadata["objects"][:2]
        with pytest.raises(ValueError, match=r"has 2 entries but the dataset holds 4 trajectories"):
            dataset.motion_labels("objects")


class TestPackContactLabel:
    def test_contact_label_is_packed_and_retimed_nearest_neighbour(self, tmp_path):
        """30 -> 50 Hz: each label is some source frame's label, never a blend."""
        labels = np.zeros((30, len(motion.ROBOT_LINKS)), dtype=np.int8)
        labels[::3] = 1
        labels[1::3] = -1
        motion.write_clip(tmp_path / "clip_00.npz", 30, framerate=30.0, overrides={"robot.contact_label": labels})

        packed = motion.load_dataset(tmp_path, target_framerate=50.0).motion_sequences["robot"].contact_label
        values = packed.concatenated_tensor
        assert values.dtype == torch.int8 and values.shape[1] == len(motion.ROBOT_LINKS)
        assert set(values.unique().tolist()) <= {-1, 0, 1} and values.shape[0] > 30

    def test_derive_body_frame_false_leaves_the_body_frame_link_fields_unset(self, tmp_path):
        dataset = motion.load_dataset(motion.write_library(tmp_path, [6]))
        pos = torch.zeros(6, 3)
        quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(6, 1)
        links = torch.zeros(6, 2, 3)
        kwargs = dict(links_pos_w=links, links_quat_w=quat[:, None].repeat(1, 2, 1))
        derived = dataset._pack_motion_sequence(50.0, pos, quat, **kwargs)
        bare = dataset._pack_motion_sequence(50.0, pos, quat, derive_body_frame=False, **kwargs)
        for field in ("links_pos_b", "links_quat_b", "links_lin_vel_b", "links_ang_vel_b"):
            assert getattr(derived, field) is not None and getattr(bare, field) is None
        torch.testing.assert_close(bare.links_lin_vel_w, derived.links_lin_vel_w)

    def test_numpy_contact_fields_keep_their_dtype_when_packed(self, raw, make_dataset):
        dataset = make_dataset(raw_data_path=str(raw))
        frames = 4
        quat = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (frames, 1))
        seq = dataset._pack_motion_sequence(
            50.0,
            np.zeros((frames, 3)),
            quat,
            links_pos_w=np.zeros((frames, 2, 3)),
            links_quat_w=np.tile(quat[:, None], (1, 2, 1)),
            contact_points_w=np.zeros((frames, 2, 1, 3)),
            contact_points_valid=np.ones((frames, 2, 1), dtype=bool),
            contact_label=np.ones((frames, 2), dtype=np.int8),
        )
        assert (seq.base_pos_w.dtype, seq.contact_points_w.dtype) == (torch.float32, torch.float32)
        assert (seq.contact_points_valid.dtype, seq.contact_label.dtype) == (torch.bool, torch.int8)


class HubClipDataset(ClipDataset):
    """``ClipDataset`` backed by a HuggingFace repo, so ``raw_data_path=None`` downloads it."""

    registry = "acme"
    dataset = "motions"


class TestHubNarrowing:
    """The download runs fetch (``_hf_allow_patterns``) → unpack (``_post_download_process``) → narrow
    (``_hf_data_subdir``).
    """

    def test_allow_patterns_reach_the_fetch_and_the_subdir_narrows_the_glob(self, hub, make_dataset):
        snapshot = hub.snapshot("acme/motions")
        write_clip(snapshot / "g1" / "walk.npz")
        write_clip(snapshot / "h1" / "run.npz", np.zeros((4, 1)))  # another robot's width: must not be read

        class G1Only(HubClipDataset):
            _hf_allow_patterns = ("g1/*", "LICENSE")
            _hf_data_subdir = "g1"

        dataset = make_dataset(G1Only)

        assert hub.calls[0]["allow_patterns"] == ["g1/*", "LICENSE"]
        assert dataset.raw_data_path == str(snapshot / "g1")
        assert dataset.metadata["all_motion_files"] == ["walk.npz"]

    def test_without_knobs_the_whole_repo_is_fetched_and_read(self, hub, make_dataset):
        """Opt-in: undeclared must reach the client as ``None`` — an empty list there matches nothing."""
        snapshot = hub.snapshot("acme/motions")
        write_clip(snapshot / "a" / "walk.npz")
        write_clip(snapshot / "b" / "run.npz")

        dataset = make_dataset(HubClipDataset)

        assert hub.calls[0]["allow_patterns"] is None
        assert dataset.raw_data_path == str(snapshot)
        assert sorted(dataset.metadata["all_motion_files"]) == ["a/walk.npz", "b/run.npz"]

    def test_unpacking_runs_before_narrowing(self, hub, make_dataset):
        """SEED's and PHUMA's subdirectory only exists once their archive is unpacked."""
        hub.snapshot("acme/motions")

        class Archived(HubClipDataset):
            _hf_data_subdir = "data/g1"

            def _post_download_process(self, local_dir):
                write_clip(Path(local_dir) / "data" / "g1" / "walk.npz")
                return local_dir

        assert make_dataset(Archived).metadata["all_motion_files"] == ["walk.npz"]

    class TestMissingSubdir:
        """``_hf_data_subdir`` absent after unpacking: the message says which of two causes it is."""

        def test_a_layout_without_the_top_directory_lists_what_is_there(self, hub, make_dataset):
            snapshot = hub.snapshot("acme/motions")
            write_clip(snapshot / "h1" / "walk.npz")
            write_clip(snapshot / "h1_2" / "walk.npz")

            class G1Only(HubClipDataset):
                _hf_data_subdir = "g1"

            with pytest.raises(FileNotFoundError) as caught:
                make_dataset(G1Only)
            message = str(caught.value)
            assert f"G1Only expects 'g1' under '{snapshot}', and there is no 'g1' there" in message
            assert "h1, h1_2" in message and "'acme/motions' repo" in message
            assert "Setting _hf_data_subdir to None is not the fix" in message
            # This class declares no allow-patterns, so they cannot be the cause.
            assert "_hf_allow_patterns" not in message

        def test_declared_allow_patterns_are_offered_as_a_cause(self, hub, make_dataset):
            hub.snapshot("acme/motions")

            class G1Only(HubClipDataset):
                _hf_allow_patterns = ("g1/*",)
                _hf_data_subdir = "g1"

            with pytest.raises(FileNotFoundError, match="`_hf_allow_patterns` does not fetch it"):
                make_dataset(G1Only)

        def test_a_partial_tree_names_its_top_directory_to_delete_never_the_root(self, hub, make_dataset):
            """The root holds the archive — and, on SEED's explicit path, is the caller's own data directory."""
            snapshot = hub.snapshot("acme/motions")
            (snapshot / "data" / "h1_2").mkdir(parents=True)

            class Archived(HubClipDataset):
                _hf_data_subdir = "data/g1"

            with pytest.raises(FileNotFoundError) as caught:
                make_dataset(Archived)
            message = str(caught.value)
            assert f"Delete '{snapshot / 'data'}' and rerun" in message
            assert "contents: h1_2" in message
            assert f"Delete '{snapshot}'" not in message

    def test_no_repo_owning_dataset_hand_rolls_its_download(self):
        """Narrowing goes through the two knobs; an override of ``_download_from_hf`` opts out of the base download
        (the shared Hub seam, a retargeted ``registry`` / ``dataset``). A loader that owns no repo may still resolve
        its own single source — the split ``DatasetBase._can_auto_download`` draws.
        """
        import eden.tasks.beyondmimic.dataset  # noqa: F401 — task-owned loaders register on import

        offenders = [
            cls.__name__
            for _, cls in DATASET_REGISTRY
            if (cls.registry is not None or cls.dataset is not None)
            and ("_download_from_hf" in cls.__dict__ or "_resolve_data_subdir" in cls.__dict__)
        ]
        assert not offenders, f"{offenders} own a repo but override the download; declare the knobs instead"
        assert {"lafan1_retargeted_g1_dataset", "seed_retargeted_g1_dataset", "phuma_dataset"} <= set(
            DATASET_REGISTRY.keys()
        )

    def test_the_download_goes_through_the_one_hub_seam(self, hub, make_dataset):
        """``hub`` records the client as ``eden.utils.assets`` holds it, which only ``_snapshot_download_locked``
        calls; a bare ``huggingface_hub.snapshot_download`` — opting out of the per-file lock that makes concurrent
        ranks safe — would fetch without being recorded.
        """
        write_clip(hub.snapshot("acme/motions") / "walk.npz")
        make_dataset(HubClipDataset)
        assert [call["repo_id"] for call in hub.calls] == ["acme/motions"]


class TestKnobDeclarations:
    """A subclass default never reaches Pydantic unless configured, so it is checked when the class is defined."""

    @pytest.mark.parametrize("subdir", ["/abs/g1", "../g1", "g1/../..", "g1//csv", "g1/", "", "g1\\csv"])
    def test_a_subdir_that_escapes_or_no_ops_the_snapshot_is_rejected(self, subdir):
        with pytest.raises(ValueError, match=r"Bad\._hf_data_subdir"):

            class Bad(ClipDataset):
                _hf_data_subdir = subdir

    @pytest.mark.parametrize(
        ("patterns", "error"),
        [
            ((), ValueError),
            ([], TypeError),
            ("g1/*", TypeError),
            (("g1/*", ""), ValueError),
            (("g1/*", None), TypeError),
        ],
    )
    def test_patterns_that_fetch_nothing_or_iterate_a_string_are_rejected(self, patterns, error):
        with pytest.raises(error, match=r"Bad\._hf_allow_patterns"):

            class Bad(ClipDataset):
                _hf_allow_patterns = patterns

    def test_the_shapes_the_shipped_datasets_use_are_accepted_and_stay_off_the_options(self):
        """The layout belongs to the class, so the options never carry it — no saved config can disagree with it."""

        class Fine(ClipDataset):
            _hf_allow_patterns = ("g1.tar.gz", "LICENSE.md")
            _hf_data_subdir = "data/g1"

        assert not {"_hf_allow_patterns", "_hf_data_subdir"} & set(Fine.get_parameter_names())
        assert "_hf_data_subdir" not in Fine.configure().model_dump()


class _BodyClipDataset(ClipDataset):
    """``ClipDataset`` whose clips carry a still MANO hand when ``body_model_params`` is on."""

    _supports_body_model_params: ClassVar[bool] = True

    def _read_single_sequence(self, path: str):
        data = np.load(path)
        frames = len(data["base_pos_w"])
        body = {}
        if self.body_model_params:
            pose = torch.zeros(frames, 15, 4)
            pose[..., 0] = 1.0
            body = dict(body_pose_quat=pose, body_shape=torch.ones(frames, 10), body_model="mano_right")
        return self._pack_motion_sequence(
            framerate=float(data["framerate"]),
            base_pos_w=torch.from_numpy(data["base_pos_w"]),
            base_quat_w=torch.from_numpy(data["base_quat_w"]),
            **body,
        )


class TestBodyModelParams:
    def test_off_by_default_and_retimed_with_the_clip_when_on(self, tmp_path, make_dataset):
        write_clip(tmp_path / "raw" / "a.npz", num_frames=9, framerate=30.0)
        plain = make_dataset(_BodyClipDataset, raw_data_path=str(tmp_path / "raw"))
        assert plain.motion_sequences["g1"].body_pose_quat is None

        loaded = make_dataset(_BodyClipDataset, raw_data_path=str(tmp_path / "raw"), body_model_params=True)
        seq = loaded.motion_sequences["g1"]
        assert seq.body_model == ("mano_right",)
        assert seq.body_pose_quat._data_shape == (15, 4)
        assert seq.body_pose_quat.concatenated_tensor.shape[0] == int(seq.buffer_length.sum()) > 9  # 30 -> 50 Hz

    def test_a_dataset_that_cannot_fill_them_refuses_the_flag(self, raw, make_dataset):
        with pytest.raises(ValueError, match=r"ClipDataset does not provide body-model parameters.*dex_ycb_dataset"):
            make_dataset(raw_data_path=str(raw), body_model_params=True)

    def test_a_cache_prepared_without_them_is_refused_rather_than_loaded_short(self, raw, tmp_path, make_dataset):
        cache = str(tmp_path / "cache.pt")
        make_dataset(_BodyClipDataset, raw_data_path=str(raw), processed_data_path=cache)
        with pytest.raises(ValueError, match="prepared without body-model parameters, but body_model_params=True"):
            make_dataset(_BodyClipDataset, raw_data_path=str(raw), processed_data_path=cache, body_model_params=True)

    def test_a_cache_prepared_with_them_loads_without_them_when_the_flag_is_off(self, raw, tmp_path, make_dataset):
        """The loaded data is what the flag says, so the modifiers that refuse the fields see none."""
        cache = str(tmp_path / "cache.pt")
        make_dataset(_BodyClipDataset, raw_data_path=str(raw), processed_data_path=cache, body_model_params=True)
        loaded = make_dataset(_BodyClipDataset, raw_data_path=str(raw), processed_data_path=cache).motion_sequences
        assert (
            loaded["g1"].body_pose_quat is None and loaded["g1"].body_shape is None and loaded["g1"].body_model is None
        )
        again = make_dataset(
            _BodyClipDataset, raw_data_path=str(raw), processed_data_path=cache, body_model_params=True
        )
        assert again.motion_sequences["g1"].body_model is not None  # the cache itself keeps them

    @pytest.mark.parametrize(
        ("body", "message"),
        [
            ({"body_model": "mano_right"}, "together; got neither tensor and body_model='mano_right'"),
            (
                {"body_pose_quat": torch.zeros(6, 15, 4), "body_shape": torch.zeros(6, 10), "body_model": "mano"},
                "Unknown",
            ),
            (
                {"body_pose_quat": torch.zeros(6, 23, 4), "body_shape": torch.zeros(6, 10), "body_model": "mano_left"},
                "has 23 joints but mano_left has 15",
            ),
            (
                {"body_pose_quat": torch.zeros(5, 15, 4), "body_shape": torch.zeros(6, 10), "body_model": "mano_left"},
                r"must be \(num_frames=6, num_body_joints, 4\)",
            ),
        ],
    )
    def test_pack_refuses_inconsistent_body_model_arguments(self, raw, make_dataset, body, message):
        dataset = make_dataset(raw_data_path=str(raw))
        quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(6, 1)
        with pytest.raises(ValueError, match=message):
            dataset._pack_motion_sequence(50.0, torch.zeros(6, 3), quat, **body)
