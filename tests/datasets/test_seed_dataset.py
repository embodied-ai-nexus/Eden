"""Tests for ``eden.datasets.seed_dataset.SEEDRetargetedG1Dataset``'s download layout and archive handling.

Building the dataset reads SEED CSVs through Genesis FK, so these drive the unpack hook and the explicit-path
tolerance on an unbuilt instance, over a real ``g1.tar.gz`` written to ``tmp_path``.
"""

from __future__ import annotations

import os
import tarfile

import pytest

from eden.datasets.seed_dataset import SEEDRetargetedG1Dataset

CLIPS = ("g1/csv/2024-01-01/walk.csv", "g1/csv/2024-01-01/run.csv")


@pytest.fixture
def archive_dir(tmp_path):
    """A directory holding ``g1.tar.gz`` the way the repo ships it: ``g1/csv/{date}/*.csv``."""
    source = tmp_path / "source"
    for rel in CLIPS:
        (source / rel).parent.mkdir(parents=True, exist_ok=True)
        (source / rel).write_text("0\n")
    root = tmp_path / "snapshot"
    root.mkdir()
    with tarfile.open(root / "g1.tar.gz", "w:gz") as tar:
        tar.add(source / "g1", arcname="g1")
    return root


def _seed() -> SEEDRetargetedG1Dataset:
    return object.__new__(SEEDRetargetedG1Dataset)


def test_only_the_g1_split_is_fetched():
    """The repo is ~114 GB; the G1 split this loader reads is 23.5 GB of it."""
    assert "g1.tar.gz" in SEEDRetargetedG1Dataset._hf_allow_patterns
    assert not any("soma" in p for p in SEEDRetargetedG1Dataset._hf_allow_patterns)
    assert SEEDRetargetedG1Dataset._hf_data_subdir == "g1/csv"


def test_unpacking_returns_the_root_for_the_base_to_narrow(archive_dir):
    """A hook that narrowed as well would double-join to ``g1/csv/g1/csv``."""
    dataset = _seed()
    assert dataset._post_download_process(str(archive_dir)) == str(archive_dir)
    assert dataset._resolve_data_subdir(str(archive_dir)) == str(archive_dir / "g1" / "csv")
    assert (archive_dir / CLIPS[1]).exists()


def test_an_untar_interrupted_after_the_csv_tree_appeared_is_redone(archive_dir):
    """Directory entries precede their files, so ``g1/csv/`` exists moments into an untar: only a marker written
    after ``extractall`` returns can tell a finished tree from a partial one.
    """
    (archive_dir / "g1" / "csv" / "2024-01-01").mkdir(parents=True)
    (archive_dir / CLIPS[0]).write_text("0\n")  # the untar stopped here

    _seed()._post_download_process(str(archive_dir))

    assert (archive_dir / CLIPS[1]).exists()


def test_a_finished_untar_is_not_repeated(archive_dir):
    dataset = _seed()
    dataset._post_download_process(str(archive_dir))
    os.remove(archive_dir / CLIPS[1])

    dataset._post_download_process(str(archive_dir))

    assert not (archive_dir / CLIPS[1]).exists()


def test_deleting_the_extracted_tree_re_arms_the_untar(archive_dir):
    """What the missing-subdirectory error tells the caller to do; the marker goes with the tree."""
    import shutil

    dataset = _seed()
    dataset._post_download_process(str(archive_dir))
    shutil.rmtree(archive_dir / "g1")

    dataset._post_download_process(str(archive_dir))

    assert (archive_dir / CLIPS[1]).exists()


@pytest.mark.parametrize("pointed_at", ["", "g1", "g1/csv"])
def test_an_explicit_raw_data_path_may_be_the_archive_dir_the_extracted_root_or_the_csv_tree(archive_dir, pointed_at):
    """Explicit-path tolerance for a manual download of the gated repo — not expressible as one subdirectory."""
    if pointed_at:
        _seed()._post_download_process(str(archive_dir))
    found = _seed()._get_all_motion_files(str(archive_dir / pointed_at), ".csv")
    assert sorted(os.path.basename(f) for f in found) == ["run.csv", "walk.csv"]


@pytest.fixture
def read_only(archive_dir):
    """Make directories read-only for the test, as on a shared mirror; restored so ``tmp_path`` can be removed."""
    if os.geteuid() == 0:
        pytest.skip("root writes through directory permissions")
    locked: list = []

    def lock(*paths):
        for path in paths:
            os.chmod(path, 0o555)
            locked.append(path)

    yield lock
    for path in locked:
        os.chmod(path, 0o755)


class TestReadOnlyCopy:
    """A tree extracted without the marker cannot be re-extracted where it is, so the untar is refused by name."""

    def test_an_unmarked_tree_names_the_csv_path_that_needs_no_untar(self, archive_dir, read_only):
        with tarfile.open(archive_dir / "g1.tar.gz") as tar:
            tar.extractall(archive_dir)  # by hand, before the marker existed
        read_only(archive_dir / "g1", archive_dir)

        with pytest.raises(PermissionError, match=r"no '\.eden_extracted' marker.*raw_data_path='[^']*/g1/csv'"):
            _seed()._post_download_process(str(archive_dir))

    def test_the_named_path_reads_the_tree_as_is(self, archive_dir, read_only):
        with tarfile.open(archive_dir / "g1.tar.gz") as tar:
            tar.extractall(archive_dir)
        read_only(archive_dir / "g1", archive_dir)

        found = _seed()._get_all_motion_files(str(archive_dir / "g1" / "csv"), ".csv")

        assert sorted(os.path.basename(f) for f in found) == ["run.csv", "walk.csv"]

    def test_a_marked_tree_is_read_without_writing(self, archive_dir, read_only):
        _seed()._post_download_process(str(archive_dir))
        read_only(archive_dir / "g1", archive_dir)

        found = _seed()._get_all_motion_files(str(archive_dir), ".csv")

        assert sorted(os.path.basename(f) for f in found) == ["run.csv", "walk.csv"]

    def test_an_archive_alone_says_where_to_extract_it(self, archive_dir, read_only):
        read_only(archive_dir)

        with pytest.raises(PermissionError, match=r"cannot extract g1\.tar\.gz into .*not writable"):
            _seed()._post_download_process(str(archive_dir))
