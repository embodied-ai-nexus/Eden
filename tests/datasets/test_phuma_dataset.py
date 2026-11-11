"""Tests for ``eden.datasets.phuma_dataset.PHUMADataset``'s download layout.

Building the dataset runs Genesis FK, so these drive the unpack hook on an unbuilt instance over a real
``data.zip`` written to ``tmp_path``.
"""

from __future__ import annotations

import zipfile

from eden.datasets.phuma_dataset import PHUMADataset


def test_the_single_zip_is_unpacked_then_narrowed_to_the_g1_configuration(tmp_path):
    """Nothing to narrow at fetch time — one ``data.zip`` holds both humanoid configurations. A hook that narrowed
    as well would double-join to ``data/g1/data/g1``.
    """
    assert PHUMADataset._hf_allow_patterns is None
    root = tmp_path / "snapshot"
    root.mkdir()
    with zipfile.ZipFile(root / "data.zip", "w") as archive:
        archive.writestr("data/g1/walk.npy", "")
        archive.writestr("data/h1_2/walk.npy", "")

    dataset = object.__new__(PHUMADataset)

    assert dataset._post_download_process(str(root)) == str(root)
    assert dataset._resolve_data_subdir(str(root)) == str(root / "data" / "g1")


def test_a_finished_unzip_is_not_repeated(tmp_path):
    """PHUMA once re-extracted ``data.zip`` on every run; a marker now records a finished extraction."""
    root = tmp_path / "snapshot"
    root.mkdir()
    with zipfile.ZipFile(root / "data.zip", "w") as archive:
        archive.writestr("data/g1/walk.npy", "")
    dataset = object.__new__(PHUMADataset)
    dataset._post_download_process(str(root))
    (root / "data" / "g1" / "walk.npy").unlink()

    dataset._post_download_process(str(root))

    assert not (root / "data" / "g1" / "walk.npy").exists()
    assert (root / "data" / ".eden_extracted").exists()
