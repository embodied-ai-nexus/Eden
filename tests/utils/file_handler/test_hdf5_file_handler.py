"""Tests for ``eden.utils.file_handler.hdf5_file_handler.HDF5FileHandler`` on a file written to ``tmp_path``."""

from __future__ import annotations

import h5py
import torch

from eden.utils.file_handler.episode_data import EpisodeData
from eden.utils.file_handler.hdf5_file_handler import HDF5FileHandler


def test_a_gap_in_the_demo_numbering_counts_what_is_there_and_the_next_write_skips_past_it(tmp_path):
    """A crash mid-rotation leaves ``demo_0`` and ``demo_2``; the count was ``max(index) + 1`` = 3."""
    file_path = str(tmp_path / "gappy.hdf5")
    with h5py.File(file_path, "w") as f:
        data = f.create_group("data")
        data.attrs["env_cfg"] = "{}"
        data.attrs["total"] = 2
        for idx in (0, 2):
            data.create_group(f"demo_{idx}").create_dataset("x", data=[1, 2, 3])

    handler = HDF5FileHandler()
    handler.open(file_path, mode="a")
    try:
        assert handler.demo_count == handler.get_num_episodes() == 2
        episode = EpisodeData()
        episode.add("x", torch.tensor([7, 8, 9]))
        handler.write_episode(episode)  # must not collide with demo_2
        assert sorted(handler.get_episode_names()) == ["demo_0", "demo_2", "demo_3"]
        assert handler.demo_count == 3
    finally:
        handler.close()
