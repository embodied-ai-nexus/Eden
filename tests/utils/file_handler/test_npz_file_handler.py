"""Tests for ``eden.utils.file_handler.npz_file_handler.NPZFileHandler`` on a file written to ``tmp_path``."""

from __future__ import annotations

import json

import numpy as np
import torch

from eden.utils.file_handler.episode_data import EpisodeData
from eden.utils.file_handler.npz_file_handler import NPZFileHandler


def test_a_gap_in_the_demo_numbering_counts_what_is_there_and_the_next_write_skips_past_it(tmp_path):
    """A crash mid-rotation leaves ``demo_0`` and ``demo_2``; the count was ``max(index) + 1`` = 3."""
    file_path = str(tmp_path / "gappy.npz")
    np.savez_compressed(
        file_path,
        **{
            "__meta__total": np.array(2),
            "__meta__env_cfg": np.frombuffer(json.dumps({}).encode("utf-8"), dtype=np.uint8),
            "demo_0/x": np.array([1, 2, 3]),
            "demo_2/x": np.array([4, 5, 6]),
        },
    )

    handler = NPZFileHandler()
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
