"""Tests for ``eden.datasets.lafan1_dataset.LAFAN1RetargetedG1Dataset`` over a pre-seeded Hugging Face cache.

The ``lvhaidong/LAFAN1_Retargeting_Dataset`` snapshot ships ``g1/`` (29 DOFs), ``h1/`` (19) and ``h1_2/``
(27) side by side; a download that globbed the snapshot root mixed their CSV widths and crashed in
``torch.cat``. The cache stands in for the Hub, with the client held offline.
"""

from __future__ import annotations

import fnmatch

import huggingface_hub.constants
import numpy as np
import pytest
import torch

from eden.datasets.lafan1_dataset import LAFAN1RetargetedG1Dataset

REPO_FOLDER = "datasets--lvhaidong--LAFAN1_Retargeting_Dataset"
COMMIT = "0" * 40


def _write_csv(path, num_dofs, num_frames=5):
    """``[pos(3), quat_xyzw(4), dofs(num_dofs)]`` per row: standing still at the origin."""
    rows = np.zeros((num_frames, 7 + num_dofs))
    rows[:, 6] = 1.0
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, rows, delimiter=",")


@pytest.fixture
def hub_cache(tmp_path, monkeypatch):
    """Point the offline Hub client at an empty cache; returns the snapshot directory to fill."""
    cache = tmp_path / "hub"
    snapshot = cache / REPO_FOLDER / "snapshots" / COMMIT
    snapshot.mkdir(parents=True)
    (cache / REPO_FOLDER / "refs").mkdir()
    (cache / REPO_FOLDER / "refs" / "main").write_text(COMMIT)
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_CACHE", str(cache))
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_OFFLINE", True)
    return snapshot


@pytest.mark.requires_genesis
def test_a_download_reads_only_the_g1_retargets(hub_cache):
    for robot, num_dofs in (("g1", 29), ("h1", 19), ("h1_2", 27)):
        _write_csv(hub_cache / robot / "walk1_subject1.csv", num_dofs)
    (hub_cache / "robot_description").mkdir()

    dataset = LAFAN1RetargetedG1Dataset(options=LAFAN1RetargetedG1Dataset.configure(), device=torch.device("cpu"))

    assert dataset.raw_data_path == str(hub_cache / "g1")
    assert dataset.metadata["all_motion_files"] == ["walk1_subject1.csv"]
    assert len(dataset) == 1
    assert dataset.motion_sequences["g1"].dofs_pos.shape[1:] == (29,)  # g1 width, retimed 30 -> 50 fps


def test_a_snapshot_without_the_g1_subdirectory_names_the_cause(hub_cache):
    """Rather than the downstream shape mismatch, if the Hub layout ever changes."""
    _write_csv(hub_cache / "h1" / "walk1_subject1.csv", 19)

    with pytest.raises(FileNotFoundError, match=r"expects 'g1' under .*there is no 'g1' there.*h1"):
        LAFAN1RetargetedG1Dataset(options=LAFAN1RetargetedG1Dataset.configure(), device=torch.device("cpu"))


def test_the_fetch_skips_the_other_robots_and_the_urdfs():
    """Narrowed at fetch time, not after: the h1 / h1_2 retargets and the ``robot_description`` tree of URDFs and
    meshes are ~2.5x the g1 CSVs this loader reads (87 of the repo's 314 MB).
    """
    layout = (
        "g1/walk1_subject1.csv",
        "h1/walk1_subject1.csv",
        "h1_2/walk1_subject1.csv",
        "robot_description/g1/g1_29dof_rev_1_0.urdf",
        "meta_data/lafan1.json",
        "LICENSE",
        "README.md",
    )
    patterns = LAFAN1RetargetedG1Dataset._hf_allow_patterns
    assert [f for f in layout if any(fnmatch.fnmatch(f, p) for p in patterns)] == [
        "g1/walk1_subject1.csv",
        "LICENSE",
        "README.md",
    ]
