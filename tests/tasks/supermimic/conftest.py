"""A small LAFAN1-format library on disk and the supermimic task built over it."""

from __future__ import annotations

from fnmatch import fnmatchcase
from pathlib import Path

import numpy as np
import pytest

from eden.options.robots.unitree_g1 import G1_DOFS29_NAME
from eden.tasks import TASK_REGISTRY
from eden.tasks.supermimic.config import KNEES_BENT_DOFS_POS

#: Frames of each clip at the LAFAN1 rate (30 fps).
CLIP_FRAMES = (90, 45)
NUM_ENVS = 4


def _knees_bent() -> np.ndarray:
    return np.array(
        [next((v for p, v in KNEES_BENT_DOFS_POS.items() if fnmatchcase(n, p)), 0.0) for n in G1_DOFS29_NAME]
    )


def write_lafan1_library(root: Path) -> Path:
    """Clips in the LAFAN1 G1 CSV layout ``[pos(3), quat xyzw(4), joints(29)]``: the robot walks forward at 0.3 m/s
    while its joints sway about the bent-knee pose.
    """
    root.mkdir(parents=True, exist_ok=True)
    for clip, frames in enumerate(CLIP_FRAMES):
        t = np.arange(frames) / 30.0
        pos = np.stack([0.3 * t, np.full(frames, 0.1 * clip), np.full(frames, 0.76)], axis=-1)
        yaw = 0.2 * t
        quat_xyzw = np.stack([np.zeros(frames), np.zeros(frames), np.sin(yaw / 2), np.cos(yaw / 2)], axis=-1)
        joints = _knees_bent()[None] + 0.1 * np.sin(2.0 * t[:, None] + np.arange(29)[None])
        np.savetxt(root / f"clip_{clip}.csv", np.concatenate([pos, quat_xyzw, joints], axis=-1), delimiter=",")
    return root


def supermimic_config(library: Path, *, rsi_noise: bool = True, **runner):
    cfg = TASK_REGISTRY.build("supermimic", raw_data_path=str(library))
    cfg.env_options.num_envs = NUM_ENVS
    if not rsi_noise:
        rsi = cfg.event_options.reset_from_motion
        rsi.pose_range, rsi.velocity_range, rsi.dofs_position_range = {}, {}, (0.0, 0.0)
    return cfg


@pytest.fixture(scope="session")
def lafan1_library(tmp_path_factory) -> Path:
    return write_lafan1_library(tmp_path_factory.mktemp("lafan1"))


@pytest.fixture(scope="session")
def env(lafan1_library):
    """The task over :func:`write_lafan1_library`, resets placing the robot exactly on the reference."""
    from tests.support.envs import build_env

    return build_env(supermimic_config(lafan1_library, rsi_noise=False), reset=True)
