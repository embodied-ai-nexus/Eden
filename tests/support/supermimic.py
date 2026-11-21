"""A small LAFAN1-format library on disk and the supermimic task built over it.

The ``supermimic_library`` and ``supermimic_env`` fixtures are registered for the whole suite by ``tests/conftest.py``
(``pytest_plugins``), so the task's tests and the tests of the learner that ships its motion term share one build.
"""

from __future__ import annotations

from fnmatch import fnmatchcase
from pathlib import Path

import numpy as np
import pytest

#: Frames of each clip at the LAFAN1 rate (30 fps).
CLIP_FRAMES = (90, 45)
NUM_ENVS = 4


# The task and the motion support are imported inside the functions: this module is a pytest plugin of the whole
# suite (``tests/conftest.py``), so a module-scope import would load the task for every test run.
def _knees_bent() -> np.ndarray:
    from eden.options.robots.unitree_g1 import G1_DOFS29_NAME
    from eden.tasks.supermimic.config import KNEES_BENT_DOFS_POS

    return np.array(
        [next((v for p, v in KNEES_BENT_DOFS_POS.items() if fnmatchcase(n, p)), 0.0) for n in G1_DOFS29_NAME]
    )


def write_lafan1_library(root: Path) -> Path:
    """LAFAN1-layout clips (:func:`tests.support.motion.write_lafan1_csv_library`) swaying about the bent-knee pose."""
    from tests.support.motion import write_lafan1_csv_library

    return write_lafan1_csv_library(root, list(CLIP_FRAMES), joints_center=_knees_bent())


def supermimic_config(library: Path, *, rsi_noise: bool = True):
    from eden.tasks import TASK_REGISTRY

    cfg = TASK_REGISTRY.build("supermimic", raw_data_path=str(library))
    cfg.env_options.num_envs = NUM_ENVS
    if not rsi_noise:
        rsi = cfg.event_options.reset_from_motion
        rsi.pose_range, rsi.velocity_range, rsi.dofs_position_range = {}, {}, (0.0, 0.0)
    return cfg


@pytest.fixture(scope="session")
def supermimic_library(tmp_path_factory) -> Path:
    return write_lafan1_library(tmp_path_factory.mktemp("lafan1"))


@pytest.fixture(scope="session")
def supermimic_env(supermimic_library):
    """The task over :func:`write_lafan1_library`, resets placing the robot exactly on the reference."""
    from tests.support.envs import build_env

    return build_env(supermimic_config(supermimic_library, rsi_noise=False), reset=True)
