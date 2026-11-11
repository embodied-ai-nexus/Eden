"""Synthetic AssistMimic clips, and a real two-humanoid ``assistmimic`` env that plays them.

The task's published clips come from converted Inter-X / HHI-Assist pkls; these tests write their own instead:
every body at PHC's SMPL zero pose (identity rotations, positions from the MJCF's body offsets), so a reset has
a known pose to land on without any dataset.
"""

from __future__ import annotations

from xml.etree import ElementTree

import numpy as np
import pytest

from eden.tasks import TASK_REGISTRY
from eden.utils.assets import get_asset_path
from tests.support.envs import build_env

ROOT_HEIGHT = 0.95
#: Two clips: (frames, x shift of the whole clip).
CLIPS = ((10, 0.0), (12, 0.5))
#: Where the recipient's reference sits relative to the caregiver's.
RECIPIENT_SHIFT = (0.0, 1.0, 0.0)


def smpl_zero_pose() -> tuple[list[str], np.ndarray]:
    """Body names in MJCF order and their world positions ``(24, 3)`` at the zero pose, root at ``ROOT_HEIGHT``."""
    root = ElementTree.parse(get_asset_path("humanoids/smpl_humanoid/smpl_humanoid_phc.xml")).getroot()
    names, pos = [], []

    def walk(body, parent_pos):
        offset = np.array([float(x) for x in body.get("pos").split()])
        here = np.array([0.0, 0.0, ROOT_HEIGHT]) if parent_pos is None else parent_pos + offset
        names.append(body.get("name"))
        pos.append(here)
        for child in body.findall("body"):
            walk(child, here)

    walk(root.find("worldbody").find("body"), None)
    return names, np.stack(pos)


def write_clips(path, body_names=None, recipient_shift=RECIPIENT_SHIFT) -> str:
    """Write a two-clip pair ``.npz`` in ``scripts/convert_pair_motion.py``'s layout; return its path."""
    names, pos0 = smpl_zero_pose()
    arrays = {
        "clips": np.array([f"clip{i}" for i in range(len(CLIPS))]),
        "body_names": np.array(names if body_names is None else body_names),
        "num_frames": np.array([f for f, _ in CLIPS]),
        "fps": np.full(len(CLIPS), 30.0),
    }
    for role, shift in (("caregiver", np.zeros(3)), ("recipient", np.asarray(recipient_shift))):
        gts = np.concatenate([np.tile(pos0 + shift + [dx, 0.0, 0.0], (f, 1, 1)) for f, dx in CLIPS])
        rot = np.zeros((len(gts), 24, 4))
        rot[..., 0] = 1.0
        arrays |= {
            f"{role}_gts": gts.astype(np.float32),
            f"{role}_grs": rot.astype(np.float32),
            f"{role}_lrs": rot.astype(np.float32),
            f"{role}_gvs": np.zeros((len(gts), 24, 3), np.float32),
            f"{role}_gavs": np.zeros((len(gts), 24, 3), np.float32),
        }
    np.savez(path, **arrays)
    return str(path)


def assistmimic_config(task: str, motion_file: str, num_envs: int = 2):
    """The registered task, playing ``motion_file`` with ``num_envs`` envs."""
    cfg = TASK_REGISTRY.build(task)
    cfg.env_options.num_envs = num_envs
    cfg.command_options.motion.motion_file = motion_file
    return cfg


@pytest.fixture(scope="package")
def clips_file(tmp_path_factory):
    return write_clips(tmp_path_factory.mktemp("assistmimic") / "clips.npz")


@pytest.fixture(scope="package")
def assistmimic_env(clips_file):
    return build_env(assistmimic_config("assistmimic", clips_file), reset=True)
