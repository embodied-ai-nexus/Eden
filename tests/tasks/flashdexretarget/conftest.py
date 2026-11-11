"""A small FlashDexRetarget dataset built through the real pipeline, and the task built on it.

Two clips of the XHand's own palm and fingertip trajectory (so the IK has an exact answer), each with a box of its
own size: two object slots, so the task builds a heterogeneous object and per-object clip groups.
"""

from __future__ import annotations

import numpy as np
import pytest

from eden.extensions.retargeting.flashdexretarget import clips as C
from eden.extensions.retargeting.flashdexretarget import run_kinematic_retargeting
from eden.options.extensions.flashdexretarget import FDRKinematicRetargetingOptions
from eden.tasks import TASK_REGISTRY
from tests.extensions.retargeting.flashdexretarget.conftest import self_fk_clip, write_box
from tests.support.envs import build_env

#: Object id -> box extents; one clip each.
BOXES = {"box_small": (0.06, 0.06, 0.06), "box_large": (0.1, 0.08, 0.12)}


@pytest.fixture(scope="session")
def motion_file(tmp_path_factory):
    """``<dataset>/xhand/motion.pt`` with two clips on two boxes, object URDFs and SDF grids."""
    pytest.importorskip("warp", reason="the SDF bake needs warp-lang (the flashdexretarget extra)")
    dataset = tmp_path_factory.mktemp("fdr_dataset")
    (dataset / "human_demo").mkdir()
    for k, (oid, extents) in enumerate(BOXES.items()):
        clip, _ = self_fk_clip(T=40, box_pos=(0.6, 0.6 + 0.3 * k, extents[2] / 2))
        clip["obj_right_id"] = np.array(oid)
        C.save_clip(dataset / "human_demo" / f"clip_{k}.npz", clip)
        write_box(dataset / "objects" / oid / "visual.obj", extents=extents)
    options = FDRKinematicRetargetingOptions(
        dataset=str(dataset),
        workers=1,
        filters={"too_short": {"min_sec": 0.1}},
        retarget={"attempts": 1, "n_guess": 4, "start_settle": 10, "approach": {"enable": False}},
        sdf={"device": "cpu", "grid_n": 128},
    )
    return run_kinematic_retargeting(options)


def build_task(motion_file, *, num_envs: int = 2, eval_mode: bool = False, **overrides):
    """The task on ``motion_file`` with ``num_envs`` envs, reset noise off, built and reset (``eval_mode``: as inference)."""
    cfg = TASK_REGISTRY.build("flashdexretarget", motion=str(motion_file))
    cfg.env_options.num_envs = num_envs
    cfg.event_options.reset_hands.noise_level = 0.0
    for path, value in overrides.items():
        obj, *attrs, last = path.split(".")
        target = getattr(cfg, obj)
        for a in attrs:
            target = getattr(target, a)
        setattr(target, last, value)
    return build_env(cfg, reset=True, eval_mode=eval_mode)
