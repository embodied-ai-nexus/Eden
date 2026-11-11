"""The backends against their real models: the contract the fakes stand in for.

Each test skips unless its package is installed, a CUDA device is visible and the model files are
on disk — they never download. GraspGenX reads its release checkpoints and gripper descriptions from
the local Hugging Face cache; GraspGen reads ``$GRASPGEN_CHECKPOINT_DIR`` (``grasp_gen download``).
The Eden env stays on the suite's ``gs.cpu``.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from eden.extensions.grasp_sampling import GraspSamplerBase
from eden.extensions.grasp_sampling.graspgen import GraspGenSampler
from eden.extensions.grasp_sampling.graspgenx import GraspGenXSampler
from eden.utils.geom import transform_by_quat
from tests.extensions.grasp_sampling.conftest import BOX_CENTER, BOX_HALF, cached_descriptions

pytestmark = pytest.mark.slow


def require(package: str):
    if importlib.util.find_spec(package) is None:
        pytest.skip(f"{package} is not installed")
    if not torch.cuda.is_available():
        pytest.skip("the model runs on CUDA only")


def check_grasps_close_on_the_box(sampler: GraspSamplerBase, results):
    for result in results:
        assert result.success
        confidences = result.confidences.cpu()
        assert (confidences[:-1] >= confidences[1:]).all() and (confidences >= sampler.grasp_threshold).all()
        best = slice(0, 10)
        tcp = result.pos[best] + transform_by_quat(
            sampler.tcp_offset.expand(len(result.pos[best]), 3), result.quat[best]
        )
        # The TCP lands on the 6 cm cube: within its half-diagonal of its center.
        distance = np.linalg.norm(tcp.cpu().numpy() - np.asarray(BOX_CENTER), axis=1)
        assert (distance < np.sqrt(3) * BOX_HALF + 0.01).all(), np.round(distance, 3)


def test_graspgenx_samples_the_box_one_env_and_batched(grasp_env, monkeypatch):
    require("graspgenx")
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    descriptions = Path(cached_descriptions()).parents[2]  # the bundle root, which holds gripper_descriptions/
    try:
        checkpoints = Path(snapshot_download("adithyamurali/GraspGenXModel", local_files_only=True)) / "release"
    except LocalEntryNotFoundError:
        pytest.skip("GraspGenX's checkpoints are not in the local Hugging Face cache")
    monkeypatch.setenv("GRASPGENX_GRIPPER_CFG_DIR", str(descriptions))

    sampler = GraspGenXSampler(
        grasp_env,
        GraspGenXSampler.configure(
            end_effector="FrankaHand", checkpoint_dir=str(checkpoints), object_cloud="surface", num_grasps=100
        ),
    )
    check_grasps_close_on_the_box(sampler, [sampler.sample_grasps("box")])
    check_grasps_close_on_the_box(sampler, sampler.sample_grasps_batch("box", [0, 0]))


def test_graspgen_samples_the_box_and_filters_against_the_table(grasp_env):
    require("grasp_gen")
    checkpoint_dir = os.environ.get("GRASPGEN_CHECKPOINT_DIR", "")
    if not (Path(checkpoint_dir) / "graspgen_franka_panda_gen.pth").is_file():
        pytest.skip("GraspGen's franka_panda model is not under $GRASPGEN_CHECKPOINT_DIR")

    sampler = GraspGenSampler(
        grasp_env, GraspGenSampler.configure(end_effector="FrankaHand", object_cloud="surface", filter_collisions=True)
    )
    result = sampler.sample_grasps("box")
    check_grasps_close_on_the_box(sampler, [result])
    # The surface cloud includes the face on the table; the filter rejects the grasps from below it.
    approach = transform_by_quat(sampler.approach_axis.expand(len(result), 3), result.quat)
    assert (approach[:, 2] < 0.5).all()
