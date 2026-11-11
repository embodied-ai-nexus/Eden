"""Tests for ``eden.extensions.grasp_sampling.graspgen.sampler.GraspGenSampler``.

``grasp_gen`` is the fake package of ``tests/extensions/grasp_sampling/conftest.py``; it records what
the collision filter was handed. The gripper configs are real files under ``tmp_path``.
"""

from __future__ import annotations

import math
import sys
from typing import get_args

import numpy as np
import pytest
import torch

import eden as en
from eden.extensions.grasp_sampling.graspgen import GraspGenEndEffector, GraspGenSampler
from tests.extensions.grasp_sampling.conftest import GRASPGEN_CONFIG as CONFIG
from tests.extensions.grasp_sampling.conftest import GRASPGEN_FRANKA_CONFIG, GRASPGEN_FRANKA_DEPTH


def test_grasps_come_back_ranked_in_the_grasp_frame(grasp_env, grasp_gen):
    sampler = GraspGenSampler(grasp_env, GraspGenSampler.configure(gripper_config=CONFIG, object_cloud="surface"))
    result = sampler.sample_grasps("box")
    assert sampler.pose_link_name is None  # the grasp frame is no link
    torch.testing.assert_close(result.confidences.cpu(), torch.tensor([0.95, 0.88, 0.81]))
    torch.testing.assert_close(result.pos[:, 0].cpu(), torch.tensor([0.2, 0.3, 0.1]))
    torch.testing.assert_close(result.quat.cpu(), torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(3, 1))


def test_filter_collisions_checks_the_gripper_mesh_against_the_obstacle_cloud(grasp_env, grasp_gen):
    grasp_gen.keep = np.array([True, False, True])
    sampler = GraspGenSampler(
        grasp_env, GraspGenSampler.configure(gripper_config=CONFIG, filter_collisions=True, collision_threshold=0.01)
    )
    result = sampler.sample_grasps("box")

    (call,) = grasp_gen.filter_calls
    assert call["gripper_collision_mesh"] == "toy mesh"
    assert call["collision_threshold"] == 0.01
    assert (call["grasp_poses"][:, 3, 3] == 1.0).all()  # GraspGen leaves this corner 0; the filter needs it
    assert len(call["scene_pc"]) > 50
    torch.testing.assert_close(result.pos[:, 0].cpu(), torch.tensor([0.3, 0.1]))


def test_gripper_config_resolves_against_the_checkpoint_dir_and_lists_it_on_a_miss(grasp_env, grasp_gen):
    with pytest.raises(
        FileNotFoundError,
        match=r"'graspgen_franka.yml' is not under .*\['graspgen_franka_panda.yml', 'graspgen_toy.yml'\]",
    ):
        GraspGenSampler(grasp_env, GraspGenSampler.configure(gripper_config="graspgen_franka.yml"))


def test_an_unset_checkpoint_dir_says_how_to_get_one(grasp_env, grasp_gen, monkeypatch):
    monkeypatch.delenv("GRASPGEN_CHECKPOINT_DIR")
    with pytest.raises(FileNotFoundError, match=r"GRASPGEN_CHECKPOINT_DIR.*grasp_gen download"):
        GraspGenSampler(grasp_env, GraspGenSampler.configure(gripper_config=CONFIG))


def test_a_missing_install_names_the_package(grasp_env, monkeypatch):
    monkeypatch.setitem(sys.modules, "grasp_gen", None)
    with pytest.raises(ImportError, match=r'pip install "eden-world\[graspgen\]".*grasp_gen download'):
        GraspGenSampler(grasp_env, GraspGenSampler.configure(gripper_config=CONFIG))


def test_the_end_effector_literal_lists_every_pairing_and_each_names_an_eden_robot():
    assert set(get_args(GraspGenEndEffector)) == set(GraspGenSampler.END_EFFECTORS)
    assert all(hasattr(en.robots, name) for name in GraspGenSampler.END_EFFECTORS)


def test_end_effector_places_the_franka_hand_a_quarter_turn_from_the_grasp_frame(grasp_env, grasp_gen):
    grasp_gen.keep = np.ones(3, dtype=bool)
    sampler = GraspGenSampler(grasp_env, GraspGenSampler.configure(end_effector="FrankaHand", filter_collisions=True))
    result = sampler.sample_grasps("box")

    assert sampler.gripper_config == GRASPGEN_FRANKA_CONFIG and sampler.pose_link_name == "hand"
    # The collision filter checks GraspGen's own grasp frames, which its gripper mesh is posed in.
    (call,) = grasp_gen.filter_calls
    np.testing.assert_allclose(call["grasp_poses"][:, :3, :3], np.eye(3)[None].repeat(3, axis=0))
    torch.testing.assert_close(result.pos[:, 0].cpu(), torch.tensor([0.2, 0.3, 0.1]))
    half = math.sqrt(0.5)
    torch.testing.assert_close(result.quat.cpu(), torch.tensor([[half, 0.0, 0.0, half]]).repeat(3, 1))
    # In the hand's frame the fingers close along y, and the TCP is GraspGen's depth along z.
    torch.testing.assert_close(sampler.closing_axis.cpu(), torch.tensor([0.0, -1.0, 0.0]), atol=1e-6, rtol=0)
    torch.testing.assert_close(sampler.approach_axis.cpu(), torch.tensor([0.0, 0.0, 1.0]), atol=1e-6, rtol=0)
    torch.testing.assert_close(sampler.tcp_offset.cpu(), torch.tensor([0.0, 0.0, GRASPGEN_FRANKA_DEPTH]))


def test_an_end_effector_and_a_different_gripper_config_are_refused(grasp_env, grasp_gen):
    with pytest.raises(ValueError, match=r"pairs with gripper_config='graspgen_franka_panda.yml', but gripper_config="):
        GraspGenSampler(grasp_env, GraspGenSampler.configure(end_effector="FrankaHand", gripper_config=CONFIG))
