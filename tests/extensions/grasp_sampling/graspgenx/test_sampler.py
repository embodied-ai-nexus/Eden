"""Tests for ``eden.extensions.grasp_sampling.graspgenx.sampler.GraspGenXSampler``.

``graspgenx`` is the fake package of ``tests/extensions/grasp_sampling/conftest.py``; it records what
the planner was handed. The gripper description and checkpoint directories are real files under
``tmp_path``, and the URDF is parsed by the real ``yourdfpy``.
"""

from __future__ import annotations

import json
import math
import os
import sys
from typing import get_args

import numpy as np
import pytest
import torch
from pydantic import ValidationError

import eden as en
from eden.extensions.grasp_sampling.graspgenx import GraspGenXEndEffector, GraspGenXGripper, GraspGenXSampler
from eden.options.extensions.grasp_sampling import GraspMoEOptions
from eden.utils.geom import trans_quat_to_T
from tests.extensions.grasp_sampling.conftest import (
    BOX_CENTER,
    CLOSE,
    GRIPPER,
    OPEN,
    cached_descriptions,
    write_checkpoints,
    write_description,
)


def make_sampler(env, **params) -> GraspGenXSampler:
    params = {"gripper_name": GRIPPER, "object_cloud": "surface", **params}
    return GraspGenXSampler(env, GraspGenXSampler.configure(**params))


def test_poses_place_the_description_base_link(grasp_env, graspgenx):
    # The description welds its base link 1 cm along the grasp frame's approach axis, yawed 90 degrees.
    # The grasp frame itself is rolled 90 degrees about world x, so that approach axis is world -y.
    graspgenx.grasp_frames = np.eye(4)[None].copy()
    graspgenx.grasp_frames[0, :3, :3] = [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]
    graspgenx.grasp_frames[0, :3, 3] = (0.5, 0.0, 0.4)
    sampler = make_sampler(grasp_env)
    result = sampler.sample_grasps("box")

    assert sampler.pose_link_name == "toy_base"
    torch.testing.assert_close(result.pos.cpu(), torch.tensor([[0.5, -0.01, 0.4]]))
    # roll(x, 90) then yaw(z, 90), composed: (0.5, 0.5, -0.5, 0.5) in wxyz.
    torch.testing.assert_close(result.quat.cpu(), torch.tensor([[0.5, 0.5, -0.5, 0.5]]), atol=1e-4, rtol=0)


def test_approach_axis_is_the_grasp_z_axis_in_the_base_link_frame(tmp_path, grasp_env, graspgenx):
    # A dexterous hand's palm is pitched against the grasp frame, as sharpa_wave's is by -0.85 rad.
    write_description(tmp_path / "descriptions", xyz="0.037 0 0.015", rpy="0 -0.85 0")
    torch.testing.assert_close(
        make_sampler(grasp_env).approach_axis.cpu(), torch.tensor([math.sin(0.85), 0.0, math.cos(0.85)])
    )


def test_closing_axis_is_the_grasp_x_axis_in_the_base_link_frame(grasp_env, graspgenx):
    # The base link is yawed 90 degrees from the grasp frame, so grasp x reads as base -y.
    torch.testing.assert_close(
        make_sampler(grasp_env).closing_axis.cpu(), torch.tensor([0.0, -1.0, 0.0]), atol=1e-4, rtol=0
    )


def test_tcp_offset_is_the_fingertip_center_in_the_base_link_frame(grasp_env, graspgenx):
    # The fingertip is 10 cm along the grasp frame's approach; the base link sits 1 cm along it, yawed 90 degrees.
    torch.testing.assert_close(make_sampler(grasp_env).tcp_offset.cpu(), torch.tensor([0.0, 0.0, 0.09]))


def test_open_and_close_configurations_come_from_the_description(grasp_env, graspgenx):
    sampler = make_sampler(grasp_env)
    assert sampler.open_dofs_pos == OPEN
    assert sampler.close_dofs_pos == CLOSE


def test_the_planner_gets_the_world_frame_cloud_and_every_knob(grasp_env, graspgenx):
    moe = GraspMoEOptions(
        num_yaws=12,
        z_offsets_cm=(-4.0,),
        outlier_threshold=0.02,
        outlier_k=10,
        obb_mode="pca",
        skip_obb_rule="never",
        obb_density="sparse",
        obb_position_spacing_cm=2.0,
    )
    sampler = make_sampler(
        grasp_env,
        planner="graspmoe",
        grasp_threshold=0.65,
        num_grasps=64,
        max_grasps=3,
        graspmoe=moe,
    )
    sampler.sample_grasps("box")

    (call,) = graspgenx.planner_calls
    np.testing.assert_allclose(call["points"].mean(axis=0), BOX_CENTER, atol=0.005)
    assert call["planner"] == "graspmoe"
    assert call["grasp_threshold"] == 0.65 and call["num_grasps"] == 64
    assert call["topk_num_grasps"] == -1  # max_grasps is applied after filtering, by the base class
    assert {key[len("moe_") :]: value for key, value in call.items() if key.startswith("moe_")} == moe.model_dump(
        exclude={"name", "class_name"}
    )


def test_graspmoe_survives_a_saved_config_round_trip(grasp_env, graspgenx):
    options = GraspGenXSampler.configure(
        gripper_name=GRIPPER, object_cloud="surface", graspmoe=GraspMoEOptions(num_yaws=8)
    )
    reloaded = type(options).model_validate(json.loads(json.dumps(options.model_dump())))
    GraspGenXSampler(grasp_env, reloaded).sample_grasps("box")
    assert graspgenx.planner_calls[0]["moe_num_yaws"] == 8


def test_the_release_checkpoints_load_by_default_and_checkpoint_dir_overrides_them(tmp_path, grasp_env, graspgenx):
    make_sampler(grasp_env)
    own = write_checkpoints(tmp_path / "own")
    make_sampler(grasp_env, checkpoint_dir=str(own))
    assert [load[0] for load in graspgenx.loaded] == [str(tmp_path / "release" / "gen"), str(own / "gen")]
    assert {load[2] for load in graspgenx.loaded} == {GRIPPER}


def test_an_unknown_gripper_names_the_descriptions_there_are(grasp_env, graspgenx):
    with pytest.raises(ValueError, match=r"gripper_name='FrankaEmikaPanda' is not a GraspGenX .*\['toy_gripper'\]"):
        make_sampler(grasp_env, gripper_name="FrankaEmikaPanda")


def test_filter_collisions_is_refused_with_the_simulation_check_to_use_instead(grasp_env, graspgenx):
    with pytest.raises(
        ValueError, match=r"GraspGenXSampler cannot filter_collisions: .*coll_mesh\.obj.*detect_collision"
    ):
        make_sampler(grasp_env, filter_collisions=True)
    assert graspgenx.loaded == []


def test_an_unknown_planner_lists_the_planners(grasp_env, graspgenx):
    with pytest.raises(ValueError, match=r"planner='moe' is not a GraspGenX planner.*'graspmoe', 'diffusion'"):
        make_sampler(grasp_env, planner="moe")


def test_a_description_that_does_not_root_at_one_fixed_joint_is_refused(tmp_path, grasp_env, graspgenx):
    write_description(tmp_path / "descriptions")
    urdf = tmp_path / "descriptions" / "x_grippers" / GRIPPER / "gripper.urdf"
    urdf.write_text(
        urdf.read_text()
        .replace('type="fixed"', 'type="revolute"')
        .replace("<origin", '<axis xyz="0 0 1"/><limit lower="0" upper="1" effort="1" velocity="1"/><origin')
    )
    with pytest.raises(ValueError, match=r"does not weld a single base link.*\('world_joint', 'revolute'\)"):
        make_sampler(grasp_env)


def test_a_missing_gripper_name_is_refused_before_any_download(grasp_env, graspgenx):
    graspgenx.descriptions = None  # a lookup would raise the download error instead
    with pytest.raises(ValueError, match=r"needs gripper_name=.*'franka_panda'"):
        make_sampler(grasp_env, gripper_name=None)


def test_the_gripper_literal_lists_every_description_graspgenx_ships():
    """Against the pinned dataset revision when it is in the local Hugging Face cache; the suite needs no network."""
    assets = cached_descriptions()
    shipped = {name for name in os.listdir(assets) if os.path.isfile(os.path.join(assets, name, "config.json"))}
    assert set(get_args(GraspGenXGripper)) == shipped


def test_the_end_effector_literal_lists_every_pairing_and_each_names_an_eden_robot():
    assert set(get_args(GraspGenXEndEffector)) == set(GraspGenXSampler.END_EFFECTORS)
    assert all(hasattr(en.robots, name) for name in GraspGenXSampler.END_EFFECTORS)
    assert {pairing.gripper for pairing in GraspGenXSampler.END_EFFECTORS.values()} <= set(get_args(GraspGenXGripper))


def test_end_effector_picks_the_description_and_places_the_eden_link(tmp_path, grasp_env, graspgenx):
    # The default toy joint: the base link 1 cm along the approach axis, yawed 90 degrees.
    write_description(tmp_path / "descriptions", name="inspire_hand")
    sampler = make_sampler(grasp_env, gripper_name=None, end_effector="InspireRH56_R")
    result = sampler.sample_grasps("box")  # the planner returns the identity grasp frame

    assert graspgenx.loaded[0][2] == "inspire_hand"
    assert sampler.gripper_name == "inspire_hand" and sampler.pose_link_name == "base"
    # The Eden root is the description's base link, then the pairing's half turn.
    base = np.eye(4)
    base[:3, :3] = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    base[2, 3] = 0.01
    pos, quat = (torch.tensor(v, dtype=torch.float64) for v in sampler.END_EFFECTORS["InspireRH56_R"].link_in_frame)
    pose = trans_quat_to_T(result.pos[0].double().cpu(), result.quat[0].double().cpu()).numpy()
    np.testing.assert_allclose(pose, base @ trans_quat_to_T(pos, quat).numpy(), atol=1e-5)
    # In the Eden root's frame, the axes still name the grasp's z and x.
    np.testing.assert_allclose(pose[:3, :3] @ sampler.approach_axis.double().cpu().numpy(), (0, 0, 1), atol=1e-5)
    np.testing.assert_allclose(pose[:3, :3] @ sampler.closing_axis.double().cpu().numpy(), (1, 0, 0), atol=1e-5)


def test_an_end_effector_and_a_different_gripper_name_are_refused(grasp_env, graspgenx):
    with pytest.raises(
        ValueError, match=r"end_effector='FrankaHand' pairs with gripper_name='franka_panda', but gripper_name="
    ):
        make_sampler(grasp_env, end_effector="FrankaHand")


def test_an_unknown_end_effector_lists_the_pairings(grasp_env, graspgenx):
    with pytest.raises(ValueError, match=r"end_effector='Panda' is not an Eden end-effector.*'FrankaEmikaPanda'"):
        make_sampler(grasp_env, gripper_name=None, end_effector="Panda")


def test_several_envs_run_through_one_batched_planner_call(grasp_env, graspgenx):
    sampler = make_sampler(grasp_env, num_grasps=32)
    results = sampler.sample_grasps_batch("box", [0, 0])

    assert graspgenx.planner_calls == []
    (call,) = graspgenx.batch_calls
    assert len(call["clouds"]) == 2 and call["num_grasps"] == 32
    assert [len(result) for result in results] == [1, 1]
    torch.testing.assert_close(results[0].pos, results[1].pos)
    # One env goes through the single-object planner.
    sampler.sample_grasps("box")
    assert len(graspgenx.planner_calls) == 1 and len(graspgenx.batch_calls) == 1


def test_a_typo_in_a_graspmoe_knob_is_refused(grasp_env, graspgenx):
    with pytest.raises(ValidationError, match="num_yaw"):
        make_sampler(grasp_env, graspmoe={"num_yaw": 8})


def test_a_checkpoint_dir_without_gen_and_dis_says_what_it_must_hold(tmp_path, grasp_env, graspgenx):
    partial = write_checkpoints(tmp_path / "partial", parts=("gen",))
    with pytest.raises(FileNotFoundError, match=r"has no dis/config\.yaml.*'gen' and 'dis'"):
        make_sampler(grasp_env, checkpoint_dir=str(partial))


def test_a_failed_download_names_the_override(grasp_env, graspgenx):
    graspgenx.descriptions = None
    with pytest.raises(RuntimeError, match=r"GRASPGENX_GRIPPER_CFG_DIR"):
        make_sampler(grasp_env)


def test_a_missing_install_names_the_extra(grasp_env, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setitem(sys.modules, "graspgenx", None)
    with pytest.raises(ImportError, match=r'eden-graspgenx.*pip install "eden-world\[graspgenx\]"'):
        make_sampler(grasp_env)


def test_no_cuda_device_is_refused_before_loading(grasp_env, graspgenx, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA only"):
        make_sampler(grasp_env)
    assert graspgenx.loaded == []
