"""Tests for ``eden.extensions.grasp_sampling.base``: the point clouds, ranking and checks every backend shares.

The backend here is a real ``GraspSamplerBase`` subclass whose "model" returns scripted grasps and
records the clouds it was handed — the two hooks a backend implements.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from eden.extensions.grasp_sampling import EndEffectorPairing, GraspSamplerBase
from tests.extensions.grasp_sampling.conftest import BOX_CENTER, BOX_HALF, TABLE_TOP, grasp_scene_config
from tests.support.envs import build_env


HALF = math.sqrt(0.5)


class ScriptedSampler(GraspSamplerBase):
    """Returns ``self.scripted = (transforms, confidences)``; keeps ``self.collision_mask`` of them.

    Its poses are the grasp frame itself, whose TCP is 10 cm along the approach axis. It pairs its one
    gripper, ``"toy"``, with the Franka hand: 2 cm along the approach axis, a quarter turn about it.
    """

    gripper: str = ""

    END_EFFECTORS = {"FrankaHand": EndEffectorPairing("toy", "hand", ((0.0, 0.0, 0.02), (HALF, 0.0, 0.0, HALF)))}
    _gripper_option = "gripper"

    def __init__(self, env, options):
        super().__init__(env, options)
        self.clouds_per_call = []
        self._set_pose_frame(None, np.eye(4), np.array([0.0, 0.0, 0.1]))

    def _infer(self, object_points):
        self.seen_object_points = object_points
        return self.scripted

    def _infer_batch(self, clouds):
        self.clouds_per_call.append(clouds)
        return super()._infer_batch(clouds)

    def _collision_free(self, grasps, scene_points):
        self.seen_grasps, self.seen_scene_points = grasps, scene_points
        return self.collision_mask


def make_sampler(env, **params) -> ScriptedSampler:
    sampler = ScriptedSampler(env, ScriptedSampler.configure(**params))
    sampler.scripted = (np.zeros((0, 4, 4)), np.zeros(0))
    return sampler


def yaw_transform(pos, yaw):
    T = np.eye(4)
    T[:2, :2] = [[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]]
    T[:3, 3] = pos
    return T


def inside_box(points, center, half, margin=0.005):
    return (np.abs(points - np.asarray(center)) <= np.asarray(half) + margin).all(axis=1)


def test_camera_cloud_splits_the_target_from_its_obstacles(grasp_env):
    sampler = make_sampler(grasp_env, filter_collisions=True, collision_ignore_entities=[])
    sampler.scripted = (np.eye(4)[None], np.ones(1))
    sampler.collision_mask = np.ones(1, dtype=bool)
    sampler.sample_grasps("box")

    target = sampler.seen_object_points
    assert len(target) > 50
    assert inside_box(target, BOX_CENTER, (BOX_HALF,) * 3).all()
    obstacles = sampler.seen_scene_points
    # A camera sees surfaces only, so "none of the box" means none on its faces above the table top.
    on_box = inside_box(obstacles, BOX_CENTER, (BOX_HALF,) * 3) & (obstacles[:, 2] > TABLE_TOP + 0.003)
    assert not on_box.any()
    table = inside_box(obstacles, (0.5, 0.0, TABLE_TOP / 2), (0.3, 0.3, TABLE_TOP / 2))
    assert table.sum() > 50


def test_actuated_entities_leave_the_obstacle_cloud_by_default(grasp_env):
    hand_points = []
    for ignore in (None, []):
        sampler = make_sampler(grasp_env, filter_collisions=True, collision_ignore_entities=ignore)
        sampler.scripted = (np.eye(4)[None], np.ones(1))
        sampler.collision_mask = np.ones(1, dtype=bool)
        sampler.sample_grasps("box")
        # The hand hangs 15 cm above the table at y=0.2; nothing else is up there.
        hand_points.append(
            inside_box(sampler.seen_scene_points, (0.5, 0.2, TABLE_TOP + 0.15), (0.06, 0.12, 0.08)).sum()
        )
    assert hand_points[0] == 0
    assert hand_points[1] > 20


def test_listed_entities_leave_the_obstacle_cloud(grasp_env):
    sampler = make_sampler(grasp_env, filter_collisions=True, collision_ignore_entities=["table"])
    sampler.scripted = (np.eye(4)[None], np.ones(1))
    sampler.collision_mask = np.ones(1, dtype=bool)
    sampler.sample_grasps("box")
    assert not inside_box(sampler.seen_scene_points, (0.5, 0.0, TABLE_TOP / 2), (0.3, 0.3, TABLE_TOP / 2 - 0.01)).any()


def test_the_camera_sees_a_pose_set_since_the_last_step(grasp_env):
    hidden = grasp_env.entities["hidden"]
    home = hidden.get_pos(relative=True).clone()
    hidden.set_pos(torch.tensor([[0.5, -0.15, TABLE_TOP + 0.025]], device=grasp_env.device), relative=True)
    try:
        sampler = make_sampler(grasp_env)
        sampler.sample_grasps("hidden")
        np.testing.assert_allclose(np.median(sampler.seen_object_points[:, :2], axis=0), (0.5, -0.15), atol=0.02)
    finally:
        hidden.set_pos(home, relative=True)


def test_surface_cloud_samples_the_whole_entity_without_a_camera(grasp_env):
    sampler = make_sampler(grasp_env, object_cloud="surface", surface_points=500, camera_name="no_such_camera")
    sampler.sample_grasps("box")
    points = sampler.seen_object_points
    assert points.shape == (500, 3)
    assert inside_box(points, BOX_CENTER, (BOX_HALF,) * 3).all()
    # Every face, the one resting on the table included — which a camera never sees.
    assert (points[:, 2] < BOX_CENTER[2] - BOX_HALF + 0.002).any()


def test_result_is_ranked_best_first_capped_and_in_pos_wxyz_quat(grasp_env):
    sampler = make_sampler(grasp_env, max_grasps=2)
    sampler.scripted = (
        np.stack([yaw_transform((0.1, 0.0, 0.3), 0.0), yaw_transform((0.2, 0.0, 0.3), math.pi / 2), np.eye(4)]),
        np.array([0.5, 0.9, 0.2], dtype=np.float32),
    )
    result = sampler.sample_grasps("box")

    assert result.success and len(result) == 2
    torch.testing.assert_close(result.confidences.cpu(), torch.tensor([0.9, 0.5]))
    torch.testing.assert_close(result.pos.cpu(), torch.tensor([[0.2, 0.0, 0.3], [0.1, 0.0, 0.3]]))
    torch.testing.assert_close(result.quat.cpu(), torch.tensor([[HALF, 0.0, 0.0, HALF], [1.0, 0.0, 0.0, 0.0]]))


def test_no_grasp_is_an_empty_result(grasp_env):
    result = make_sampler(grasp_env).sample_grasps("box")
    assert not result.success
    assert result.pos.shape == (0, 3) and result.quat.shape == (0, 4)


def test_filter_collisions_drops_the_grasps_the_backend_rejects(grasp_env):
    sampler = make_sampler(grasp_env, filter_collisions=True)
    sampler.scripted = (
        np.stack([yaw_transform((0.1 * i, 0.0, 0.3), 0.0) for i in range(3)]),
        np.array([0.9, 0.8, 0.7]),
    )
    sampler.collision_mask = np.array([True, False, True])
    result = sampler.sample_grasps("box")
    torch.testing.assert_close(result.pos[:, 0].cpu(), torch.tensor([0.0, 0.2]))


def test_max_grasps_keeps_the_best_of_the_collision_free_grasps(grasp_env):
    sampler = make_sampler(grasp_env, filter_collisions=True, max_grasps=1)
    sampler.scripted = (
        np.stack([yaw_transform((0.1 * i, 0.0, 0.3), 0.0) for i in range(3)]),
        np.array([0.9, 0.8, 0.7]),
    )
    sampler.collision_mask = np.array([False, True, True])
    torch.testing.assert_close(sampler.sample_grasps("box").pos[:, 0].cpu(), torch.tensor([0.1]))


def test_end_effector_poses_place_its_link_and_the_filter_sees_grasp_frames(grasp_env):
    sampler = make_sampler(grasp_env, end_effector="FrankaHand", filter_collisions=True)
    sampler.scripted = (yaw_transform((0.5, 0.0, 0.4), 0.0)[None], np.ones(1))
    sampler.collision_mask = np.ones(1, dtype=bool)
    result = sampler.sample_grasps("box")

    assert sampler.gripper == "toy" and sampler.pose_link_name == "hand"
    np.testing.assert_allclose(sampler.seen_grasps, sampler.scripted[0])
    torch.testing.assert_close(result.pos.cpu(), torch.tensor([[0.5, 0.0, 0.42]]))
    torch.testing.assert_close(result.quat.cpu(), torch.tensor([[HALF, 0.0, 0.0, HALF]]))
    # The same grasp, read in the hand's frame: it closes along -y, and the TCP is 8 cm out.
    torch.testing.assert_close(sampler.closing_axis.cpu(), torch.tensor([0.0, -1.0, 0.0]), atol=1e-6, rtol=0)
    torch.testing.assert_close(sampler.approach_axis.cpu(), torch.tensor([0.0, 0.0, 1.0]), atol=1e-6, rtol=0)
    torch.testing.assert_close(sampler.tcp_offset.cpu(), torch.tensor([0.0, 0.0, 0.08]), atol=1e-6, rtol=0)


def test_without_end_effector_the_poses_are_the_backend_frame(grasp_env):
    sampler = make_sampler(grasp_env)
    assert sampler.pose_link_name is None
    torch.testing.assert_close(sampler.tcp_offset.cpu(), torch.tensor([0.0, 0.0, 0.1]))


def test_an_unknown_end_effector_lists_the_pairings(grasp_env):
    with pytest.raises(ValueError, match=r"end_effector='Panda' is not an Eden end-effector .*\['FrankaHand'\]"):
        make_sampler(grasp_env, end_effector="Panda")


def test_an_end_effector_and_a_different_gripper_are_refused(grasp_env):
    with pytest.raises(ValueError, match=r"pairs with gripper='toy', but gripper='other' is configured too"):
        make_sampler(grasp_env, end_effector="FrankaHand", gripper="other")


def test_a_backend_that_never_declares_its_pose_frame_fails_at_the_first_sample(grasp_env):
    class UndeclaredSampler(GraspSamplerBase):
        def _infer(self, object_points):
            return np.zeros((0, 4, 4)), np.zeros(0)

    sampler = UndeclaredSampler(grasp_env, UndeclaredSampler.configure(object_cloud="surface"))
    with pytest.raises(RuntimeError, match=r"UndeclaredSampler never declared .*_set_pose_frame"):
        sampler.sample_grasps("box")


def test_a_backend_without_a_collision_filter_refuses_filter_collisions_at_construction(grasp_env):
    class NoFilterSampler(GraspSamplerBase):
        def _infer(self, object_points):
            return np.zeros((0, 4, 4)), np.zeros(0)

    with pytest.raises(ValueError, match=r"NoFilterSampler cannot filter_collisions.*detect_collision"):
        NoFilterSampler(grasp_env, NoFilterSampler.configure(filter_collisions=True))


def test_a_backend_returning_mismatched_shapes_fails_at_the_boundary(grasp_env):
    sampler = make_sampler(grasp_env)
    sampler.scripted = (np.stack([np.eye(4)] * 3), np.ones(2))
    with pytest.raises(
        RuntimeError, match=r"_infer returned transforms of shape \(3, 4, 4\) and confidences of shape \(2,\)"
    ):
        sampler.sample_grasps("box")


def test_unknown_camera_names_the_cameras_there_are(grasp_env):
    with pytest.raises(ValueError, match=r"camera_name='cam' is not a camera of this env; it has \['camera'\]"):
        make_sampler(grasp_env, camera_name="cam")


def test_unknown_entity_names_the_entities_there_are(grasp_env):
    with pytest.raises(KeyError, match=r"No scene entity named 'boxx'.*'box'"):
        make_sampler(grasp_env).sample_grasps("boxx")


def test_unknown_ignored_entity_fails_at_construction_even_with_the_filter_off(grasp_env):
    with pytest.raises(KeyError, match=r"No scene entity named 'tabel'.*collision_ignore_entities"):
        make_sampler(grasp_env, collision_ignore_entities=["tabel"])


def test_an_entity_the_camera_cannot_see_raises_rather_than_returning_no_grasp(grasp_env):
    with pytest.raises(ValueError, match=r"Camera 'camera' sees no point of entity 'hidden'.*object_cloud='surface'"):
        make_sampler(grasp_env).sample_grasps("hidden")


def test_a_bad_shared_option_fails_at_configure():
    with pytest.raises(ValueError, match="object_cloud"):
        ScriptedSampler.configure(object_cloud="mesh")
    with pytest.raises(ValueError, match="grasp_threshold"):
        ScriptedSampler.configure(grasp_threshold=1.5)


def test_a_camera_shared_by_several_envs_is_refused():
    env = build_env(grasp_scene_config(num_envs=2))
    with pytest.raises(ValueError, match=r"renders 1 view\(s\) for 2 envs.*env_separate_rigid=True"):
        make_sampler(env)
    # A surface cloud reads no camera, so it samples each env's own copy.
    sampler = make_sampler(env, object_cloud="surface", surface_points=200)
    offset = (
        (env.entities["box"].get_pos(relative=True)[1] - env.entities["box"].get_pos(relative=True)[0]).cpu().numpy()
    )
    sampler.sample_grasps("box", env_idx=1)
    np.testing.assert_allclose(sampler.seen_object_points.mean(axis=0), np.asarray(BOX_CENTER) + offset, atol=0.005)
    with pytest.raises(IndexError, match=r"env_idx=2 is out of range for this env's num_envs=2"):
        sampler.sample_grasps("box", env_idx=2)


@pytest.fixture(scope="module")
def separate_env():
    """Two envs, each rendered by the camera on its own."""
    return build_env(grasp_scene_config(num_envs=2, env_separate_rigid=True))


def test_a_camera_that_renders_each_env_samples_the_env_asked_for(separate_env):
    env = separate_env
    hidden = env.entities["hidden"]
    # Only env 1 moves its 'hidden' box into the camera's view.
    hidden.set_pos(torch.tensor([[0.5, -0.15, TABLE_TOP + 0.025]], device=env.device), envs_idx=[1], relative=True)
    sampler = make_sampler(env)
    sampler.sample_grasps("hidden", env_idx=1)
    np.testing.assert_allclose(np.median(sampler.seen_object_points[:, :2], axis=0), (0.5, -0.15), atol=0.02)
    with pytest.raises(ValueError, match=r"sees no point of entity 'hidden' in env 0"):
        sampler.sample_grasps("hidden", env_idx=0)


def test_a_batch_renders_once_and_hands_the_model_each_env_cloud_in_order(separate_env, monkeypatch):
    env = separate_env
    sampler = make_sampler(env)
    sampler.scripted = (np.eye(4)[None], np.ones(1))
    renders = []
    render = sampler.camera.render
    monkeypatch.setattr(sampler.camera, "render", lambda *args, **kwargs: renders.append(1) or render(*args, **kwargs))

    results = sampler.sample_grasps_batch("box")

    assert len(renders) == 1 and len(results) == 2
    (clouds,) = sampler.clouds_per_call
    box = env.entities["box"].get_pos(relative=True).cpu().numpy()
    for cloud, center in zip(clouds, box):
        assert inside_box(cloud, center, (BOX_HALF,) * 3).all()
    with pytest.raises(IndexError, match=r"env_idx=2 is out of range"):
        sampler.sample_grasps_batch("box", [0, 2])
