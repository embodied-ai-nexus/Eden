"""``TrackedObjectMixin`` on a real built env: the frame chain, the name checks, and the staleness guard.

The robot's wire is the loopback double and the pose source a scripted one (the camera boundary);
the env, the entities and the deployment are real.
"""

from __future__ import annotations

import math
import time

import numpy as np
import pytest
import torch

from eden.extensions.deployment.object_pose import PoseSample, PoseSource, TrackedObjectMixin
from eden.extensions.deployment.object_pose.backend import ObjectPoseLost
from eden.options.entities import BoxOptions
from eden.utils.geom import transform_by_quat
from tests.extensions.deployment.conftest import two_link_robot
from tests.support.deploy import LoopbackDeployment, deployments, robot_env_config, start_policy

IDENTITY = (1.0, 0.0, 0.0, 0.0)
YAW_90 = (math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4))


class ScriptedPoseSource(PoseSource):
    """Hands out :attr:`sample`; :meth:`go_stale` ages it past any limit."""

    name = "the scripted source"

    def __init__(self, pos, quat=IDENTITY) -> None:
        self.sample = PoseSample(np.array(pos, dtype=np.float64), np.array(quat, dtype=np.float64), time.monotonic())
        self.started = self.closed = False

    def start(self) -> None:
        self.started = True

    def latest(self) -> PoseSample | None:
        return self.sample

    def go_stale(self) -> None:
        self.sample = PoseSample(self.sample.pos, self.sample.quat, time.monotonic() - 60.0)

    def close(self) -> None:
        self.closed = True


class TrackedLoopback(TrackedObjectMixin, LoopbackDeployment):
    pass


@pytest.fixture(scope="module")
def cube_env(two_link_asset):
    """The fixed-base two-link arm and a free cube, built and reset once per module."""
    from tests.support.envs import build_env

    config = robot_env_config(two_link_robot(two_link_asset))
    config.scene_options.cube = BoxOptions(size=(0.05, 0.05, 0.05), default_root_pos=(0.5, 0.5, 0.1))
    return build_env(config, reset=True)


@pytest.fixture
def deploy(cube_env):
    with deployments(cube_env) as make:
        yield make


def _connected(deploy, source, **spec):
    dep = deploy(TrackedLoopback, tracked_objects={"cube": spec})
    dep.attach_pose_source("cube", source)
    dep.connect()
    dep.reset()
    return dep


def _cube_pos(env) -> np.ndarray:
    return env.entities["cube"].get_pos(relative=True)[0].cpu().numpy()


def test_a_world_fixed_camera_places_the_object_through_its_extrinsic(deploy, cube_env):
    source = ScriptedPoseSource([0.2, 0.0, 0.3])
    dep = _connected(deploy, source, parent_offset_pos=(1.0, 0.0, 0.0), parent_offset_quat=YAW_90)
    # The camera sits at (1, 0, 0) turned 90 deg about z: its +x is the world's +y.
    np.testing.assert_allclose(_cube_pos(cube_env), [1.0, 0.2, 0.3], atol=1e-5)
    np.testing.assert_allclose(dep.state.extra["cube.pos"], [1.0, 0.2, 0.3], atol=1e-5)
    assert source.started


def test_the_object_offset_applies_in_the_tracked_frame(deploy, cube_env):
    _connected(deploy, ScriptedPoseSource([0.2, 0.0, 0.3], YAW_90), object_offset_pos=(0.1, 0.0, 0.0))
    # The tracked frame is turned 90 deg, so its +x offset lands along the world's +y.
    np.testing.assert_allclose(_cube_pos(cube_env), [0.2, 0.1, 0.3], atol=1e-5)


def test_a_link_parented_source_follows_the_links_forward_kinematics(deploy, cube_env):
    _connected(deploy, ScriptedPoseSource([0.1, 0.0, 0.0]), parent_link="link2")
    robot = cube_env.entities["robot"]
    _, (idx,) = robot.find_named_links_idx_local(["link2"])
    link_pos, link_quat = robot.get_links_pos([idx])[:, 0], robot.get_links_quat([idx])[:, 0]
    offset = torch.tensor([[0.1, 0.0, 0.0]], device=link_pos.device, dtype=link_pos.dtype)
    expected = (link_pos + transform_by_quat(offset, link_quat))[0].cpu().numpy()
    np.testing.assert_allclose(_cube_pos(cube_env), expected, atol=1e-5)


@pytest.mark.parametrize(
    ("objects", "error", "match"),
    [
        ({"ball": {}}, KeyError, r"'ball', which is not a scene entity; entities: \['cube', 'robot'\]"),
        ({"robot": {}}, ValueError, "the deployed robot itself"),
        ({"cube": {"parent_link": "hand"}}, KeyError, r"parent_link='hand' is not a link of 'robot'; its links: \["),
        ({"cube": {"parent_entity": "robot"}}, ValueError, "without parent_link"),
        ({"cube": {"parent_offset_quat": (1.0, 1.0, 0.0, 0.0)}}, ValueError, "not a unit wxyz quaternion"),
    ],
)
def test_a_bad_spec_raises_at_construction_naming_the_value(deploy, objects, error, match):
    with pytest.raises(error, match=match):
        deploy(TrackedLoopback, tracked_objects=objects)


def test_connect_refuses_an_object_without_a_source(deploy):
    dep = deploy(TrackedLoopback, tracked_objects={"cube": {}})
    with pytest.raises(RuntimeError, match=r"no pose source attached for \['cube'\]"):
        dep.connect()


def test_a_stale_pose_under_the_policy_stops_it_and_start_is_refused_until_it_is_back(deploy, cube_env):
    source = ScriptedPoseSource([0.2, 0.0, 0.3])
    dep = _connected(deploy, source, max_age_s=0.2)
    start_policy(dep)
    assert dep.policy_active
    source.go_stale()
    dep.step(torch.zeros((1, 2), device=cube_env.device))
    assert not dep.policy_active
    dep.request_start()
    assert not dep.policy_active
    source.sample = PoseSample(source.sample.pos, source.sample.quat, time.monotonic())
    dep.request_start()
    assert dep.policy_active


def test_on_stale_raise_ends_the_tick(deploy, cube_env):
    source = ScriptedPoseSource([0.2, 0.0, 0.3])
    dep = deploy(TrackedLoopback, tracked_objects={"cube": {}}, on_stale="raise")
    dep.attach_pose_source("cube", source)
    dep.connect()
    dep.reset()
    start_policy(dep)
    source.go_stale()
    with pytest.raises(ObjectPoseLost, match="'cube'"):
        dep.step(torch.zeros((1, 2), device=cube_env.device))


def test_close_closes_every_source(deploy):
    source = ScriptedPoseSource([0.2, 0.0, 0.3])
    dep = _connected(deploy, source)
    dep.close()
    assert source.closed
