"""Every backend's ``END_EFFECTORS`` pairing, against the built Eden robot it names.

A pairing claims that a pose placed on the Eden link it names puts the real gripper where the
backend's gripper would be. The pairing is geometry only, so the backends' models are the fakes of
``conftest.py`` and the robots are real:

- a parallel jaw, open: its pads straddle the TCP (``tcp_offset``) along the closing axis
  (``closing_axis``), both read in the posed link's frame;
- a dexterous hand, at its zero configuration: every link the GraspGenX description and the Eden
  asset both name sits where the description puts it.

GraspGenX's descriptions are the real ones, from the local Hugging Face cache (skipped without them).
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch
import yourdfpy

import eden as en
from eden.extensions.grasp_sampling.graspgen import GraspGenSampler
from eden.extensions.grasp_sampling.graspgenx import GraspGenXSampler
from eden.options import EnvOptions, SceneOptions
from eden.options.entities import BoxOptions
from eden.utils.configs import EdenConfig
from eden.utils.geom import subtract_frame_transforms, trans_quat_to_T
from tests.extensions.grasp_sampling.conftest import cached_descriptions
from tests.support.envs import build_env

#: Each parallel jaw's two pads, whose centers of mass a closing grasp brings onto the object.
PADS = {
    "FrankaEmikaPanda": ("left_gripper_pad", "right_gripper_pad"),
    "FrankaHand": ("left_gripper_pad", "right_gripper_pad"),
    "Robotiq2f85": ("left_pad", "right_pad"),
}
#: How far the pads' midpoint may sit from the TCP. The 2F-85's pad centers of mass sit 5 mm from
#: GraspGenX's TCP, and GraspGen's Franka TCP is 2.4 mm deeper than the pads; a 90 degree error moves
#: neither, which is what the closing-axis check is for.
TCP_TOLERANCE = 0.01

GRASPGENX_JAWS = sorted(name for name in GraspGenXSampler.END_EFFECTORS if name in PADS)
GRASPGENX_HANDS = sorted(name for name in GraspGenXSampler.END_EFFECTORS if name not in PADS)
GRASPGEN_JAWS = sorted(GraspGenSampler.END_EFFECTORS)


@pytest.fixture(scope="module")
def robots_env():
    names = sorted(set(GraspGenXSampler.END_EFFECTORS) | set(GraspGenSampler.END_EFFECTORS))
    robots = {
        name: getattr(en.robots, name)(is_fixed_base=True, default_root_pos=(2.0 * (i + 1), 0.0, 1.0))
        for i, name in enumerate(names)
    }
    return build_env(
        EdenConfig(
            env_options=EnvOptions(num_envs=1),
            # Something to sample from; the fake models ignore its points.
            scene_options=SceneOptions(target=BoxOptions(size=(0.05,) * 3, is_fixed_base=True), **robots),
        )
    )


def in_link_frame(entity, link_name: str, names: list[str], ref: str = "link_origin") -> np.ndarray:
    """Where the links ``names`` are (their ``ref`` point) in link ``link_name``'s frame, shape ``(n, 3)``."""
    _, (base,) = entity.find_named_links_idx_local([link_name])
    found, idx = entity.find_named_links_idx_local(names)
    assert found == names
    base_pos = entity.get_links_pos(ls_idx_local=[base])[0].expand(len(idx), 3)
    base_quat = entity.get_links_quat(ls_idx_local=[base])[0].expand(len(idx), 4)
    pos = entity.get_links_pos(ls_idx_local=idx, ref=ref)[0]
    quat = entity.get_links_quat(ls_idx_local=idx)[0]
    return subtract_frame_transforms(base_pos, base_quat, pos, quat)[0].double().cpu().numpy()


def set_dofs(entity, targets: dict[str, float]):
    names, idx = entity.find_named_dofs_idx_local(list(targets))
    entity.set_dofs_pos(torch.tensor([[targets[name] for name in names]], device=entity.device), dofs_idx_local=idx)


def check_pads_straddle_the_tcp(env, sampler, end_effector: str):
    robot = env.entities[end_effector]
    set_dofs(robot, dict(getattr(en.robots, end_effector).open_dofs_pos))
    left, right = in_link_frame(robot, sampler.pose_link_name, list(PADS[end_effector]), ref="link_com")
    tcp = sampler.tcp_offset.double().cpu().numpy()
    closing = sampler.closing_axis.double().cpu().numpy()
    assert np.linalg.norm((left + right) / 2 - tcp) < TCP_TOLERANCE
    assert abs(np.dot((left - right) / np.linalg.norm(left - right), closing)) > 0.99
    # The fingers reach out along the approach axis.
    assert np.dot(sampler.approach_axis.double().cpu().numpy(), tcp / np.linalg.norm(tcp)) > 0.95


@pytest.mark.parametrize("end_effector", GRASPGENX_JAWS)
def test_a_graspgenx_parallel_jaw_closes_its_pads_on_the_tcp(end_effector, robots_env, graspgenx):
    graspgenx.descriptions = cached_descriptions()
    sampler = GraspGenXSampler(
        robots_env, GraspGenXSampler.configure(end_effector=end_effector, object_cloud="surface")
    )
    check_pads_straddle_the_tcp(robots_env, sampler, end_effector)


@pytest.mark.parametrize("end_effector", GRASPGEN_JAWS)
def test_a_graspgen_parallel_jaw_closes_its_pads_on_the_tcp(end_effector, robots_env, grasp_gen):
    sampler = GraspGenSampler(robots_env, GraspGenSampler.configure(end_effector=end_effector, object_cloud="surface"))
    check_pads_straddle_the_tcp(robots_env, sampler, end_effector)


@pytest.mark.parametrize("end_effector", GRASPGENX_HANDS)
def test_a_graspgenx_dexterous_hand_matches_its_description_link_for_link(end_effector, robots_env, graspgenx):
    graspgenx.descriptions = cached_descriptions()
    sampler = GraspGenXSampler(
        robots_env, GraspGenXSampler.configure(end_effector=end_effector, object_cloud="surface")
    )
    # The fake planner returns the identity grasp frame, so the pose is the posed link in the grasp frame.
    result = sampler.sample_grasps("target")
    grasp_T_link = trans_quat_to_T(result.pos[0].double().cpu(), result.quat[0].double().cpu()).numpy()

    robot = robots_env.entities[end_effector]
    robot.set_dofs_pos(torch.zeros((1, robot.n_dofs), device=robot.device))
    urdf = yourdfpy.URDF.load(
        os.path.join(graspgenx.descriptions, sampler.gripper_name, "gripper.urdf"),
        load_meshes=False,
        build_collision_scene_graph=False,
        load_collision_meshes=False,
    )
    shared = sorted({link.name for link in urdf.robot.links} & {link.name for link in robot.links} - {"world"})
    assert len(shared) >= 10
    eden_links = in_link_frame(robot, sampler.pose_link_name, shared)
    link_T_grasp = np.linalg.inv(grasp_T_link)
    description_links = np.array(
        [(link_T_grasp @ urdf.get_transform(frame_to=name, frame_from="world"))[:3, 3] for name in shared]
    )
    error = np.linalg.norm(eden_links - description_links, axis=1)
    # WujiHand1_R's fingertips differ from its description's by up to 4.7 mm; the rest match to 0.2 mm.
    assert np.median(error) < 0.001 and error.max() < 0.006, dict(zip(shared, np.round(1000 * error, 1)))
