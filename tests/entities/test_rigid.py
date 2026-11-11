"""``RigidEntity``: pose and velocity frames, attachment, DOF and link accessors, post-build setup, and index selectors.

Each section shares one module-scoped scene; a test that moves an entity resets the env afterwards.
"""

from __future__ import annotations

import gc
import math
import os

import genesis as gs
import genesis.utils.geom as gu
import numpy as np
import pytest
import torch

from eden.entities.rigid import _RESOLVED_SELECTIONS, StagedStateWrite, _genesis_selection, _kernel_selection
from eden.envs.base import EnvBase
from eden.options import BoxOptions, EntityOptions, EnvOptions, PlaneOptions, SceneOptions
from eden.options.entities import GroupedEntityOptions, RobotOptions, SphereOptions
from eden.options.materials import KinematicMaterialOptions, RigidMaterialOptions
from eden.options.actuators import Unitree5020_16_DOUBLE, Unitree7520_14, Unitree7520_22
from eden.options.robots import FrankaEmikaPanda, FrankaHand, FrankaResearch3, Robotiq2f85
from eden.options.robots.unitree_g1 import G1Dofs29
from eden.utils.configs import EdenConfig
from eden.utils.geom import align_up_and_front
from eden.utils.misc import DofPermutation, DofRuns, get_gs_assets_dir
from tests.entities.conftest import HAND_STL, assert_quat_close, captured_eden_logs, euler_quat, to_np
from tests.support.envs import build_env, control_force_over_one_step

pytestmark = pytest.mark.requires_genesis


def _config(num_envs: int = 1, **entities) -> EdenConfig:
    return EdenConfig(env_options=EnvOptions(num_envs=num_envs), scene_options=SceneOptions(**entities))


# ═════════════════════════════════════════════════════════════════════════════════════════════════
# Pose frames (https://github.com/Genesis-Embodied-AI/Genesis/pull/2934)
#
# The default getters report Genesis' solver frame (a free body re-origined onto its centre of mass);
# ``relative=True`` reports the authored frame ``default_root_pos`` / ``default_root_quat`` live in.
# ═════════════════════════════════════════════════════════════════════════════════════════════════

# up=(0,1,0)/front=(0,0,1) aligns by the non-identity [.5, .5, .5, .5].
_UP, _FRONT = (0, 1, 0), (0, 0, 1)
_ALIGN = np.asarray(gu.R_to_quat(align_up_and_front(_UP, _FRONT)), dtype=np.float64)
_HAND_POS, _HAND_QUAT = (0.3, 0.2, 0.6), euler_quat((0, 0, 90))
_PLAIN_QUAT = euler_quat((0, 0, 30))
_BOX_POS = (0.4, -0.3, 0.7)
_POSED_ROOT_POS = (0.3, -0.8, 0.78)
_TOOL_ORIGIN = np.array((-0.8, -0.8, 0.5))
_UNALIGNED_TOOL_ORIGIN = np.array((-1.2, -1.6, 0.5))
_Z90 = (math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4))
_GROUP_POS = (2.5, 2.0, 0.5)
_OFFSET_GROUP_POS, _VARIANT_OFFSETS = (2.5, -2.0, 0.5), ((0.1, 0.0, 0.0), (-0.1, 0.0, 0.0))
_WHEEL = os.path.join(get_gs_assets_dir(), "urdf/wheel/wheel.urdf")

# A floating root whose MJCF body carries its own pose, which the default root pose must replace.
_POSED_ROOT_MJCF = """<mujoco model="posed_root"><worldbody>
  <body name="pelvis" pos="0 0 0.793" quat="0.7071068 0 0 0.7071068">
    <freejoint/><geom type="box" size="0.05 0.05 0.05" mass="1.0"/>
  </body>
</worldbody></mujoco>"""

# A free root whose centre of mass is 0.3 m off its origin, carrying a fixed child link.
_TOOL_MJCF = """<mujoco model="tool"><worldbody>
  <body name="root">
    <freejoint/><geom type="box" size="0.02 0.02 0.02" pos="0.3 0 0" mass="1.0"/>
    <body name="handle" pos="0 0.1 0"><geom type="box" size="0.01 0.01 0.01" mass="0.01"/></body>
  </body>
</worldbody></mujoco>"""


@pytest.fixture(scope="module")
def _pose_env(tmp_path_factory):
    assets = tmp_path_factory.mktemp("pose_assets")
    (assets / "posed_root.xml").write_text(_POSED_ROOT_MJCF)
    (assets / "tool.xml").write_text(_TOOL_MJCF)
    wheel = {}
    if os.path.exists(_WHEEL):
        # A fixed URDF whose base link has no geometry, so it has no settable base pose at all.
        wheel["wheel"] = EntityOptions(
            file=_WHEEL, scale=0.6, default_root_pos=(-1.5, 0.0, 1.0), default_root_quat=_Z90, is_fixed_base=True
        )
    return build_env(
        _config(
            num_envs=2,
            plane=PlaneOptions(),
            box=BoxOptions(size=(0.2, 0.2, 0.2), default_root_pos=_BOX_POS),
            hand=EntityOptions(
                file=HAND_STL,
                up=_UP,
                front=_FRONT,
                default_root_pos=_HAND_POS,
                default_root_quat=tuple(float(v) for v in _HAND_QUAT),
                convexify=True,
            ),
            plain=EntityOptions(
                file=HAND_STL,
                default_root_pos=(-0.5, 0.5, 0.5),
                default_root_quat=tuple(float(v) for v in _PLAIN_QUAT),
                convexify=True,
            ),
            posed_root=EntityOptions(file=str(assets / "posed_root.xml"), default_root_pos=_POSED_ROOT_POS),
            tool=EntityOptions(file=str(assets / "tool.xml"), default_root_pos=tuple(_TOOL_ORIGIN)),
            unaligned_tool=EntityOptions(
                file=str(assets / "tool.xml"), default_root_pos=tuple(_UNALIGNED_TOOL_ORIGIN), align=False
            ),
            fixed_box=BoxOptions(
                size=(0.2, 0.2, 0.2), default_root_pos=(1.5, 0.0, 0.5), default_root_quat=_Z90, fixed=True
            ),
            turntable=BoxOptions(
                size=(0.1, 0.1, 0.1), default_root_pos=(1.5, 1.0, 0.5), fixed=True, batch_fixed_verts=True
            ),
            group=GroupedEntityOptions(
                grouped_entities=[
                    BoxOptions(size=(0.10, 0.10, 0.10), default_root_pos=(0.0, 0.0, 0.0)),
                    BoxOptions(size=(0.15, 0.15, 0.15), default_root_pos=(0.0, 0.0, 0.0)),
                ],
                is_fixed_base=True,
                default_root_pos=_GROUP_POS,
                default_root_quat=_Z90,
            ),
            # Env-specific poses on a fixed base need batch_fixed_verts on the group and its variants.
            offset_group=GroupedEntityOptions(
                grouped_entities=[
                    BoxOptions(size=(0.10, 0.10, 0.10), default_root_pos=off, batch_fixed_verts=True)
                    for off in _VARIANT_OFFSETS
                ],
                is_fixed_base=True,
                default_root_pos=_OFFSET_GROUP_POS,
                batch_fixed_verts=True,
            ),
            **wheel,
        )
    )


@pytest.fixture
def pose_env(_pose_env):
    yield _pose_env
    _pose_env.reset()


def test_forward_vec_is_the_options_front_and_x_for_a_primitive(pose_env):
    np.testing.assert_allclose(to_np(pose_env.entities["hand"].forward_vec)[0], _FRONT, atol=1e-6)
    np.testing.assert_allclose(to_np(pose_env.entities["box"].forward_vec)[0], (1, 0, 0), atol=1e-6)


def test_a_primitive_reports_the_pose_it_was_placed_at(pose_env):
    box = pose_env.entities["box"]
    np.testing.assert_allclose(to_np(box.get_pos()), np.tile(_BOX_POS, (2, 1)), atol=1e-5)
    for quat in to_np(box.get_quat()):
        assert_quat_close(quat, (1.0, 0.0, 0.0, 0.0))


def test_default_root_pos_places_the_authored_origin_not_the_solver_origin(pose_env):
    """With up/front and a yaw applied, so the position write has to see the final orientation."""
    hand = pose_env.entities["hand"]
    np.testing.assert_allclose(to_np(hand.get_pos(relative=True)), np.tile(_HAND_POS, (2, 1)), atol=1e-4)
    assert np.linalg.norm(to_np(hand.get_pos())[0] - np.asarray(_HAND_POS)) > 1e-3  # the frames do differ


def test_world_quat_is_the_alignment_composed_with_default_root_quat(pose_env):
    expected = gu.transform_quat_by_quat(_ALIGN, _HAND_QUAT)
    for quat in to_np(pose_env.entities["hand"].get_quat()):
        assert_quat_close(quat, expected)
    # With the default up/front the alignment is the identity.
    for quat in to_np(pose_env.entities["plain"].get_quat()):
        assert_quat_close(quat, _PLAIN_QUAT)


@pytest.mark.parametrize("relative", [False, True])
def test_base_link_and_entity_getters_agree_in_both_frames(pose_env, relative):
    hand = pose_env.entities["hand"]
    base = hand.get_link(hand.base_link_name).idx_local
    pos, quat = to_np(hand.get_pos(relative=relative)), to_np(hand.get_quat(relative=relative))
    np.testing.assert_allclose(to_np(hand.get_links_pos(relative=relative))[:, base], pos, atol=1e-5)
    for link_quat, root_quat in zip(to_np(hand.get_links_quat(relative=relative))[:, base], quat):
        assert_quat_close(link_quat, root_quat)
    expected_T = gu.trans_quat_to_T(hand.get_pos(relative=relative), hand.get_quat(relative=relative))
    np.testing.assert_allclose(to_np(hand.get_T(relative=relative)), to_np(expected_T), atol=1e-5)


def test_relative_quat_is_the_user_frame_default_root_quat(pose_env):
    hand = pose_env.entities["hand"]
    for quat in to_np(hand.get_quat(relative=True)):
        assert_quat_close(quat, _HAND_QUAT)
    assert not np.allclose(gu.quat_to_R(to_np(hand.get_quat())[0]), gu.quat_to_R(_HAND_QUAT), atol=1e-3)


def test_reset_restores_the_build_pose_in_both_frames(pose_env):
    hand = pose_env.entities["hand"]
    built_user, built_world = to_np(hand.get_pos(relative=True)), to_np(hand.get_pos())
    hand.set_pos(np.tile((1.0, 1.0, 1.0), (2, 1)))
    pose_env.reset()
    np.testing.assert_allclose(to_np(hand.get_pos(relative=True)), built_user, atol=1e-5)
    np.testing.assert_allclose(to_np(hand.get_pos()), built_world, atol=1e-5)


def test_default_root_pose_replaces_an_mjcf_root_bodys_own_pose(pose_env):
    """The G1 pelvis sits at pos z=0.793 in its MJCF; composing it with default_root_pos spawned it 0.79 m up."""
    robot = pose_env.entities["posed_root"]
    for when in ("after build", "after env.reset()"):
        np.testing.assert_allclose(to_np(robot.get_pos(relative=True)), np.tile(_POSED_ROOT_POS, (2, 1)), atol=1e-5)
        for quat in to_np(robot.get_quat(relative=True)):
            assert_quat_close(quat, (1.0, 0.0, 0.0, 0.0), err_msg=when)
        robot.set_pos(np.tile((1.0, 1.0, 1.0), (2, 1)))
        pose_env.reset()


@pytest.mark.parametrize("entity, quat", [("box", (0, 0, 75)), ("hand", (10, 0, -60))], ids=["primitive", "mesh"])
def test_set_quat_composes_onto_the_alignment(pose_env, entity, quat):
    """``set_quat(q)`` yields world = alignment ∘ q; for a primitive the alignment is the identity."""
    target = pose_env.entities[entity]
    q = euler_quat(quat)
    target.set_quat(torch.tensor(np.tile(q, (2, 1)), dtype=torch.float32, device=pose_env.device))
    align = _ALIGN if entity == "hand" else np.array([1.0, 0.0, 0.0, 0.0])
    for got in to_np(target.get_quat()):
        assert_quat_close(got, gu.transform_quat_by_quat(align, q))


@pytest.mark.parametrize("entity, relative", [("box", False), ("hand", True)])
def test_set_pos_round_trips_through_get_pos(pose_env, entity, relative):
    target = pose_env.entities[entity]
    new_pos = torch.tensor([[0.25, -0.35, 0.8]] * 2, device=pose_env.device)
    target.set_pos(new_pos, relative=relative)
    np.testing.assert_allclose(to_np(target.get_pos(relative=relative)), to_np(new_pos), atol=1e-5)


def test_a_fixed_child_link_carries_the_root_shift_that_relative_cannot_strip(pose_env):
    """Genesis anchors the free root on its CoM; only the root link carries the offset ``relative`` strips.

    Pins the recovery ``docs/guides/entities.md`` tells users to apply, so a Genesis change shows here.
    """
    tool = pose_env.entities["tool"]
    handle = [tool.get_link("handle").idx_local]
    authored_handle = _TOOL_ORIGIN + np.array((0.0, 0.1, 0.0))

    handle_geom = next(g for g in tool.geoms if g.link.name == "handle")
    np.testing.assert_allclose(to_np(handle_geom.get_pos())[0], authored_handle, atol=1e-5)
    solver = to_np(tool.get_links_pos(ls_idx_local=handle))[0, 0]
    np.testing.assert_allclose(to_np(tool.get_links_pos(ls_idx_local=handle, relative=True))[0, 0], solver)
    assert np.linalg.norm(solver - authored_handle) > 0.2

    # authored child = authored root ∘ solver root⁻¹ ∘ solver child
    in_root = gu.inv_transform_by_trans_quat(solver, to_np(tool.get_pos())[0], to_np(tool.get_quat())[0])
    recovered = gu.transform_by_trans_quat(
        in_root, to_np(tool.get_pos(relative=True))[0], to_np(tool.get_quat(relative=True))[0]
    )
    np.testing.assert_allclose(recovered, authored_handle, atol=1e-5)


def test_align_false_keeps_a_free_body_at_its_authored_frame(pose_env):
    """``align=False`` puts the free joint on the authored body frame, so both getters agree.

    Aligned (the default), they differ by the CoM offset: the tool's box sits 0.3 m off its origin.
    """
    aligned, unaligned = pose_env.entities["tool"], pose_env.entities["unaligned_tool"]
    np.testing.assert_allclose(to_np(unaligned.get_pos()), to_np(unaligned.get_pos(relative=True)), atol=1e-6)
    np.testing.assert_allclose(to_np(unaligned.get_pos()), np.tile(_UNALIGNED_TOOL_ORIGIN, (2, 1)), atol=1e-5)
    for solver, authored in zip(to_np(unaligned.get_quat()), to_np(unaligned.get_quat(relative=True))):
        assert_quat_close(solver, authored)
    gap = to_np(aligned.get_pos()) - to_np(aligned.get_pos(relative=True))
    assert (np.linalg.norm(gap, axis=-1) > 0.25).all()


def _quat_matches(actual, expected, tol=1e-3) -> bool:
    return abs(float(np.dot(actual, expected))) > 1 - tol


def test_a_fixed_base_primitive_is_oriented_by_its_baked_morph_pose(pose_env):
    """A fixed base cannot be re-posed after build, so its root pose is baked into the morph instead."""
    for quat in to_np(pose_env.entities["fixed_box"].get_quat()):
        assert _quat_matches(quat, _Z90)


def test_a_fixed_urdf_without_base_geometry_builds_and_is_oriented(pose_env):
    """wheel.urdf has no settable base pose; the old post-build set_quat raised ``Expected shape: (0,)``."""
    if "wheel" not in pose_env.entities:
        pytest.skip("wheel.urdf asset not available")
    wheel = pose_env.entities["wheel"]
    assert _quat_matches(to_np(wheel.get_quat())[0], _Z90)
    np.testing.assert_allclose(to_np(wheel.get_pos())[0], (-1.5, 0.0, 1.0), atol=1e-4)


def test_a_fixed_heterogeneous_group_takes_the_group_pose_composed_with_each_variant(pose_env):
    """Grouped morphs bake only each variant's own pose; the group pose arrives through the post-build write."""
    group = pose_env.entities["group"]
    np.testing.assert_allclose(to_np(group.get_pos()), np.tile(_GROUP_POS, (2, 1)), atol=1e-4)
    for quat in to_np(group.get_quat()):
        assert _quat_matches(quat, _Z90)
    expected = [np.add(_OFFSET_GROUP_POS, off) for off in _VARIANT_OFFSETS]  # variant i -> env i
    np.testing.assert_allclose(to_np(pose_env.entities["offset_group"].get_pos()), expected, atol=1e-4)


def test_per_env_orientation_of_a_fixed_body_needs_batch_fixed_verts(pose_env):
    per_env = torch.tensor([[1.0, 0.0, 0.0, 0.0], list(_Z90)], device=pose_env.device)
    turntable = pose_env.entities["turntable"]
    turntable.set_quat(per_env)
    got = to_np(turntable.get_quat())
    assert _quat_matches(got[0], (1.0, 0.0, 0.0, 0.0)) and _quat_matches(got[1], _Z90)

    with pytest.raises(Exception, match="env-specific quat for fixed links"):
        pose_env.entities["fixed_box"].set_quat(per_env)


def _spin(pose_env):
    """Spin the aligned mesh about a tilted axis, so the transport term ``omega x d`` is non-zero."""
    hand = pose_env.entities["hand"]
    hand.set_dofs_vel([0.3, -0.2, 0.1, 4.0, -3.0, 5.0])  # linear DOFs 0-2, angular 3-5
    pose_env.scene.step()
    return hand


def test_velocity_getters_report_the_point_the_pose_getters_report(pose_env):
    """Genesis' default is the authored origin; Eden pins the solver one.

    Upstream default flip: https://github.com/Genesis-Embodied-AI/Genesis/pull/3250.
    """
    hand = _spin(pose_env)
    base = hand.get_link(hand.base_link_name).idx_local
    authored = to_np(hand.get_links_vel(relative=True))
    offset = to_np(hand.get_links_pos()) - to_np(hand.get_links_pos(relative=True))
    # The authored-origin velocity is the default reading minus omega x d.
    expected = to_np(hand.get_links_vel()) - np.cross(to_np(hand.get_links_ang()), offset)
    np.testing.assert_allclose(authored, expected, atol=1e-4)
    assert not np.allclose(to_np(hand.get_links_vel()), authored, atol=1e-4)  # the frames genuinely differ
    np.testing.assert_allclose(to_np(hand.get_vel()), to_np(hand.get_links_vel())[:, base], atol=1e-6)
    # The Genesis-level reading is the oracle for which frame Eden pins (private: no other spelling).
    np.testing.assert_allclose(
        to_np(hand.get_links_vel()), to_np(hand._entity.get_links_vel(relative=False)), atol=1e-6
    )
    np.testing.assert_allclose(
        to_np(hand.get_links_acc()), to_np(hand._entity.get_links_acc(relative=False)), atol=1e-6
    )


def test_the_free_joint_angular_dofs_are_in_the_body_frame(pose_env):
    """Linear DOFs 0-2 are world frame, angular DOFs 3-5 body frame: a world omega must be rotated in.

    Writing a world angular velocity straight into DOFs 3-5 is exact on an upright, un-yawed base, so
    the error only shows on tilted poses; this pins the convention the reset and push terms rely on.
    """
    box = pose_env.entities["box"]
    box.set_quat(torch.tensor(np.tile(euler_quat((30.0, -20.0, 45.0)), (2, 1)), device=pose_env.device))
    q = box.get_quat()
    omega = torch.tensor([[0.7, -0.4, 0.9]] * 2, device=pose_env.device)
    zero = torch.zeros(2, 3, device=pose_env.device)

    box.set_dofs_vel(torch.cat([zero, gu.inv_transform_by_quat(omega, q)], dim=1))
    torch.testing.assert_close(box.get_ang(), omega, atol=1e-4, rtol=0)

    box.set_dofs_vel(torch.cat([zero, omega], dim=1))
    torch.testing.assert_close(box.get_ang(), gu.transform_by_quat(omega, q), atol=1e-4, rtol=0)
    assert not torch.allclose(box.get_ang(), omega, atol=1e-2)


# ═════════════════════════════════════════════════════════════════════════════════════════════════
# attach_to: the mounting transform and the welded child's index maps
# (https://github.com/Genesis-Embodied-AI/Genesis/pull/3041)
# ═════════════════════════════════════════════════════════════════════════════════════════════════

# 45 deg about y, deliberately not axis-aligned with the parent link frame.
_MOUNT_POS = (0.10, 0.20, 0.30)
_MOUNT_QUAT = (math.cos(math.pi / 8), 0.0, math.sin(math.pi / 8), 0.0)
_ARM_POSE = [0.3, -0.5, 0.2, -1.8, 0.4, 2.5, 0.7]


@pytest.fixture(scope="module")
def attach_env():
    """An arm with three welded children: two boxes on link6 and a floating MJCF gripper on the flange.

    ``box`` mounts at its own default root pose, ``offset_box`` at the explicit 4-tuple offset from an
    identity default. ``free_cube`` is declared between the arm and its welded gripper, which Genesis'
    attach forbids unless the env instantiates a parent and its welded subtree consecutively.
    """
    gravity_free = RigidMaterialOptions(gravity_compensation=1.0)
    mount_box = dict(size=(0.04, 0.04, 0.04), fixed=True, batch_fixed_verts=True, collision=False)
    env = build_env(
        _config(
            plane=PlaneOptions(),
            robot=FrankaResearch3(material=gravity_free),
            box=BoxOptions(**mount_box, default_root_pos=_MOUNT_POS, default_root_quat=_MOUNT_QUAT),
            free_cube=BoxOptions(size=(0.04, 0.04, 0.04), default_root_pos=(0.5, 0.0, 0.1)),
            gripper=Robotiq2f85(
                is_fixed_base=False,
                batch_fixed_verts=True,
                material=gravity_free,
                default_root_pos=_MOUNT_POS,
                default_root_quat=_MOUNT_QUAT,
            ),
            offset_box=BoxOptions(**mount_box),
            attachments_dict={
                "box": ("robot", "link6"),
                "gripper": ("robot", "attachment"),
                "offset_box": ("robot", "link6", _MOUNT_POS, _MOUNT_QUAT),
            },
        )
    )
    _pose_arm(env, _ARM_POSE)  # off the neutral pose, so a wrong parent frame cannot pass by luck
    return env


def _pose_arm(env, qpos):
    env.entities["robot"].set_qpos(
        torch.tensor([qpos], device=env.device), qs_idx_local=torch.arange(7, device=env.device)
    )


def _assert_mounted(parent, parent_link: str, child, child_link: str) -> None:
    """The child link's world pose is ``parent_link_pose ∘ (_MOUNT_POS, _MOUNT_QUAT)``."""
    parent_link, child_link = parent.get_link(parent_link), child.get_link(child_link)
    parent_pos, parent_quat = to_np(parent_link.get_pos()).reshape(-1), to_np(parent_link.get_quat()).reshape(-1)
    child_pos, child_quat = to_np(child_link.get_pos()).reshape(-1), to_np(child_link.get_quat()).reshape(-1)
    np.testing.assert_allclose(
        child_pos, parent_pos + gu.transform_by_quat(np.asarray(_MOUNT_POS), parent_quat), atol=1e-5
    )
    assert_quat_close(child_quat, gu.transform_quat_by_quat(np.asarray(_MOUNT_QUAT), parent_quat), atol=1e-5)


@pytest.mark.parametrize(
    "child, parent_link",
    [("box", "link6"), ("offset_box", "link6"), ("gripper", "attachment")],
    ids=["primitive-default-root", "explicit-offset", "floating-mjcf"],
)
def test_a_welded_child_mounts_at_the_full_transform_and_follows_the_arm(attach_env, child, parent_link):
    """Rotation included: the floating MJCF child used to keep only the translation."""
    robot, entity = attach_env.entities["robot"], attach_env.entities[child]
    child_link = entity.base_link_name
    _assert_mounted(robot, parent_link, entity, child_link)
    _pose_arm(attach_env, [-0.4, 0.3, -0.6, -1.2, 0.9, 1.8, -0.5])
    try:
        _assert_mounted(robot, parent_link, entity, child_link)
    finally:
        _pose_arm(attach_env, _ARM_POSE)


def test_welded_children_are_instantiated_right_after_their_parent(attach_env):
    order = list(attach_env.entities)
    robot = order.index("robot")
    assert order[robot + 1 : robot + 4] == ["box", "gripper", "offset_box"], order


def test_a_welded_floating_child_has_no_base_left_in_its_index_maps(attach_env):
    """``attach`` removes the child's root joint; stale floating-base maps indexed out of bounds."""
    gripper = attach_env.entities["gripper"]
    assert torch.equal(gripper.qs_idx_local, gripper.dofs_idx_local)
    assert int(gripper.qs_idx_local.max()) < gripper.n_qs
    gripper.set_qpos(gripper.get_qpos())  # raised (CPU IndexError / CUDA assert) with the stale map
    # Private, but the only view: Genesis cannot take a Jacobian of a welded child to observe it through.
    assert torch.equal(gripper._jac_perm, gripper.dofs_idx_local)


@pytest.mark.parametrize(
    "name, is_attached, has_immobile_base",
    [("robot", False, True), ("box", True, True), ("gripper", True, True), ("free_cube", False, False)],
)
def test_has_immobile_base_covers_fixed_and_welded_entities(attach_env, name, is_attached, has_immobile_base):
    """A welded floating child reports ``is_fixed_base=False`` yet owns no base DOFs."""
    entity = attach_env.entities[name]
    assert (entity.is_attached, entity.has_immobile_base) == (is_attached, has_immobile_base)


def test_attaching_a_heterogeneous_child_raises_at_construction():
    """It would mount every env at one shared pose and drop the per-variant offsets."""
    with pytest.raises(NotImplementedError, match="heterogeneous"):
        EnvBase.from_config(
            _config(
                num_envs=2,
                plane=PlaneOptions(),
                robot=FrankaResearch3(),
                widget=GroupedEntityOptions(
                    grouped_entities=[
                        BoxOptions(size=(0.04, 0.04, 0.04), default_root_pos=(0.10, 0.0, 0.0)),
                        BoxOptions(size=(0.04, 0.04, 0.04), default_root_pos=(0.20, 0.0, 0.0)),
                    ],
                ),
                attachments_dict={"widget": ("robot", "attachment")},
            ),
            show_viewer=False,
        )


# ═════════════════════════════════════════════════════════════════════════════════════════════════
# DOF and link *info* getters under both batching modes
#
# Genesis batches per-DOF / per-link model parameters only when ``batch_dofs_info`` /
# ``batch_links_info`` is set, and raises on ``envs_idx`` when not. Eden's getters always return a
# leading env axis and accept every ``envs_idx`` form. ``KinematicSolver`` never batches, whatever
# the env option, so the kinematic ghost is a third configuration.
# ═════════════════════════════════════════════════════════════════════════════════════════════════

NUM_ENVS = 4
SCALAR_GETTERS = [
    "get_dofs_kp",
    "get_dofs_kd",
    "get_dofs_stiffness",
    "get_dofs_damping",
    "get_dofs_armature",
    "get_dofs_frictionloss",
    "get_dofs_invweight",
]
PAIR_GETTERS = ["get_dofs_limit", "get_dofs_force_range"]
ENVS_IDX_FORMS = pytest.mark.parametrize(
    "envs_idx, rows",
    [(None, NUM_ENVS), (0, 1), (slice(1, 3), 2), ([0, 2], 2), (torch.tensor([True, False, True, False]), 2)],
    ids=["all", "int", "slice", "list", "bool-mask"],
)


_TABLE_SIZE, _TABLE_TOP, _CUBE_HALF = (0.8, 0.8, 0.4), 0.4, 0.025


def _info_env(batched: bool) -> EnvBase:
    return build_env(
        EdenConfig(
            env_options=EnvOptions(num_envs=NUM_ENVS, batch_dofs_info=batched, batch_links_info=batched),
            scene_options=SceneOptions(
                robot=FrankaHand(default_root_pos=(0.0, 0.0, 0.5)),
                # Articulated like the robot, so it takes the DOF-property path, but in the kinematic solver.
                ghost=FrankaHand(default_root_pos=(1.0, 0.0, 0.5), material=KinematicMaterialOptions()),
                cube=BoxOptions(size=(0.2, 0.2, 0.2), default_root_pos=(-1.0, 0.0, 0.5)),
                # A support-enabled table and a small cube for place_on_to.
                table=BoxOptions(
                    size=_TABLE_SIZE, fixed=True, default_root_pos=(0.0, 3.0, 0.2), is_support_enabled=True
                ),
                placed=BoxOptions(size=(2 * _CUBE_HALF,) * 3, default_root_pos=(0.0, 3.0, 0.6)),
            ),
        )
    )


@pytest.fixture(scope="module")
def batched_env() -> EnvBase:
    return _info_env(batched=True)


@pytest.fixture(scope="module")
def unbatched_env() -> EnvBase:
    return _info_env(batched=False)


@pytest.fixture(params=["batched", "unbatched"])
def robot(request):
    return request.getfixturevalue(f"{request.param}_env").entities["robot"]


@ENVS_IDX_FORMS
@pytest.mark.parametrize("getter", SCALAR_GETTERS)
def test_dof_info_getters_carry_the_env_axis_for_every_envs_idx_form(robot, getter, envs_idx, rows):
    """Including under ``batch_dofs_info=False``, where raw Genesis rejects ``envs_idx``."""
    assert getattr(robot, getter)(envs_idx=envs_idx).shape == (rows, robot.num_dofs)


@pytest.mark.parametrize("envs_idx, rows", [(None, NUM_ENVS), ([0, 2], 2)])
@pytest.mark.parametrize("getter", PAIR_GETTERS)
def test_bound_getters_carry_the_env_axis_on_both_bounds(robot, getter, envs_idx, rows):
    lower, upper = getattr(robot, getter)(envs_idx=envs_idx)
    assert lower.shape == upper.shape == (rows, robot.num_dofs)
    assert (lower <= upper).all()


@pytest.mark.parametrize("getter", SCALAR_GETTERS)
def test_a_dof_subset_is_returned_in_the_requested_order(robot, getter):
    full = getattr(robot, getter)()
    reversed_idx = torch.arange(robot.num_dofs - 1, -1, -1, device=robot.device)
    assert torch.equal(getattr(robot, getter)(reversed_idx), full.flip(dims=(1,)))


@pytest.mark.parametrize("getter", [*SCALAR_GETTERS, "get_links_inertial_mass"])
def test_info_getters_return_a_copy_not_a_solver_view(robot, getter):
    """Callers write into these in place (``EnvelopeClip.reset`` does), which must not reach the solver."""
    out = getattr(robot, getter)()
    pristine = out.clone()
    out += 100.0
    assert torch.equal(getattr(robot, getter)(), pristine)


@pytest.mark.parametrize("getter", SCALAR_GETTERS)
def test_unbatched_dof_info_rows_are_one_shared_value(unbatched_env, getter):
    out = getattr(unbatched_env.entities["robot"], getter)()
    assert torch.equal(out, out[:1].expand_as(out))


def test_batched_dof_info_resolves_genuine_per_env_values(batched_env):
    robot = batched_env.entities["robot"]
    base = robot.get_dofs_kp()
    bumped = base.clone()
    bumped[1] += 7.0
    robot.set_dofs_kp(bumped)
    try:
        assert torch.allclose(robot.get_dofs_kp(), bumped)
        assert torch.allclose(robot.get_dofs_kp(envs_idx=1), bumped[1:2])
        assert torch.allclose(robot.get_dofs_kp(envs_idx=[0, 2]), base[[0, 2]])
    finally:
        robot.set_dofs_kp(base)


@pytest.mark.parametrize("env_name", ["batched_env", "unbatched_env"])
def test_a_kinematic_entity_ignores_batch_dofs_info(request, env_name):
    """Keying the wrapper off the env option would raise here under ``batch_dofs_info=True`` -- in post_build."""
    ghost = request.getfixturevalue(env_name).entities["ghost"]
    assert ghost.is_kinematic
    lower, upper = ghost.get_dofs_limit()
    assert lower.shape == (NUM_ENVS, ghost.num_dofs)
    assert torch.equal(lower, lower[:1].expand_as(lower)) and torch.equal(upper, upper[:1].expand_as(upper))


def test_post_build_defaults_agree_with_the_solver(robot):
    """``post_build`` seeds every ``default_dofs_*`` from ``envs_idx=0`` in either mode."""
    for attr, getter in [
        ("default_dofs_stiffness", "get_dofs_stiffness"),
        ("default_dofs_damping", "get_dofs_damping"),
        ("default_dofs_frictionloss", "get_dofs_frictionloss"),
    ]:
        cached = getattr(robot, attr)
        assert cached.ndim == 2, attr
        assert torch.allclose(cached[0], getattr(robot, getter)(envs_idx=0)[0]), attr
    # post_build may fold kd * substep_dt into the solver armature, never below the cached default.
    assert (robot.get_dofs_armature(envs_idx=0)[0] >= robot.default_dofs_armature[0]).all()
    lower, upper = robot.get_dofs_limit(envs_idx=0)
    assert torch.allclose(robot.default_dofs_pos_limits[0, :, 0], lower[0])
    assert torch.allclose(robot.default_dofs_pos_limits[0, :, 1], upper[0])


@pytest.mark.parametrize("getter", ["get_dofs_kp", "get_dofs_armature", "get_dofs_damping", "get_dofs_limit"])
def test_the_default_dof_selector_reads_row_major_like_an_index_tensor(robot, getter):
    """``dofs_idx_local=None`` reaches Genesis as a ``range``, which Genesis lays out env-minor.

    ``rand_like`` keeps strides, so a column-major return would move observation noise onto other elements.
    """

    def as_tuple(out):
        return out if isinstance(out, tuple) else (out,)

    default, by_tensor = getattr(robot, getter)(), getattr(robot, getter)(dofs_idx_local=robot.dofs_idx_local)
    for got, want in zip(as_tuple(default), as_tuple(by_tensor)):
        assert got.shape == (NUM_ENVS, robot.num_dofs) and got.is_contiguous()
        assert torch.equal(got, want)


def test_place_on_to_rests_the_object_on_the_support_surface(batched_env):
    """Support extraction, AABB kernel, broad phase, occupancy filter, sample, set_pos, in every env.

    The one Eden path into Genesis' broad phase and packed solver state (repacked by
    https://github.com/Genesis-Embodied-AI/Genesis/pull/3060), so it doubles as a canary for upstream collider
    refactors.
    """
    from eden.managers.terms.events.placement import uniform_sampler

    cube, table_xy = batched_env.entities["placed"], torch.tensor([0.0, 3.0], device=batched_env.device)
    cube.place_on_to(batched_env.entities["table"], sampler=uniform_sampler)
    pos = cube.get_pos()
    assert torch.all(pos[:, 2] > _TABLE_TOP) and torch.all(pos[:, 2] < _TABLE_TOP + 3 * _CUBE_HALF), pos
    assert torch.all((pos[:, :2] - table_xy).abs() < _TABLE_SIZE[0] / 2), pos
    for _ in range(20):  # settled, not tunnelling through or bouncing off
        batched_env.scene.step()
    assert torch.all(cube.get_pos()[:, 2] > _TABLE_TOP)


# ── per-link mass ─────────────────────────────────────────────────────────────────────────────────


def test_per_link_masses_decompose_the_total_mass(unbatched_env):
    """``get_mass()`` is the total over links not fixed to the world; those links carry real mass.

    A link the world carries and that states no inertial reads exactly 0
    (https://github.com/Genesis-Embodied-AI/Genesis/pull/3311 stopped fabricating a floor), like the FrankaHand's
    synthetic ``world`` link.
    """
    for name in ("robot", "cube"):
        entity = unbatched_env.entities[name]
        moving = torch.tensor([not link.is_fixed for link in entity.links], device=entity.device)
        mass = entity.get_links_inertial_mass()
        assert mass.shape == (NUM_ENVS, entity.num_links)
        assert (mass[:, moving] > 0.0).all() and (mass >= 0.0).all()
        total = entity.get_mass()
        assert torch.allclose(mass[:, moving].sum(dim=1), total.expand(NUM_ENVS), rtol=1e-5)


def test_per_link_mass_is_not_the_total_broadcast(unbatched_env):
    """The regression ``RandomizeLinkMassScale`` hit: it indexed ``get_mass()`` as if it were per link."""
    robot = unbatched_env.entities["robot"]
    mass = robot.get_links_inertial_mass()
    assert not torch.allclose(mass, robot.get_mass().expand_as(mass))
    assert mass[0].min() < mass[0].max()


def test_a_link_subset_is_returned_in_the_requested_order(unbatched_env):
    robot = unbatched_env.entities["robot"]
    reversed_idx = torch.arange(robot.num_links - 1, -1, -1, device=robot.device)
    assert torch.equal(robot.get_links_inertial_mass(reversed_idx), robot.get_links_inertial_mass().flip(dims=(1,)))
    invweight = robot.get_links_invweight()
    assert invweight.shape == (NUM_ENVS, robot.num_links, 2) and (invweight >= 0.0).all()
    assert torch.equal(robot.get_links_invweight([1]), invweight[:, 1:2])


@ENVS_IDX_FORMS
def test_unbatched_link_mass_answers_every_envs_idx_form_with_the_shared_row(unbatched_env, envs_idx, rows):
    robot = unbatched_env.entities["robot"]
    expected = robot.get_links_inertial_mass()[:1].expand(rows, robot.num_links)
    assert torch.equal(robot.get_links_inertial_mass(envs_idx=envs_idx), expected)


def test_batched_link_mass_resolves_genuine_per_env_values(batched_env):
    robot = batched_env.entities["robot"]
    ls_idx = torch.arange(robot.num_links, device=robot.device)
    base = robot.get_links_inertial_mass()
    bumped = base.clone()
    bumped[2] += 1.0
    robot._entity.set_links_mass(bumped, ls_idx)  # a Genesis-level write, as a DR term would make
    try:
        assert torch.allclose(robot.get_links_inertial_mass(), bumped)
        assert torch.allclose(robot.get_links_inertial_mass(envs_idx=2), bumped[2:3])
        assert torch.allclose(robot.get_links_inertial_mass(envs_idx=[0, 1, 3]), base[[0, 1, 3]])
    finally:
        robot._entity.set_links_mass(base, ls_idx)


@pytest.fixture
def batched_robot(batched_env):
    """The batched gripper, with its mass, COM and friction shifts restored to nominal afterwards."""
    robot = batched_env.entities["robot"]
    yield robot
    links, n = _all_links(robot), robot.num_links
    robot.set_mass_shift(torch.zeros(NUM_ENVS, n, device=robot.device), ls_idx_local=links)
    robot.set_COM_shift(torch.zeros(NUM_ENVS, n, 3, device=robot.device), ls_idx_local=links)
    robot.set_friction_ratio(torch.ones(NUM_ENVS, n, device=robot.device), ls_idx_local=links)


def _all_links(entity):
    return torch.arange(entity.num_links, device=entity.device)


def test_mass_shift_moves_the_live_mass_and_reads_back_the_difference(batched_robot):
    """Genesis holds one mass channel: ``set_mass_shift`` writes ``nominal + shift`` as an absolute value."""
    nominal = batched_robot.get_links_inertial_mass()
    shift = torch.full((NUM_ENVS, batched_robot.num_links), 0.25, device=batched_robot.device)
    batched_robot.set_mass_shift(shift, ls_idx_local=_all_links(batched_robot))
    assert torch.allclose(batched_robot.get_links_inertial_mass(), nominal + shift)
    assert torch.allclose(batched_robot.get_mass_shift(), shift)

    batched_robot.set_mass_shift(torch.zeros_like(shift), ls_idx_local=_all_links(batched_robot))
    assert torch.allclose(batched_robot.get_links_inertial_mass(), nominal)


def test_a_bool_mask_mass_shift_lands_on_exactly_the_masked_envs(batched_robot):
    mask = torch.tensor([True, False, True, False], device=batched_robot.device)
    shift = torch.full((2, batched_robot.num_links), 0.5, device=batched_robot.device)
    batched_robot.set_mass_shift(shift, ls_idx_local=_all_links(batched_robot), envs_idx=mask)
    got = batched_robot.get_mass_shift()
    assert torch.allclose(got[[0, 2]], shift) and (got[[1, 3]] == 0.0).all()


def test_unrandomized_links_read_back_as_nominal(batched_robot):
    n, device = batched_robot.num_links, batched_robot.device
    assert torch.equal(batched_robot.get_friction_ratio(), torch.ones(NUM_ENVS, n, device=device))
    assert torch.equal(batched_robot.get_COM_shift(), torch.zeros(NUM_ENVS, n, 3, device=device))


def test_the_mass_shift_getter_selects_bool_mask_rows(batched_robot):
    """The old shadow buffer answered a mask with rows 0 and 1."""
    hand_link = torch.tensor([batched_robot.get_link("hand").idx_local], device=batched_robot.device)
    batched_robot.set_mass_shift(
        torch.tensor([[1.0], [2.0], [3.0], [4.0]], device=batched_robot.device), ls_idx_local=hand_link
    )
    mask = torch.tensor([True, False, True, False], device=batched_robot.device)
    torch.testing.assert_close(
        batched_robot.get_mass_shift(hand_link, envs_idx=mask).cpu(), torch.tensor([[1.0], [3.0]])
    )


# ``hand`` is fixed to the world on the FrankaHand, yet holds 0.73 kg; Genesis checks every selection form but
# a tensor on the host, counts such a link as massless and raised "no mass to set" for them.
FIXED_LINK_FORMS = pytest.mark.parametrize(
    "form",
    [lambda i: [i], lambda i: (i,), int, lambda i: range(i, i + 1), lambda i: torch.tensor([i])],
    ids=["list", "tuple", "int", "range", "tensor"],
)


@FIXED_LINK_FORMS
def test_a_mass_shift_on_a_world_fixed_link_writes_whatever_form_the_selection_takes(batched_robot, form):
    """``find_named_links_idx_local`` hands back plain lists, so a caller that forwards one passes this form."""
    hand = batched_robot.get_link("hand").idx_local
    batched_robot.set_mass_shift(torch.full((NUM_ENVS, 1), 0.5, device=batched_robot.device), ls_idx_local=form(hand))
    expected = torch.zeros(NUM_ENVS, batched_robot.num_links, device=batched_robot.device)
    expected[:, hand] = 0.5
    torch.testing.assert_close(batched_robot.get_mass_shift(), expected)


@FIXED_LINK_FORMS
def test_a_com_shift_on_a_world_fixed_link_writes_whatever_form_the_selection_takes(batched_robot, form):
    """Genesis has no host check on the COM write, but it shares the mass write's link selector."""
    hand = batched_robot.get_link("hand").idx_local
    shift = torch.tensor([0.01, -0.02, 0.03], device=batched_robot.device)
    batched_robot.set_COM_shift(shift.expand(NUM_ENVS, 1, 3).contiguous(), ls_idx_local=form(hand))
    expected = torch.zeros(NUM_ENVS, batched_robot.num_links, 3, device=batched_robot.device)
    expected[:, hand] = shift
    torch.testing.assert_close(batched_robot.get_COM_shift(), expected)


def test_a_mass_shift_on_a_fixed_prop_writes_from_a_list(batched_env):
    """A ``fixed=True`` box has no moving link at all: the host check refused its only one outright."""
    table = batched_env.entities["table"]
    try:
        table.set_mass_shift(torch.full((NUM_ENVS, 1), 0.5, device=table.device), ls_idx_local=[0])
        assert (table.get_mass_shift() == 0.5).all()
    finally:
        table.set_mass_shift(torch.zeros(NUM_ENVS, 1, device=table.device), ls_idx_local=None)


def test_a_per_env_com_shift_round_trips_the_xyz_axis(batched_robot):
    hand_link = torch.tensor([batched_robot.get_link("hand").idx_local], device=batched_robot.device)
    shift = torch.tensor([[[0.01, -0.02, 0.03]]], device=batched_robot.device).expand(NUM_ENVS, 1, 3).contiguous()
    batched_robot.set_COM_shift(shift, ls_idx_local=hand_link)
    torch.testing.assert_close(batched_robot.get_COM_shift(hand_link), shift)


def test_the_nominal_is_captured_at_the_first_shift_call(batched_robot):
    """A Genesis-level write before the first shift call folds into the nominal; one after reads as shift."""
    ls_idx = _all_links(batched_robot)
    base = batched_robot._entity.get_links_mass().clone()
    batched_robot._nominal_links_inertials_cache = None  # private: the capture point is the thing under test
    batched_robot._entity.set_links_mass(base + 1.0, ls_idx)
    try:
        assert (batched_robot.get_mass_shift() == 0.0).all()
        shift = torch.full_like(base, 0.25)
        batched_robot.set_mass_shift(shift, ls_idx_local=ls_idx)
        assert torch.allclose(batched_robot.get_mass_shift(), shift)
        assert torch.allclose(batched_robot.get_links_inertial_mass(), base + 1.0 + shift)
    finally:
        batched_robot._entity.set_links_mass(base, ls_idx)
        batched_robot._nominal_links_inertials_cache = None


def test_a_per_env_mass_shift_needs_batched_link_info(unbatched_env):
    robot = unbatched_env.entities["robot"]
    shift = torch.full((NUM_ENVS, robot.num_links), 0.25, device=robot.device)
    with pytest.raises(gs.GenesisException, match="non-batched links info"):
        robot.set_mass_shift(
            shift, ls_idx_local=_all_links(robot), envs_idx=torch.arange(NUM_ENVS, device=robot.device)
        )


def test_unbatched_com_shift_reads_back_env_broadcast_and_takes_a_shared_write(unbatched_env):
    robot = unbatched_env.entities["robot"]
    assert torch.equal(robot.get_COM_shift(), torch.zeros(NUM_ENVS, robot.num_links, 3, device=robot.device))
    assert robot.get_COM_shift(envs_idx=[1, 2]).shape == (2, robot.num_links, 3)
    shift = torch.full((robot.num_links, 3), 0.01, device=robot.device)
    robot.set_COM_shift(shift, ls_idx_local=_all_links(robot))
    try:
        assert torch.allclose(robot.get_COM_shift(), shift.expand(NUM_ENVS, -1, -1))
    finally:
        robot.set_COM_shift(torch.zeros_like(shift), ls_idx_local=_all_links(robot))


# ═════════════════════════════════════════════════════════════════════════════════════════════════
# default_dofs_* on a real robot: the cached defaults are the setter inputs
# ═════════════════════════════════════════════════════════════════════════════════════════════════

_ANKLE, _ANKLE_LIMITS = "left_ankle_pitch_joint", (-0.25, 0.25)
_DOFS_ATTRS = [
    ("default_dofs_pos", "get_dofs_pos", "set_dofs_pos"),
    ("default_dofs_vel", "get_dofs_vel", "set_dofs_vel"),
    ("default_dofs_stiffness", "get_dofs_stiffness", "set_dofs_stiffness"),
    ("default_dofs_damping", "get_dofs_damping", "set_dofs_damping"),
    ("default_dofs_armature", "get_dofs_armature", "set_dofs_armature"),
    ("default_dofs_kp", "get_dofs_kp", "set_dofs_kp"),
    ("default_dofs_kd", "get_dofs_kd", "set_dofs_kd"),
]


@pytest.fixture(scope="module")
def g1():
    """G1 with every ``default_dofs_*`` map set, so ``_init_dofs_property`` takes the user-map path for all."""
    robot = G1Dofs29(
        default_dofs_vel={"*": 0.0},
        default_dofs_stiffness={"*": 1.0},
        default_dofs_damping={"*": 0.5},
        default_dofs_pos_limits={_ANKLE: _ANKLE_LIMITS},
    )
    config = EdenConfig(
        env_options=EnvOptions(num_envs=2, sim_dt=0.005, decimation=4), scene_options=SceneOptions(robot=robot)
    )
    return build_env(config).entities["robot"]


@pytest.mark.parametrize("attr, getter, setter", _DOFS_ATTRS)
def test_mutating_a_cached_default_does_not_reach_the_solver(g1, attr, getter, setter):
    """``_init_dofs_property`` hands the cached tensor straight to the setter, so no setter may keep it."""
    cached = getattr(g1, attr)
    snapshot = cached.clone()

    def push():  # exactly as post_build hands it over: a broadcast row goes in unbatched
        getattr(g1, setter)(cached[0] if cached.shape[0] == 1 else cached)

    push()
    before = getattr(g1, getter)().clone()
    cached[0, 0] = 1234.5
    try:
        assert torch.equal(getattr(g1, getter)(), before), f"{setter} retained a reference to its input"
    finally:
        cached.copy_(snapshot)
        push()


def test_the_cached_defaults_do_not_share_storage(g1):
    ptrs = [getattr(g1, attr).data_ptr() for attr, _, _ in _DOFS_ATTRS]
    assert len(set(ptrs)) == len(ptrs)


def test_a_pos_limits_override_moves_edens_range_and_its_soft_range(g1):
    """The one range-per-DOF entry: its tuple path used to raise ``can't assign a tuple to a torch.FloatTensor``."""
    parsed = g1.default_dofs_pos_limits
    i = g1.dofs_name.index(_ANKLE)
    assert parsed.shape == (1, g1.num_dofs, 2)
    assert parsed[0, i].tolist() == pytest.approx(list(_ANKLE_LIMITS))
    others = [j for j in range(g1.num_dofs) if j != i]
    asset_lo, asset_hi = g1.get_dofs_limit()
    assert torch.allclose(parsed[0, others, 0], asset_lo[0, others])
    assert torch.allclose(parsed[0, others, 1], asset_hi[0, others])
    assert g1.soft_dofs_pos_limits[0, i].tolist() == pytest.approx([-0.225, 0.225])  # 0.9 about the midpoint


def test_the_default_state_selectors_are_ranges_read_like_the_index_tensor(g1):
    """The explicit reads go through the Genesis entity, the only index-tensor reference there is."""
    assert (g1._genesis_dofs_idx, g1._genesis_qs_idx) == (range(6, 35), range(0, 36))  # 29 DOFs after the free base
    for default, explicit in (
        (g1.get_dofs_pos(), g1._entity.get_dofs_position(g1.dofs_idx_local)),
        (g1.get_dofs_vel(), g1._entity.get_dofs_velocity(g1.dofs_idx_local)),
        (g1.get_qpos(), g1._entity.get_qpos(g1.qs_idx_local)),
        (g1.get_links_pos(ls_idx_local=range(2, 5)), g1._entity.get_links_pos(torch.tensor([2, 3, 4]))),
    ):
        assert torch.equal(default, explicit)
        assert default.is_contiguous() and default.stride() == explicit.stride()


_SPEC_PROPS = [
    "FULL_TORQUE_SPEED",
    "NO_LOAD_SPEED",
    "DRIVING_TORQUE_LIMIT",
    "BRAKING_TORQUE_LIMIT",
    "STATIC_FRICTION",
    "DYNAMIC_FRICTION",
    "FRICTION_ACTIVATION_SPEED",
]


@pytest.mark.parametrize(
    "dofs_name, spec_cls",
    [
        (["left_knee_joint", "right_knee_joint"], Unitree7520_22),
        (["left_hip_pitch_joint", "right_hip_pitch_joint"], Unitree7520_14),
        # The ankles run through a parallel 4-bar linkage, hence the doubled spec.
        (["left_ankle_pitch_joint", "right_ankle_pitch_joint"], Unitree5020_16_DOUBLE),
        (["left_ankle_roll_joint", "right_ankle_roll_joint"], Unitree5020_16_DOUBLE),
    ],
    ids=["knee", "hip-pitch", "ankle-pitch", "ankle-roll"],
)
def test_each_joint_carries_the_spec_of_its_motor(g1, dofs_name, spec_cls):
    """``dofs_spec`` glob patterns resolved per DOF: the values the actuator modifiers read.

    Ground truth named here rather than re-derived from ``dofs_spec``. Private: the entity keeps the
    resolved spec only as per-DOF tensors, which is where the modifiers read it.
    """
    for dof_name, idx in zip(dofs_name, g1.find_named_dofs(dofs_name).columns.tolist()):
        for prop in _SPEC_PROPS:
            assert getattr(g1, f"_{prop.lower()}")[0, idx].item() == pytest.approx(getattr(spec_cls, prop), rel=1e-5)


# ═════════════════════════════════════════════════════════════════════════════════════════════════
# post_build on hand-written MJCF assets: free joints, soft limits, joint defaults, the fixed-base
# declaration and per-geom friction. One scene, torsional and rolling friction switched on.
# ═════════════════════════════════════════════════════════════════════════════════════════════════

# Free joints named outside any convention; the gizmo's hinge matches a ``*_base_joint`` regex, so a
# name-based rule would give it the free DOFs' stability damping.
_FLOATER_MJCF = """<mujoco model="floater"><worldbody>
  <body name="bob" pos="0.2 0.2 0.4"><freejoint name="weird_floaty_thing"/>
    <geom type="box" size="0.05 0.05 0.02" density="500"/></body>
</worldbody></mujoco>"""
_GIZMO_MJCF = """<mujoco model="gizmo"><compiler angle="radian" autolimits="true"/><worldbody>
  <body name="puck" pos="0.6 0.6 0.3"><freejoint name="another_floaty_thing"/>
    <geom type="box" size="0.05 0.05 0.02" density="500"/>
    <body name="turntable" pos="0 0 0.05">
      <joint name="turntable_base_joint" axis="0 0 1" range="-3 3" damping="0.05"/>
      <geom type="cylinder" size="0.03 0.01" density="500"/></body></body>
</worldbody></mujoco>"""
_TWINS_MJCF = """<mujoco model="twins"><worldbody>
  <body name="alpha" pos="1.0 1.0 0.3"><freejoint name="first_free"/>
    <geom type="box" size="0.05 0.05 0.02" density="500"/></body>
  <body name="beta" pos="1.3 1.0 0.3"><freejoint name="second_free"/>
    <geom type="box" size="0.05 0.05 0.02" density="500"/></body>
</worldbody></mujoco>"""

# Soft limits: every DOF bounded / a ball joint and a range-less hinge / one of each.
_BOUNDED_MJCF = """<mujoco model="bounded"><compiler angle="radian" autolimits="true"/><worldbody>
  <body name="chassis" pos="0 0 0.5"><freejoint name="chassis_float"/>
    <geom type="box" size="0.1 0.1 0.05" density="500"/>
    <body name="arm_a" pos="0.1 0 0"><joint name="alpha_joint" axis="0 1 0" range="-1.2 1.2"/>
      <geom type="capsule" size="0.02" fromto="0 0 0 0.1 0 0" density="500"/>
      <body name="arm_b" pos="0.1 0 0"><joint name="beta_joint" axis="0 1 0" range="-1.2 1.2"/>
        <geom type="capsule" size="0.02" fromto="0 0 0 0.1 0 0" density="500"/></body></body></body>
</worldbody></mujoco>"""
_FLOPPY_MJCF = """<mujoco model="floppy"><compiler angle="radian" autolimits="true"/><worldbody>
  <body name="anchor" pos="-0.6 -0.6 0.4"><geom type="box" size="0.04 0.04 0.02" density="500"/>
    <body name="socket" pos="0 0 0.05"><joint name="ball_joint" type="ball"/>
      <geom type="capsule" size="0.02" fromto="0 0 0 0.08 0 0" density="500"/>
      <body name="spindle" pos="0.08 0 0"><joint name="free_spin_joint" axis="0 0 1"/>
        <geom type="cylinder" size="0.02 0.01" density="500"/></body></body></body>
</worldbody></mujoco>"""
_MIXED_MJCF = """<mujoco model="mixed"><compiler angle="radian" autolimits="true"/><worldbody>
  <body name="post" pos="0.6 0.6 0.4"><geom type="box" size="0.04 0.04 0.02" density="500"/>
    <body name="hinged" pos="0 0 0.05"><joint name="limited_joint" axis="0 1 0" range="-0.8 0.4"/>
      <geom type="capsule" size="0.02" fromto="0 0 0 0.08 0 0" density="500"/>
      <body name="spinner" pos="0.08 0 0"><joint name="unlimited_joint" axis="0 0 1"/>
        <geom type="cylinder" size="0.02 0.01" density="500"/></body></body></body>
</worldbody></mujoco>"""

# Joint defaults since https://github.com/Genesis-Embodied-AI/Genesis/pull/3328. One body per case, off a free base:
#   hinge_zero authors armature="0" (filled), hinge_kept a nonzero one (kept), slide_bare none
#   (filled), pair carries two hinges (not a single-joint body: left at 0). hinge_zero also states
#   an actuatorfrcrange behind a range-less motor, the only bound its force range can come from.
_DEFAULT_ARMATURE = EntityOptions.model_fields["default_armature"].default
_AUTHORED_ARMATURE, _FRC_RANGE = 0.02, 7.0
_JOINT_DEFAULTS_MJCF = f"""<mujoco model="joint_defaults"><worldbody>
  <body name="base" pos="0 0 1"><freejoint name="root_joint"/>
    <geom name="base_geom" type="box" size="0.05 0.05 0.05" mass="1"/>
    <body name="hinge_zero" pos="0.2 0 0">
      <joint name="hinge_zero" type="hinge" axis="0 0 1" armature="0" actuatorfrcrange="-{_FRC_RANGE} {_FRC_RANGE}"/>
      <geom type="box" size="0.02 0.02 0.02" mass="0.1"/></body>
    <body name="hinge_kept" pos="-0.2 0 0"><joint name="hinge_kept" type="hinge" axis="0 0 1" armature="{_AUTHORED_ARMATURE}"/>
      <geom type="box" size="0.02 0.02 0.02" mass="0.1"/></body>
    <body name="slide_bare" pos="0 0.2 0"><joint name="slide_bare" type="slide" axis="1 0 0"/>
      <geom type="box" size="0.02 0.02 0.02" mass="0.1"/></body>
    <body name="pair" pos="0 -0.2 0"><joint name="pair_a" type="hinge" axis="1 0 0"/>
      <joint name="pair_b" type="hinge" axis="0 1 0"/><geom type="box" size="0.02 0.02 0.02" mass="0.1"/></body>
  </body>
</worldbody><actuator><motor name="hinge_zero" joint="hinge_zero"/></actuator></mujoco>"""
_JOINT_DEFAULTS_DOFS = ["hinge_zero", "hinge_kept", "slide_bare", "pair_a", "pair_b"]

# The fixed-base declaration: an MJCF's base link and ``is_fixed_base`` must agree.
_FREE_BODY_MJCF = """<mujoco model="fixture"><worldbody>
  <body name="base" pos="0 0 0.5"><freejoint/><geom type="box" size="0.05 0.05 0.05" mass="0.5"/></body>
</worldbody></mujoco>"""
_JOINTLESS_MJCF = _FREE_BODY_MJCF.replace("<freejoint/>", "")
# A root body moved by its own joint, at the identity pose so Genesis makes it the base link itself.
_JOINT_ROOT_MJCF = """<mujoco model="fixture"><worldbody>
  <body name="link1"><joint name="j1" type="{joint_type}" axis="0 0 1"/>
    <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.03" mass="1.0"/></body>
</worldbody></mujoco>"""
# A geom directly under <worldbody> keeps Genesis' ``world`` link first, so the free body is not the base.
_FREE_BESIDE_WORLD_GEOM_MJCF = """<mujoco model="fixture"><worldbody>
  <geom name="post" type="box" pos="1 1 0.1" size="0.05 0.05 0.1"/>
  <body name="box" pos="0 0 0.5"><freejoint/><geom type="box" size="0.05 0.05 0.05" mass="0.5"/></body>
</worldbody></mujoco>"""

# Per-geom friction axes. Two links with one geom each; condim stays at MuJoCo's default 3,
# which is what makes the parsed torsional/rolling coefficients zero. They are lengths, hence small.
_TWO_LINK_MJCF = """<mujoco model="two_link"><worldbody>
  <body name="base" pos="0 0 0.5"><freejoint name="root_joint"/><geom name="base_geom" type="box" size="0.05 0.05 0.05"/>
    <body name="pad" pos="0.12 0 0"><joint name="slide" type="slide" axis="1 0 0"/>
      <geom name="pad_geom" type="box" size="0.02 0.02 0.02"/></body></body>
</worldbody></mujoco>"""
_TORSIONAL, _ROLLING, _MATERIAL_TORSIONAL = 0.02, 0.003, 0.001

_FREE_DOFS = list(range(6))
_STABILITY_DAMPING = 2e-4


@pytest.fixture(scope="module")
def mjcf_assets(tmp_path_factory):
    root = tmp_path_factory.mktemp("mjcf_assets")
    files = {
        "floater": _FLOATER_MJCF,
        "gizmo": _GIZMO_MJCF,
        "twins": _TWINS_MJCF,
        "bounded": _BOUNDED_MJCF,
        "floppy": _FLOPPY_MJCF,
        "mixed": _MIXED_MJCF,
        "joint_defaults": _JOINT_DEFAULTS_MJCF,
        "free_body": _FREE_BODY_MJCF,
        "jointless": _JOINTLESS_MJCF,
        "free_beside_world_geom": _FREE_BESIDE_WORLD_GEOM_MJCF,
        "two_link": _TWO_LINK_MJCF,
        **{f"{j}_root": _JOINT_ROOT_MJCF.format(joint_type=j) for j in ("hinge", "slide", "ball")},
    }
    for name, xml in files.items():
        (root / f"{name}.xml").write_text(xml)
    return lambda name: str(root / f"{name}.xml")


def _two_link(asset, **options) -> EntityOptions:
    return EntityOptions(file=asset("two_link"), is_articulated=True, default_root_pos=(1.0, 0.0, 0.5), **options)


def _joint_defaults_rig(asset, **options) -> EntityOptions:
    return EntityOptions(
        file=asset("joint_defaults"),
        is_articulated=True,
        dofs_name=_JOINT_DEFAULTS_DOFS,
        default_root_pos=(0, 0, 1),
        **options,
    )


@pytest.fixture(scope="module")
def mjcf_build(mjcf_assets):
    asset = mjcf_assets
    with captured_eden_logs() as records:
        env = build_env(
            EdenConfig(
                env_options=EnvOptions(num_envs=2, enable_torsional_friction=True, enable_rolling_friction=True),
                scene_options=SceneOptions(
                    plane=PlaneOptions(),
                    box=BoxOptions(size=(0.2, 0.2, 0.2)),
                    floater=EntityOptions(file=asset("floater")),
                    gizmo=EntityOptions(file=asset("gizmo"), is_articulated=True, dofs_name=["turntable_base_joint"]),
                    twins=EntityOptions(file=asset("twins")),
                    bounded=RobotOptions(
                        file=asset("bounded"),
                        dofs_name=["alpha_joint", "beta_joint"],
                        default_dofs_pos={"alpha_joint": 0.3, "beta_joint": -0.2},
                        default_dofs_kp={"*": 5.0},
                        default_dofs_kd={"*": 0.5},
                    ),
                    floppy=EntityOptions(
                        file=asset("floppy"),
                        is_fixed_base=True,
                        is_articulated=True,
                        dofs_name=["ball_joint", "free_spin_joint"],
                    ),
                    mixed=EntityOptions(
                        file=asset("mixed"),
                        is_fixed_base=True,
                        is_articulated=True,
                        dofs_name=["limited_joint", "unlimited_joint"],
                    ),
                    rig=_joint_defaults_rig(asset),
                    rig_unfilled=_joint_defaults_rig(asset, default_armature=0.0),
                    rig_overridden=_joint_defaults_rig(
                        asset,
                        default_dofs_armature={"hinge_zero": 0.0, "hinge_kept": 0.5, "pair_*": 0.3},
                        default_dofs_force_limits={"hinge_zero": 3.0},
                    ),
                    free_declared_free=EntityOptions(file=asset("free_body"), is_fixed_base=False),
                    jointless_declared_fixed=EntityOptions(file=asset("jointless"), is_fixed_base=True),
                    **{
                        f"{j}_root": EntityOptions(file=asset(f"{j}_root"), is_fixed_base=True, is_articulated=True)
                        for j in ("hinge", "slide", "ball")
                    },
                    tuned_box=BoxOptions(
                        size=(0.1, 0.1, 0.1),
                        default_root_pos=(1.0, 0.0, 0.5),
                        material=RigidMaterialOptions(friction_torsional=_TORSIONAL, friction_rolling=_ROLLING),
                    ),
                    pads=_two_link(
                        asset,
                        default_geoms_friction_torsional={"pad": _TORSIONAL},
                        default_geoms_friction_rolling={"base": _ROLLING},
                    ),
                    layered=_two_link(
                        asset,
                        material=RigidMaterialOptions(friction_torsional=_MATERIAL_TORSIONAL),
                        default_geoms_friction_torsional={"pad": _TORSIONAL},
                    ),
                    plain_pads=_two_link(asset),
                    # A kinematic entity has no collision geoms, so a geom option cannot be written there.
                    **{
                        f"marker_{field}": BoxOptions(
                            size=(0.05, 0.05, 0.05),
                            material=KinematicMaterialOptions(),
                            **{field: {"box_baselink": 0.02}},
                        )
                        for field in ("default_geoms_friction_torsional", "default_geoms_friction")
                    },
                    # Collision vis / a collision mask on a kinematic marker, and collision vis on a rigid block.
                    vis_marker=SphereOptions(radius=0.05, material=KinematicMaterialOptions(), vis_mode="collision"),
                    masked_marker=SphereOptions(
                        radius=0.05, material=KinematicMaterialOptions(), collision_link_patterns=[".*"]
                    ),
                    vis_block=BoxOptions(size=(0.1, 0.1, 0.1), vis_mode="collision", default_root_pos=(0.3, 0.0, 0.05)),
                ),
            )
        )
    return env, [r.getMessage() for r in records if r.levelno >= 30]


@pytest.fixture
def mjcf_env(mjcf_build):
    return mjcf_build[0]


# ── free-base DOFs by joint type ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("name", ["floater", "gizmo"])
def test_free_dofs_are_found_by_type_and_get_the_stability_damping(mjcf_env, name):
    entity = mjcf_env.entities[name]
    assert entity.find_free_base_dofs_idx_local() == _FREE_DOFS
    damping = entity.get_dofs_damping(dofs_idx_local=_FREE_DOFS, envs_idx=0)
    assert torch.allclose(damping, torch.full_like(damping, _STABILITY_DAMPING))


def test_a_hinge_matching_the_old_name_regex_keeps_its_own_damping(mjcf_env):
    gizmo = mjcf_env.entities["gizmo"]
    _, idx = gizmo.find_named_dofs_idx_local(["turntable_base_joint"])
    assert torch.allclose(gizmo.get_dofs_damping(dofs_idx_local=idx, envs_idx=0), torch.tensor(0.05))


def test_an_explicit_free_joint_name_is_validated(mjcf_env):
    assert mjcf_env.entities["floater"].find_free_base_dofs_idx_local("weird_floaty_thing") == _FREE_DOFS
    with pytest.raises(ValueError, match="does not name a free joint"):
        mjcf_env.entities["gizmo"].find_free_base_dofs_idx_local("turntable_base_joint")


def test_free_base_lookup_raises_without_a_floating_base(mjcf_env):
    with pytest.raises(ValueError, match="no floating base"):
        mjcf_env.entities["plane"].find_free_base_dofs_idx_local()


def test_two_free_joints_are_ambiguous_until_one_is_named(mjcf_env):
    twins = mjcf_env.entities["twins"]
    with pytest.raises(ValueError, match="found 2.*first_free.*second_free"):
        twins.find_free_base_dofs_idx_local()
    assert twins.find_free_base_dofs_idx_local("first_free") == list(range(6))
    assert twins.find_free_base_dofs_idx_local("second_free") == list(range(6, 12))


# ── soft position limits on unbounded DOFs ────────────────────────────────────────────────────────


def test_soft_limits_shrink_bounded_rows_and_pass_unbounded_rows_through(mjcf_env):
    """An infinite range has no midpoint to shrink about: ``inf - inf`` was NaN, which reached the
    ``dofs_pos_limits`` penalty and the ``reset_dofs_by_*`` clamps. The decision is per row.
    """
    mixed = mjcf_env.entities["mixed"]
    hard, soft = mixed.default_dofs_pos_limits[0], mixed.soft_dofs_pos_limits[0]
    limited, unlimited = mixed.find_named_dofs(["limited_joint", "unlimited_joint"]).columns.tolist()
    assert torch.isfinite(hard[limited]).all() and torch.isinf(hard[unlimited]).all()
    mid = hard[limited].mean()
    assert torch.allclose(soft[limited] - mid, 0.9 * (hard[limited] - mid))
    assert torch.equal(soft[unlimited], hard[unlimited])

    bounded = mjcf_env.entities["bounded"]
    hard, soft = bounded.default_dofs_pos_limits[0], bounded.soft_dofs_pos_limits[0]
    mid = hard.mean(dim=1, keepdim=True)
    assert torch.allclose(soft - mid, 0.9 * (hard - mid))

    floppy = mjcf_env.entities["floppy"]  # ball + range-less hinge: every row unbounded
    assert torch.isinf(floppy.default_dofs_pos_limits).all()
    assert torch.equal(floppy.soft_dofs_pos_limits, floppy.default_dofs_pos_limits)


# ── name resolution: an empty scope, and the deprecated name map ───────────────────────────────────


@pytest.mark.parametrize("resolver", ["find_named_dofs_idx_local", "find_named_links_idx_local"])
def test_an_empty_name_scope_searches_nothing(mjcf_env, resolver):
    """Not every joint or link the asset has, which is what an empty scope used to fall back to."""
    with pytest.raises(ValueError, match="Not all query patterns are matched"):
        getattr(mjcf_env.entities["mixed"], resolver)(".*", name_scope=[])


# ── joint defaults the Genesis loader fills in ────────────────────────────────────────────────────


def _by_dof(entity, values) -> dict[str, float]:
    return dict(zip(entity.dofs_name, values.reshape(-1).tolist()))


@pytest.mark.parametrize(
    "name, expected",
    [
        # An authored armature="0" counts as "left at zero"; a two-joint body is skipped.
        ("rig", dict(hinge_zero=_DEFAULT_ARMATURE, hinge_kept=_AUTHORED_ARMATURE, slide_bare=_DEFAULT_ARMATURE)),
        # default_armature=0.0 keeps the asset's zeros (the HuskySkateboard recipe).
        ("rig_unfilled", dict(hinge_zero=0.0, hinge_kept=_AUTHORED_ARMATURE, slide_bare=0.0)),
        # The per-DOF map wins over both the fill and an authored value (the porting recipe).
        ("rig_overridden", dict(hinge_zero=0.0, hinge_kept=0.5, slide_bare=_DEFAULT_ARMATURE, pair_a=0.3, pair_b=0.3)),
    ],
)
def test_default_armature_fills_the_zero_dofs_of_single_joint_bodies(mjcf_env, name, expected):
    rig = mjcf_env.entities[name]
    expected = {"pair_a": 0.0, "pair_b": 0.0, **expected}
    built = _by_dof(rig, rig.get_dofs_armature(envs_idx=0))
    assert built == pytest.approx(expected)
    assert _by_dof(rig, rig.default_dofs_armature[:1]) == pytest.approx(built)  # seeded from the built value


def _write_usd_hinge(path) -> str:
    """A fixed base and one arm on a revolute joint with no authored armature, as a USD stage."""
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/hinge_rig")
    stage.SetDefaultPrim(root.GetPrim())
    UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())
    for name, z in (("base", 0.0), ("arm", 0.2)):
        body = UsdGeom.Xform.Define(stage, f"/hinge_rig/{name}")
        body.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, z))
        UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
        UsdPhysics.MassAPI.Apply(body.GetPrim()).CreateMassAttr(1.0)
        cube = UsdGeom.Cube.Define(stage, f"/hinge_rig/{name}/geom")
        cube.CreateSizeAttr(0.1)
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    joint = UsdPhysics.RevoluteJoint.Define(stage, "/hinge_rig/hinge")
    joint.CreateBody0Rel().SetTargets(["/hinge_rig/base"])
    joint.CreateBody1Rel().SetTargets(["/hinge_rig/arm"])
    joint.CreateAxisAttr("Y")
    joint.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, 0.1))
    joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, -0.1))
    stage.Save()
    return str(path)


def test_default_armature_reaches_a_usd_asset(tmp_path):
    # Genesis' USD morph defaults its own default_armature to 0.1, so the option must be passed through for
    # default_armature=0.0 to disable the fill on a USD asset, as it does on URDF and MJCF.
    pytest.importorskip("pxr")
    usd = _write_usd_hinge(tmp_path / "hinge.usda")
    rig = dict(file=usd, is_fixed_base=True, is_articulated=True, dofs_name=["/hinge_rig/hinge"])
    env = build_env(
        EdenConfig(
            env_options=EnvOptions(num_envs=1),
            scene_options=SceneOptions(
                plane=PlaneOptions(),
                filled=EntityOptions(**rig, default_root_pos=(0.0, 0.0, 0.5)),
                unfilled=EntityOptions(**rig, default_root_pos=(1.0, 0.0, 0.5), default_armature=0.0),
            ),
        )
    )
    assert env.entities["filled"].get_dofs_armature(envs_idx=0).item() == pytest.approx(_DEFAULT_ARMATURE)
    assert env.entities["unfilled"].get_dofs_armature(envs_idx=0).item() == pytest.approx(0.0)


_GATE_KD, _GATE_ARMATURE = 2.0, 0.05
_GATE_HINGE_MJCF = """<mujoco model="gate_hinge"><worldbody>
  <body name="base"><geom type="box" size="0.05 0.05 0.05" mass="1"/>
    <body name="arm" pos="0 0 0.1"><joint name="hinge" type="hinge" axis="0 1 0"/>
      <geom type="capsule" fromto="0 0 0 0 0 0.2" size="0.02" mass="0.1"/></body></body>
</worldbody></mujoco>"""


@pytest.fixture(scope="module", params=["approximate_implicitfast", "implicitfast", "euler"])
def integrator_env(request, tmp_path_factory):
    path = tmp_path_factory.mktemp("gate_assets") / "hinge.xml"
    path.write_text(_GATE_HINGE_MJCF)
    return build_env(
        EdenConfig(
            env_options=EnvOptions(num_envs=1, integrator=request.param),
            scene_options=SceneOptions(
                arm=EntityOptions(
                    file=str(path),
                    is_fixed_base=True,
                    is_articulated=True,
                    is_actuated=True,
                    dofs_name=["hinge"],
                    default_dofs_kp={"hinge": 10.0},
                    default_dofs_kd={"hinge": _GATE_KD},
                    default_dofs_armature={"hinge": _GATE_ARMATURE},
                )
            ),
        )
    )


def test_post_build_folds_kd_dt_into_the_armature_under_euler_only(integrator_env):
    """Only ``euler`` integrates the PD damping explicitly and needs it folded into the armature.

    ``approximate_implicitfast`` and ``implicitfast`` integrate it implicitly: a fold there would count it twice.
    """
    options = integrator_env.env_options
    substep_dt = options.sim_dt / options.sim_substeps
    fold = _GATE_KD * substep_dt if options.integrator == "euler" else 0.0
    built = integrator_env.entities["arm"].get_dofs_armature(envs_idx=0)
    assert built.reshape(-1).tolist() == pytest.approx([_GATE_ARMATURE + fold])


def test_a_joint_actuatorfrcrange_seeds_the_effort_limit_and_an_override_wins(mjcf_env):
    rig, overridden = mjcf_env.entities["rig"], mjcf_env.entities["rig_overridden"]
    lower, upper = (_by_dof(rig, bound[:1]) for bound in rig.get_dofs_force_range())
    assert (lower["hinge_zero"], upper["hinge_zero"]) == pytest.approx((-_FRC_RANGE, _FRC_RANGE))
    assert _by_dof(rig, rig.default_dofs_force_limits[:1])["hinge_zero"] == pytest.approx(_FRC_RANGE)
    assert math.isinf(upper["hinge_kept"])  # a joint stating no bound stays unbounded
    assert _by_dof(overridden, overridden.get_dofs_force_range()[1][:1])["hinge_zero"] == pytest.approx(3.0)


# ── the fixed-base declaration must match the MJCF base link ──────────────────────────────────────


def test_declarations_that_match_the_asset_build_without_a_warning(mjcf_build):
    env, warnings = mjcf_build
    assert [w for w in warnings if "is_fixed_base" in w] == []
    assert not env.entities["free_declared_free"].has_immobile_base
    assert env.entities["jointless_declared_fixed"].has_immobile_base


@pytest.mark.parametrize("joint, n_dofs", [("hinge", 1), ("slide", 1), ("ball", 3)])
def test_a_root_moved_by_its_own_joint_is_a_fixed_base(mjcf_env, joint, n_dofs):
    """Its base link moves, but not on a free joint, so there is no floating base."""
    root = mjcf_env.entities[f"{joint}_root"]
    assert root.n_dofs == n_dofs and root.has_immobile_base


@pytest.mark.parametrize(
    "asset, is_fixed_base, match",
    [
        # Off the origin, so Genesis keeps its virtual ``world`` link as the base link.
        ("jointless", False, r"base link 'world' of .* carries no joint, so Genesis welds it to the world"),
        ("free_body", True, r"base link 'base' of .* carries a free joint"),
        ("hinge_root", False, r"is moved by its own 'j1' \(revolute\) joint\(s\), not a free joint"),
        ("free_beside_world_geom", False, r"free joint\(s\) \['box'\] sit on a body other than the base link"),
    ],
)
def test_a_declaration_the_asset_contradicts_raises_at_build(mjcf_assets, asset, is_fixed_base, match):
    fixture = EntityOptions(file=mjcf_assets(asset), is_fixed_base=is_fixed_base)
    with pytest.raises(ValueError, match=match):
        build_env(_config(plane=PlaneOptions(), fixture=fixture))


# ── per-geom friction axes ────────────────────────────────────────────────────────────────────────


def _solver_by_link(env, entity, attr) -> dict[str, float]:
    """``{link: coefficient}`` read from the solver's per-geom info, where the contact kernels look.

    The per-link maps are written after ``scene.build()``; the geom's own python-side getter would
    agree even if the solver write were skipped.
    """
    values = getattr(env.rigid_solver.dyn_info.geoms, attr).to_numpy()
    return {geom.link.name: float(values[geom.idx]) for geom in entity.geoms}


def test_a_material_coefficient_lands_on_every_geom_of_its_entity_only(mjcf_env):
    """Unset material axes are not forwarded as an explicit 0: the untuned box keeps Genesis' fallbacks."""
    tuned, untuned = mjcf_env.entities["tuned_box"], mjcf_env.entities["box"]
    assert _solver_by_link(mjcf_env, tuned, "friction_torsional") == {"box_baselink": pytest.approx(_TORSIONAL)}
    assert _solver_by_link(mjcf_env, tuned, "friction_rolling") == {"box_baselink": pytest.approx(_ROLLING)}
    assert _solver_by_link(mjcf_env, untuned, "friction_torsional") == {"box_baselink": pytest.approx(0.005)}
    assert _solver_by_link(mjcf_env, untuned, "friction_rolling") == {"box_baselink": pytest.approx(0.0001)}


def test_link_patterns_target_one_axis_on_the_matching_links(mjcf_env):
    """Fingertips want more spin resistance than the palm, which a material-wide value cannot say."""
    pads = mjcf_env.entities["pads"]
    assert _solver_by_link(mjcf_env, pads, "friction_torsional") == {"pad": pytest.approx(_TORSIONAL), "base": 0.0}
    assert _solver_by_link(mjcf_env, pads, "friction_rolling") == {"base": pytest.approx(_ROLLING), "pad": 0.0}
    assert {g.link.name: float(g.get_friction()) for g in pads.geoms} == {"base": 1.0, "pad": 1.0}  # sliding untouched


def test_a_link_pattern_composes_over_the_material_coefficient(mjcf_env):
    layered = mjcf_env.entities["layered"]
    assert _solver_by_link(mjcf_env, layered, "friction_torsional") == {
        "base": pytest.approx(_MATERIAL_TORSIONAL),
        "pad": pytest.approx(_TORSIONAL),
    }


def test_mjcf_geoms_without_condim_parse_to_zero_spin_resistance(mjcf_env):
    """So on an MJCF robot the switch alone buys nothing; the coefficient has to be set."""
    plain = mjcf_env.entities["plain_pads"]
    assert _solver_by_link(mjcf_env, plain, "friction_torsional") == {"base": 0.0, "pad": 0.0}
    assert _solver_by_link(mjcf_env, plain, "friction_rolling") == {"base": 0.0, "pad": 0.0}


@pytest.mark.parametrize("field", ["default_geoms_friction_torsional", "default_geoms_friction"])
def test_geom_options_on_a_kinematic_entity_are_skipped_with_a_warning(mjcf_build, field):
    """Its solver builds visual geoms only; writing one used to abort post_build with an AttributeError."""
    env, warnings = mjcf_build
    assert env.entities[f"marker_{field}"].is_kinematic
    assert any(field in w and f"'marker_{field}'" in w for w in warnings), warnings


def test_collision_vis_on_a_kinematic_entity_falls_back_to_visual_and_says_so(mjcf_build):
    """Genesis' rasterizer reads ``geoms`` for any non-visual mode; a kinematic entity has only ``vgeoms``.

    It used to abort ``scene.build()`` with an ``AttributeError``. The downgrade is render-time only, so
    ``env.config`` still round-trips what was asked for.
    """
    env, warnings = mjcf_build
    assert env.entities["vis_marker"].entity.surface.vis_mode == "visual"
    assert env.entities["vis_block"].entity.surface.vis_mode == "collision"  # it has collision geoms
    assert env.config.scene_options.vis_marker.vis_mode == "collision"
    assert any("vis_marker" in w and "collision" in w for w in warnings), warnings
    assert any("collision_link_patterns" in w and "masked_marker" in w for w in warnings), warnings


@pytest.mark.parametrize(
    "options, env_options, match",
    [
        (
            dict(material=RigidMaterialOptions(friction_torsional=_TORSIONAL)),
            {},
            r"material\.friction_torsional.*enable_torsional_friction",
        ),
        (
            dict(default_geoms_friction_rolling={"base": _ROLLING}),
            dict(enable_torsional_friction=True),
            r"'rig'.*default_geoms_friction_rolling.*enable_rolling_friction",
        ),
    ],
    ids=["material-torsional", "pattern-rolling"],
)
def test_a_coefficient_whose_switch_is_off_raises_at_build(mjcf_assets, options, env_options, match):
    """A value the solver never reads is a misconfiguration, and the message names entity and option."""
    config = EdenConfig(
        env_options=EnvOptions(num_envs=1, **env_options),
        scene_options=SceneOptions(plane=PlaneOptions(), rig=_two_link(mjcf_assets, **options)),
    )
    with pytest.raises(ValueError, match=match):
        build_env(config)


def test_one_raise_reports_every_dead_coefficient_with_a_fix_that_works(mjcf_assets):
    """Rolling needs torsional too, so a rolling clause naming only its own switch would fail again."""
    rig = _two_link(
        mjcf_assets,
        material=RigidMaterialOptions(friction_torsional=_TORSIONAL),
        default_geoms_friction_rolling={"base": _ROLLING},
    )
    with pytest.raises(ValueError) as excinfo:
        build_env(_config(plane=PlaneOptions(), rig=rig))
    torsional_clause, rolling_clause = str(excinfo.value).split("; ")
    assert "material.friction_torsional" in torsional_clause and "enable_torsional_friction" in torsional_clause
    assert all(s in rolling_clause for s in ("default_geoms_friction_rolling", "enable_rolling_friction"))
    assert "enable_torsional_friction" in rolling_clause


# ═════════════════════════════════════════════════════════════════════════════════════════════════
# Index selectors and control setters, and grouped contacts, on a stepped scene
# ═════════════════════════════════════════════════════════════════════════════════════════════════


def _devices():
    return ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.fixture(scope="module")
def control_env():
    """A free box over a plane, a small box resting on a fixed slab, and a Panda at kp = 10, kd = 1; three envs."""
    env = build_env(
        EdenConfig(
            env_options=EnvOptions(num_envs=3),
            scene_options=SceneOptions(
                plane=PlaneOptions(),
                box=BoxOptions(size=(0.2, 0.2, 0.2), default_root_pos=(3.0, 0.0, 1.0)),
                top=BoxOptions(size=(0.1, 0.1, 0.1), default_root_pos=(0.0, 3.0, 0.6)),
                bottom=BoxOptions(size=(0.5, 0.5, 0.1), default_root_pos=(0.0, 3.0, 0.05), fixed=True),
                robot=FrankaEmikaPanda(default_dofs_kp={".*": 10.0}, default_dofs_kd={".*": 1.0}),
            ),
        )
    )
    for _ in range(200):  # the small box comes to rest on the slab; the box over the plane lands too
        env.scene.step()
    return env


# ---------------------------------------------------------------------------
# The setter forms MotionCommand's resets use
# ---------------------------------------------------------------------------


def test_genesis_setters_write_only_the_masked_envs_given_a_dof_range(control_env):
    """A ``range`` DOF axis and a bool ``envs_idx``: Genesis takes every env's row and writes the masked ones.

    No ``nonzero()``, then forward kinematics for those envs only. Its indexed path would reject the
    full rows, so a Genesis change routing this form there fails here rather than silently.
    """
    assert gs.use_zerocopy
    box = control_env.entities["box"]
    assert box.setters_take_every_row  # what MotionCommand.build() checks before using this form
    mask = torch.tensor([True, False, True])
    qpos_before = box.get_qpos().clone()
    vel_before = box.get_dofs_vel(dofs_idx_local=range(6)).clone()
    qpos = qpos_before.clone()
    qpos[:, :3] += torch.tensor([0.5, -0.25, 0.75])
    vel = torch.full((3, 6), 0.3)

    box.set_dofs_vel(vel, dofs_idx_local=range(6), envs_idx=mask, skip_forward=True)
    box.set_qpos(qpos, qs_idx_local=range(7), envs_idx=mask, zero_velocity=False)

    torch.testing.assert_close(box.get_qpos()[mask], qpos[mask])
    assert torch.equal(box.get_qpos()[~mask], qpos_before[~mask])
    torch.testing.assert_close(box.get_dofs_vel(dofs_idx_local=range(6))[mask], vel[mask])
    assert torch.equal(box.get_dofs_vel(dofs_idx_local=range(6))[~mask], vel_before[~mask])
    torch.testing.assert_close(box.get_pos()[mask], qpos[mask, :3], atol=1e-6, rtol=0)
    torch.testing.assert_close(box.get_pos()[~mask], qpos_before[~mask, :3], atol=1e-6, rtol=0)
    box.set_qpos(qpos_before)


def _hibernating_box_env() -> EnvBase:
    """A free box over a plane, three envs, hibernation switched on behind ``EnvOptions.use_hibernation``."""
    env_options = EnvOptions(num_envs=3)
    env = EnvBase(
        env_options=env_options,
        scene_options=SceneOptions(
            plane=PlaneOptions(), box=BoxOptions(size=(0.2, 0.2, 0.2), default_root_pos=(0.0, 0.0, 1.0))
        ),
        rigid_options=gs.options.RigidOptions(
            use_hibernation=True,
            batch_dofs_info=env_options.batch_dofs_info,
            batch_links_info=env_options.batch_links_info,
            batch_joints_info=env_options.batch_joints_info,
        ),
    )
    env.build()
    return env


def test_setters_take_every_row_reads_the_solvers_own_hibernation_flag():
    """Hibernation switched on behind ``EnvOptions.use_hibernation`` must still close the mask path.

    A ``rigid_options`` handed to the env constructor replaces the one Eden builds. Under hibernation
    ``set_dofs_velocity`` wakes what it touches first, turning the mask into indices and rejecting every
    env's row, so a gate reading the Eden option would send ``MotionCommand``'s resets into a shape error.
    """
    env = _hibernating_box_env()
    box = env.entities["box"]
    assert not env.env_options.use_hibernation and not box.setters_take_every_row

    mask = torch.tensor([True, False, True])
    qpos_before = box.get_qpos().clone()
    qpos = qpos_before.clone()
    qpos[:, :3] += torch.tensor([0.5, -0.25, 0.75])
    vel = torch.full((3, 6), 0.3)
    box.set_dofs_vel(vel[mask], dofs_idx_local=torch.arange(6), envs_idx=mask, skip_forward=True)
    box.set_qpos(qpos[mask], qs_idx_local=torch.arange(7), envs_idx=mask, zero_velocity=False)
    torch.testing.assert_close(box.get_qpos()[mask], qpos[mask])
    assert torch.equal(box.get_qpos()[~mask], qpos_before[~mask])
    torch.testing.assert_close(box.get_dofs_vel(dofs_idx_local=range(6))[mask], vel[mask])


# ---------------------------------------------------------------------------
# set_dofs_pos writes hinge / slide DOFs as their coordinates, mask and all
# ---------------------------------------------------------------------------


def _panda_targets(robot, rows: int) -> torch.Tensor:
    """Distinct in-limit DOF positions, one row per env."""
    low, high = robot.get_dofs_limit()
    frac = torch.linspace(0.3, 0.7, rows, device=robot.device)[:, None]
    return low[:rows] + (high[:rows] - low[:rows]) * frac


@pytest.mark.parametrize("rows", ["every-env", "selected"])
def test_set_dofs_pos_through_a_mask_leaves_what_genesis_set_dofs_position_leaves(control_env, rows):
    """Envs 0 and 1 through the mask path (``qpos0 + position`` via ``set_qpos``), env 2 through Genesis' own setter.

    Two selected envs, so the selected-rows form takes Genesis' ``masked_scatter_`` branch, not a broadcast.
    """
    env = control_env
    robot = env.entities["robot"]
    assert robot.setters_take_every_row
    targets = _panda_targets(robot, 3)
    targets[2] = targets[1]
    mask = torch.tensor([True, True, False], device=env.device)
    env_2_before = robot.get_qpos()[2].clone()

    robot.set_dofs_pos(targets if rows == "every-env" else targets[:2], envs_idx=mask)
    torch.testing.assert_close(robot.get_dofs_pos()[:2], targets[:2])
    assert torch.equal(robot.get_qpos()[2], env_2_before)  # untouched by the masked write
    robot._entity.set_dofs_position(targets[2:], robot._genesis_dofs_idx, torch.tensor([2], device=env.device))

    torch.testing.assert_close(robot.get_qpos()[1], robot.get_qpos()[2])
    torch.testing.assert_close(robot.get_links_pos()[1], robot.get_links_pos()[2])
    assert robot.get_dofs_vel()[:2].abs().max() == 0  # zero_velocity, the default


def test_set_dofs_pos_spreads_one_row_over_a_kinematic_entitys_masked_envs(staged_env):
    """The kinematic solver scatters a one-row value into the masked rows rather than broadcasting it."""
    env = staged_env
    ghost = env.entities["ghost"]
    mask = torch.tensor([True, False, True, False], device=env.device)
    before = ghost.get_dofs_pos().clone()
    fingers = torch.tensor([0.01, 0.03], device=env.device)

    ghost.set_dofs_pos(fingers, envs_idx=mask)
    ghost.set_qpos(ghost.get_qpos()[:1].clone(), envs_idx=mask)  # one row, two envs: the same spread

    torch.testing.assert_close(ghost.get_dofs_pos()[mask], fingers.expand(2, -1))
    assert torch.equal(ghost.get_dofs_pos()[~mask], before[~mask])


def test_set_dofs_pos_over_a_free_joint_takes_every_row_with_a_mask_too(control_env):
    """A free joint's DOFs have no coordinates of their own, so Genesis' setter writes them: given the selected rows."""
    env = control_env
    box = env.entities["box"]
    mask = torch.tensor([False, True, False], device=env.device)
    before = box.get_dofs_pos(dofs_idx_local=range(6)).clone()
    target = before.clone()
    target[:, 2] += 0.25

    box.set_dofs_pos(target, dofs_idx_local=range(6), envs_idx=mask)

    after = box.get_dofs_pos(dofs_idx_local=range(6))
    torch.testing.assert_close(after[1], target[1])
    assert torch.equal(after[[0, 2]], before[[0, 2]])
    box.set_dofs_pos(before, dofs_idx_local=range(6))


# ---------------------------------------------------------------------------
# Control setters over gappy and reordered DOF selections
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dofs", [[0, 3, 4], [4, 3, 5]], ids=["gappy", "reordered"])
@pytest.mark.parametrize("value_form", ["tensor", "list", "scalar"])
def test_control_setters_command_each_selected_dof_its_own_column(control_env, dofs, value_form):
    """A gappy selection is written one contiguous run at a time, a reordered one through its permutation.

    Read back as control forces: ``kp (target - pos)``, ``kd (target - vel)`` and the force itself.
    """
    robot = control_env.entities["robot"]
    qpos = robot.get_qpos().clone()
    robot.set_dofs_pos(torch.zeros(3, robot.num_dofs))  # at rest at zero: every read-back stays in force range
    values = torch.tensor([[0.1, 0.2, 0.3]] * 3)
    value = {"tensor": values, "list": [0.1, 0.2, 0.3], "scalar": 0.1}[value_form]
    expected = values if value_form != "scalar" else torch.full((3, 3), 0.1)
    pos, vel = robot.get_dofs_pos()[:, dofs], robot.get_dofs_vel()[:, dofs]
    for setter, readback in (
        (robot.control_dofs_pos, 10.0 * (expected - pos) - vel),
        (robot.control_dofs_vel, expected - vel),
        (robot.control_dofs_force, expected),
    ):
        setter(value, dofs_idx_local=torch.tensor(dofs) if value_form == "tensor" else dofs)
        torch.testing.assert_close(
            control_force_over_one_step(control_env, robot)[:, dofs], readback, atol=1e-5, rtol=0
        )
    robot.set_qpos(qpos)


def test_link_reads_through_an_index_tensor_match_the_genesis_read(control_env, count_cuda_syncs):
    robot = control_env.entities["robot"]
    for selection in (torch.tensor([0], dtype=gs.tc_int), torch.tensor([3, 4, 5], dtype=gs.tc_int)):
        got = robot.get_links_pos(ls_idx_local=selection)
        assert torch.equal(got, robot._entity.get_links_pos(links_idx_local=selection)) and got.is_contiguous()
        if gs.device.type == "cuda":  # the suite runs Genesis on the CPU, where nothing can sync
            assert count_cuda_syncs(lambda: robot.get_links_pos(ls_idx_local=selection)) == 0
    # Counted from the end: the last links, as Genesis reads them, never the empty `range(-1, 0)`.
    for selection in ([-1], torch.tensor([-2, -1], dtype=gs.tc_int)):
        got = robot.get_links_pos(ls_idx_local=selection)
        assert got.shape == (control_env.num_envs, len(selection), 3)
        assert torch.equal(got, robot._entity.get_links_pos(links_idx_local=selection))


# ---------------------------------------------------------------------------
# _genesis_selection: explicit index selectors resolved once
# ---------------------------------------------------------------------------


def test_explicit_tensor_selection_resolves_to_a_range_once(count_cuda_syncs):
    one_link = torch.tensor([7], device="cuda")
    assert count_cuda_syncs(lambda: _genesis_selection(one_link)) >= 1  # read once, on first sight
    assert _genesis_selection(one_link) == range(7, 8)
    assert count_cuda_syncs(lambda: _genesis_selection(one_link)) == 0
    # A Python list, as `find_named_links_idx_local` returns it: sync-free from the first call on.
    assert count_cuda_syncs(lambda: _genesis_selection([2, 5, 6])) == 0
    assert count_cuda_syncs(lambda: _genesis_selection([2, 3])) == 0


@pytest.mark.parametrize("device", _devices())
def test_explicit_selection_follows_an_in_place_write(device):
    idx = torch.tensor([3, 4], device=device)
    assert _genesis_selection(idx) == range(3, 5)
    idx[1] = 9
    assert _genesis_selection(idx) is idx  # no longer contiguous: handed through as the tensor
    idx[1] = 4
    assert _genesis_selection(idx) == range(3, 5)


def test_explicit_selection_forms():
    gappy = torch.tensor([0, 3])
    assert _genesis_selection(gappy) is gappy
    assert _genesis_selection(gappy, control=True) == DofRuns([(range(0, 1), slice(0, 1)), (range(3, 4), slice(1, 2))])
    assert _genesis_selection([5, 6, 7]) == range(5, 8)
    gappy_list = _genesis_selection([5, 7])  # no range form: a device index tensor, built once per value
    assert isinstance(gappy_list, torch.Tensor) and gappy_list.tolist() == [5, 7]
    assert _genesis_selection([5, 7]) is gappy_list
    assert _genesis_selection(range(1, 3)) == range(1, 3)
    with torch.inference_mode():
        inference = torch.tensor([1, 2])
    assert _genesis_selection(inference) is inference  # no version counter, so nothing is cached


@pytest.mark.parametrize("control", [False, True])
def test_explicit_selection_cache_lets_go_of_a_tensor_it_hands_on(control):
    # No range form either way: gapped for a getter, too many runs for a control setter.
    idx = torch.tensor([0, 2, 4, 6, 8, 10])
    key = (id(idx), control)
    assert _genesis_selection(idx, control=control) is idx
    assert key in _RESOLVED_SELECTIONS
    assert _genesis_selection(idx, control=control) is idx  # the cached verdict, same tensor back
    del idx
    gc.collect()
    assert key not in _RESOLVED_SELECTIONS


def test_explicit_selection_cache_does_not_grow_with_per_call_tensors():
    before = len(_RESOLVED_SELECTIONS)
    for _ in range(50):
        _genesis_selection(torch.tensor([0, 2, 5]))
    assert len(_RESOLVED_SELECTIONS) <= before + 1


@pytest.mark.parametrize("dofs", [[-1], [-2, -1], [-1, 0], [-3, -2]])
def test_selections_counted_from_the_end_keep_their_meaning(dofs):
    # Genesis makes `range(-2, 0)` the empty `slice(-2, 0)`: a negative index must never become a range
    # (the misc helpers' side of this is in tests/utils/test_misc.py).
    idx = torch.tensor(dofs)
    assert _genesis_selection(idx) is idx
    assert _genesis_selection(idx, control=True) is idx
    assert _genesis_selection(dofs) is dofs  # handed to Genesis exactly as before
    assert _genesis_selection(dofs, control=True) is dofs


def test_a_permutation_resolved_from_a_cpu_index_keeps_its_order_on_the_sim_device(monkeypatch):
    import genesis as gs

    # `meta` stands in for the sim device, so a CPU `order` is told apart on the CPU suite too. A CPU index,
    # as `torch.tensor(perm)` without `device=` is once no torch default device is set, must not leave the
    # order on the CPU: that would pull the sim-device values to the host on every control call.
    monkeypatch.setattr(gs, "device", torch.device("meta"))
    resolved = _genesis_selection(torch.tensor([4, 3, 5]), control=True)
    assert isinstance(resolved, DofPermutation) and resolved.dofs == range(3, 6)
    assert resolved.order.device.type == "meta"


def test_kernel_selection_is_a_cached_device_arange():
    first = _kernel_selection(range(6, 35))
    assert first.tolist() == list(range(6, 35)) and first.dtype == gs.tc_int and first.device == gs.device
    assert _kernel_selection(range(6, 35)) is first
    idx = torch.tensor([1, 4])
    assert _kernel_selection(idx) is idx


# ---------------------------------------------------------------------------
# get_grouped_contacts
# ---------------------------------------------------------------------------


def test_grouped_contacts_put_a_resting_contact_between_the_two_bodies(control_env):
    """The small box rests on the slab: one valid (top, bottom) pair per env, at the interface."""
    top, bottom = control_env.entities["top"], control_env.entities["bottom"]
    # The contacts are those of the last step, and a pose setter of an earlier test of the module clears them
    control_env.scene.step()
    out = top.get_grouped_contacts(bottom)
    assert set(out) == {"position", "valid", "force_norm_sum"}
    assert out["position"].shape == (3, top.num_links, bottom.num_links, 3)
    assert out["valid"].flatten(start_dim=1).any(dim=1).all()
    best = out["force_norm_sum"].flatten(start_dim=1).argmax(dim=-1)
    rep = out["position"].flatten(start_dim=1, end_dim=2)[torch.arange(3), best]
    assert (rep[:, :2] - torch.tensor([0.0, 3.0])).abs().max() < 0.1
    assert ((rep[:, 2] > 0.05) & (rep[:, 2] < 0.6)).all()


def test_grouped_contacts_reject_self_contact(control_env):
    """The OR-of-flips aggregation double-stamps overlapping groups; ``get_contacts(with_entity=self)`` is the tool."""
    with pytest.raises(ValueError, match="does not support self-contact"):
        control_env.entities["top"].get_grouped_contacts(control_env.entities["top"])


def test_a_padded_contact_query_zeroes_its_invalid_slots_and_holds_the_trimmed_queries_contacts(control_env):
    """A padded slot past an env's live contacts holds whatever an earlier step left there, a NaN force included.

    Zeroed, ``force * valid_mask`` cannot pick up a stale NaN. The valid slots are the trimmed query's
    contacts.
    """
    top, bottom = control_env.entities["top"], control_env.entities["bottom"]
    control_env.scene.step()
    trimmed = top.get_contacts(with_entity=bottom)
    padded = top.get_contacts(with_entity=bottom, is_padded=True)
    invalid = ~padded["valid_mask"]
    assert padded["valid_mask"].any(dim=1).all() and invalid.any()
    for key in ("position", "force_a", "force_b"):
        assert (padded[key][invalid] == 0).all(), key
        torch.testing.assert_close(
            (padded[key] * padded["valid_mask"][..., None]).sum(dim=1),
            (trimmed[key] * trimmed["valid_mask"][..., None]).sum(dim=1),
        )


def test_a_contact_query_is_padded_while_the_env_grades_under_a_cuda_graph(control_env):
    """``EnvOptions.cuda_graph_options`` runs each graded step in ``env.fixed_shape_query_scope()``.

    The eager runs that validate the graph then reduce over the same slots as the graph does, so a float sum
    agrees bit for bit.
    """
    top, bottom = control_env.entities["top"], control_env.entities["bottom"]
    control_env.scene.step()
    padded = top.get_contacts(with_entity=bottom, is_padded=True)
    with control_env.fixed_shape_query_scope():
        with control_env.fixed_shape_query_scope():  # a nested scope leaves the outer one open
            pass
        assert control_env.fixed_shape_queries
        under_graph = top.get_contacts(with_entity=bottom)
    assert not control_env.fixed_shape_queries
    with pytest.raises(RuntimeError, match="capture failed"), control_env.fixed_shape_query_scope():
        raise RuntimeError("capture failed")  # a failed capture leaves the env querying trimmed contacts again
    assert not control_env.fixed_shape_queries
    assert set(under_graph) == set(padded)
    for key, value in padded.items():
        assert torch.equal(under_graph[key], value), key
    assert top.get_contacts(with_entity=bottom)["valid_mask"].shape[-1] < padded["valid_mask"].shape[-1]


@pytest.mark.parametrize("other", [None, "not_an_entity"])
def test_grouped_contacts_reject_a_non_rigid_partner(control_env, other):
    with pytest.raises(TypeError, match="must be a RigidEntity"):
        control_env.entities["top"].get_grouped_contacts(other)


# ------------------------------------------------------------------
# StagedStateWrite: several entities' writes, applied as one Genesis call per kind
# ------------------------------------------------------------------


@pytest.fixture
def staged_env(batched_env):
    yield batched_env
    batched_env.reset()


def _box_qpos(env, x: float) -> torch.Tensor:
    """Two rows of a free box's qpos at ``(x, 0, 1)``, tilted, with ``w < 0``.

    A free joint's DOF state holds its orientation as Euler angles, so a qpos re-derived from it comes
    back as another quaternion of the same rotation: the negative ``w`` shows the difference.
    """
    return torch.tensor([[x, 0.0, 1.0, -0.8, 0.0, 0.6, 0.0]] * 2, device=env.device)


def test_a_staged_write_leaves_what_each_entitys_own_setters_leave(staged_env):
    """Envs 1 and 3 through one staged write, envs 0 and 2 through the per-entity setters: same state.

    The boxes' qpos carries ``w < 0``, which a write re-deriving qpos from the DOF state would flip.
    """
    env = staged_env
    robot, cube, placed = env.entities["robot"], env.entities["cube"], env.entities["placed"]
    fingers = torch.tensor([[0.01, 0.02], [0.03, 0.035]], device=env.device)
    cube_vel = torch.tensor([[0.1, 0.0, 0.0, 0.0, 0.0, 0.3]] * 2, device=env.device)
    staged, direct = torch.tensor([1, 3], device=env.device), torch.tensor([0, 2], device=env.device)

    write = StagedStateWrite()
    write.set_qpos(placed, _box_qpos(env, -2.0), range(7))
    write.set_dofs_pos(robot, fingers, robot.dofs_idx_local.tolist())
    write.set_dofs_vel(cube, cube_vel, range(6))
    write.set_qpos(cube, _box_qpos(env, -1.0), range(7))
    write.apply(staged)

    robot.set_dofs_pos(fingers, envs_idx=direct, zero_velocity=False)
    cube.set_dofs_vel(cube_vel, envs_idx=direct, skip_forward=True)
    cube.set_qpos(_box_qpos(env, -1.0), envs_idx=direct, zero_velocity=False)
    placed.set_qpos(_box_qpos(env, -2.0), envs_idx=direct, zero_velocity=False)

    for entity in (robot, cube, placed):
        torch.testing.assert_close(entity.get_qpos()[staged], entity.get_qpos()[direct])
        torch.testing.assert_close(entity.get_links_pos()[staged], entity.get_links_pos()[direct])
    torch.testing.assert_close(robot.get_dofs_pos()[staged], fingers)
    torch.testing.assert_close(cube.get_qpos()[staged], _box_qpos(env, -1.0))
    torch.testing.assert_close(placed.get_qpos()[staged], _box_qpos(env, -2.0))
    torch.testing.assert_close(cube.get_dofs_vel()[staged], cube_vel)


def test_a_staged_write_through_a_mask_leaves_what_it_leaves_through_the_ids(staged_env):
    """The rows go in ascending env order either way; through the mask, Genesis writes them on the device."""
    env = staged_env
    robot, cube = env.entities["robot"], env.entities["cube"]
    fingers = torch.tensor([[0.01, 0.02], [0.03, 0.035]], device=env.device)
    cube_vel = torch.tensor([[0.1, 0.0, 0.0, 0.0, 0.0, 0.3], [0.0, 0.2, 0.0, 0.1, 0.0, 0.0]], device=env.device)
    ids, mask = torch.tensor([0, 2], device=env.device), torch.tensor([False, True, False, True], device=env.device)

    def staged() -> StagedStateWrite:
        write = StagedStateWrite()
        write.set_dofs_pos(robot, fingers, robot.dofs_idx_local.tolist())
        write.set_dofs_vel(cube, cube_vel, range(6))
        write.set_qpos(
            cube,
            torch.cat([_box_qpos(env, -1.0)[:, :3] + 0.1 * fingers[:, :1], _box_qpos(env, -1.0)[:, 3:]], 1),
            range(7),
        )
        return write

    staged().apply(ids)
    staged().apply(mask)

    for entity in (robot, cube):
        torch.testing.assert_close(entity.get_qpos()[mask], entity.get_qpos()[ids])
        torch.testing.assert_close(entity.get_links_pos()[mask], entity.get_links_pos()[ids])
        torch.testing.assert_close(entity.get_dofs_vel()[mask], entity.get_dofs_vel()[ids])
    torch.testing.assert_close(robot.get_dofs_pos()[mask], fingers)


@pytest.mark.parametrize("selected", [[False, True, False, True], [False] * 4], ids=["two-envs", "no-env"])
def test_a_staged_write_of_every_envs_row_writes_only_the_masked_ones(staged_env, selected):
    """Every env's row with the mask, as the motion-end resync stages it each step: the unmasked envs are untouched.

    No env selected is the resync's quiet step, which must write nothing at all.
    """
    env = staged_env
    robot, cube, ghost = env.entities["robot"], env.entities["cube"], env.entities["ghost"]
    mask = torch.tensor(selected, device=env.device)
    fingers = torch.tensor([[0.01, 0.02], [0.03, 0.035]] * 2, device=env.device)
    cube_vel = torch.tensor([[0.1, 0.0, 0.0, 0.0, 0.0, 0.3]] * 4, device=env.device)
    cube_qpos = torch.cat([_box_qpos(env, -1.0)] * 2)
    before = {
        name: (e.get_qpos().clone(), e.get_dofs_vel().clone())
        for name, e in env.entities.items()
        if hasattr(e, "get_qpos") and e.get_qpos().numel()
    }

    write = StagedStateWrite()
    write.set_dofs_pos(robot, fingers, robot.dofs_idx_local.tolist())
    write.set_dofs_pos(ghost, fingers, ghost.dofs_idx_local.tolist())
    write.set_dofs_vel(cube, cube_vel, range(6))
    write.set_qpos(cube, cube_qpos, range(7))
    write.apply(mask)

    for name, (qpos, vel) in before.items():
        entity = env.entities[name]
        assert torch.equal(entity.get_qpos()[~mask], qpos[~mask]), name
        assert torch.equal(entity.get_dofs_vel()[~mask], vel[~mask]), name
    torch.testing.assert_close(robot.get_dofs_pos()[mask], fingers[mask])
    torch.testing.assert_close(ghost.get_dofs_pos()[mask], fingers[mask])
    torch.testing.assert_close(cube.get_qpos()[mask], cube_qpos[mask])
    torch.testing.assert_close(cube.get_dofs_vel()[mask], cube_vel[mask])


def test_a_staged_write_of_every_envs_row_gathers_the_masked_ones_under_hibernation():
    """Genesis' hibernating setters take the selected rows only, so the staged write gathers them for it."""
    env = _hibernating_box_env()
    box = env.entities["box"]
    mask = torch.tensor([True, False, True])
    qpos_before, vel_before = box.get_qpos().clone(), box.get_dofs_vel().clone()
    qpos = qpos_before.clone()
    qpos[:, :3] += torch.tensor([0.5, -0.25, 0.75])
    vel = torch.full((3, 6), 0.3)

    write = StagedStateWrite()
    write.set_dofs_vel(box, vel, range(6))
    write.set_qpos(box, qpos, range(7))
    write.apply(mask)

    torch.testing.assert_close(box.get_qpos()[mask], qpos[mask])
    torch.testing.assert_close(box.get_dofs_vel()[mask], vel[mask])
    assert torch.equal(box.get_qpos()[~mask], qpos_before[~mask])
    assert torch.equal(box.get_dofs_vel()[~mask], vel_before[~mask])


def test_a_staged_write_takes_dofs_in_any_order(staged_env):
    """A motion's DOF order (reversed here) is written column by column, as ``set_dofs_pos`` would."""
    env = staged_env
    robot = env.entities["robot"]
    fingers = torch.tensor([[0.01, 0.03]] * 2, device=env.device)
    ids = torch.tensor([1, 3], device=env.device)

    write = StagedStateWrite()
    write.set_dofs_pos(robot, fingers, robot.dofs_idx_local.tolist()[::-1])
    write.apply(ids)

    torch.testing.assert_close(robot.get_dofs_pos()[ids], fingers.flip(-1))


def test_an_index_staged_twice_keeps_the_later_value(staged_env):
    env = staged_env
    cube = env.entities["cube"]
    envs_idx = torch.tensor([0, 1], device=env.device)

    write = StagedStateWrite()
    write.set_qpos(cube, _box_qpos(env, -1.0), range(7))
    write.set_qpos(cube, torch.tensor([[-1.5, 0.5, 2.0]] * 2, device=env.device), range(3))  # the position only
    write.apply(envs_idx)

    expected = torch.tensor([[-1.5, 0.5, 2.0, -0.8, 0.0, 0.6, 0.0]] * 2, device=env.device)
    torch.testing.assert_close(cube.get_qpos()[envs_idx], expected)


def test_a_kinematic_entity_is_written_in_its_own_solver(staged_env):
    """``ghost`` is a FrankaHand in the kinematic solver, whose DOF indices restart from 0 — the robot's."""
    env = staged_env
    robot, ghost = env.entities["robot"], env.entities["ghost"]
    envs_idx = torch.tensor([0, 2], device=env.device)
    robot_fingers = torch.tensor([[0.01, 0.02]] * 2, device=env.device)
    ghost_fingers = torch.tensor([[0.03, 0.035]] * 2, device=env.device)

    write = StagedStateWrite()
    write.set_dofs_pos(robot, robot_fingers, robot.dofs_idx_local.tolist())
    write.set_dofs_pos(ghost, ghost_fingers, ghost.dofs_idx_local.tolist())
    write.apply(envs_idx)

    torch.testing.assert_close(robot.get_dofs_pos()[envs_idx], robot_fingers)
    torch.testing.assert_close(ghost.get_dofs_pos()[envs_idx], ghost_fingers)


def test_a_dofs_only_write_leaves_link_velocities_from_the_new_dof_velocities(staged_env):
    """Only DOFs staged: the one forward pass, the position write's, must see the new velocities."""
    env = staged_env
    robot = env.entities["robot"]
    fingers = torch.tensor([[0.01, 0.02]] * 2, device=env.device)
    velocity = torch.tensor([[0.3, -0.2]] * 2, device=env.device)
    staged, direct = torch.tensor([1, 3], device=env.device), torch.tensor([0, 2], device=env.device)
    robot.set_dofs_vel(torch.full((NUM_ENVS, 2), -1.0, device=env.device))  # the velocities to leave behind

    write = StagedStateWrite()
    write.set_dofs_vel(robot, velocity, robot.dofs_idx_local.tolist())
    write.set_dofs_pos(robot, fingers, robot.dofs_idx_local.tolist())
    write.apply(staged)

    robot.set_dofs_pos(fingers, envs_idx=direct, zero_velocity=False)
    robot.set_dofs_vel(velocity, envs_idx=direct)

    torch.testing.assert_close(robot.get_links_vel()[staged], robot.get_links_vel()[direct])
    assert robot.get_links_vel()[staged].abs().max() > 0


def test_a_dof_write_leaves_the_quaternion_an_earlier_write_left(staged_env):
    """Two writes, the box's quaternion first: Genesis ``set_dofs_position`` would re-derive it as ``-q``.

    Two separate runs of RSI terms write this way (a throttled RSI term between two others, say).
    """
    env = staged_env
    robot, cube = env.entities["robot"], env.entities["cube"]
    envs_idx = torch.tensor([1, 2], device=env.device)

    first = StagedStateWrite()
    first.set_qpos(cube, _box_qpos(env, -1.0), range(7))
    first.apply(envs_idx)
    second = StagedStateWrite()
    second.set_dofs_pos(robot, torch.tensor([[0.01, 0.02]] * 2, device=env.device), robot.dofs_idx_local.tolist())
    second.apply(envs_idx)

    torch.testing.assert_close(cube.get_qpos()[envs_idx], _box_qpos(env, -1.0))


def test_a_free_joints_dof_position_leaves_what_its_own_setter_leaves(staged_env):
    """Its position as it is, its Euler angles as the quaternion Genesis' kernel makes of them."""
    env = staged_env
    cube = env.entities["cube"]
    dofs = torch.tensor([[-1.0, 0.5, 1.0, 0.3, -0.7, 2.5]] * 2, device=env.device)
    staged, direct = torch.tensor([1, 3], device=env.device), torch.tensor([0, 2], device=env.device)

    write = StagedStateWrite()
    write.set_dofs_pos(cube, dofs, range(6))
    write.apply(staged)
    cube.set_dofs_pos(dofs, range(6), envs_idx=direct, zero_velocity=False)

    torch.testing.assert_close(cube.get_qpos()[staged], cube.get_qpos()[direct])
    torch.testing.assert_close(cube.get_links_pos()[staged], cube.get_links_pos()[direct])


@pytest.mark.parametrize(
    "dofs_idx_local, match",
    [
        ([3, 4], r"joint '\w+' has rotation DOFs \[3, 4, 5\] \(entity-local\), and only \[3, 4\] are staged"),
        ([0, 6], r"entity 'cube' has 6 DOFs, so the entity-local DOF indices \[6\] select none of them"),
    ],
)
def test_a_dof_position_that_makes_no_qpos_is_refused(staged_env, dofs_idx_local, match):
    cube = staged_env.entities["cube"]
    with pytest.raises(ValueError, match=match):
        StagedStateWrite.check_dofs_pos(cube, dofs_idx_local)
    with pytest.raises(ValueError, match=match):
        StagedStateWrite().set_dofs_pos(cube, torch.zeros(2, len(dofs_idx_local)), dofs_idx_local)


# A fixed-base arm: a hinge with a reference angle (its qpos0 is 0.4 rad, not 0), then a ball joint.
_REF_ARM_MJCF = """<mujoco model="ref_arm"><compiler angle="radian"/><worldbody>
  <body name="base" pos="0 0 0.5"><geom type="box" size="0.05 0.05 0.05" mass="1.0"/>
    <body name="link" pos="0.1 0 0"><joint name="hinge" axis="0 0 1" ref="0.4" range="-2 2"/>
      <geom type="capsule" size="0.02" fromto="0 0 0 0.1 0 0" mass="0.1"/>
      <body name="tip" pos="0.1 0 0"><joint name="ball" type="ball"/>
        <geom type="capsule" size="0.02" fromto="0 0 0 0.1 0 0" mass="0.1"/></body></body></body>
</worldbody></mujoco>"""


@pytest.fixture(scope="module")
def ref_arm_env(tmp_path_factory):
    path = tmp_path_factory.mktemp("ref_arm") / "ref_arm.xml"
    path.write_text(_REF_ARM_MJCF)
    return build_env(_config(num_envs=4, arm=EntityOptions(file=str(path), is_fixed_base=True)))


def test_a_staged_dof_position_leaves_what_the_entity_setter_leaves_on_a_hinge_and_a_ball_joint(ref_arm_env):
    """The hinge's q is ``qpos0 + value``; the ball's three DOFs become one quaternion."""
    env = ref_arm_env
    arm = env.entities["arm"]
    hinge, ball = arm._entity.get_joint("hinge"), arm._entity.get_joint("ball")
    dofs_idx_local = [*ball.dofs_idx_local, *hinge.dofs_idx_local]  # out of joint order: each value to its own DOF
    position = torch.tensor([[0.3, -0.5, 1.2, 0.1]] * 2, device=env.device)
    staged, direct = torch.tensor([1, 3], device=env.device), torch.tensor([0, 2], device=env.device)

    write = StagedStateWrite()
    write.set_dofs_pos(arm, position, dofs_idx_local)
    write.apply(staged)
    arm.set_dofs_pos(position, dofs_idx_local, envs_idx=direct, zero_velocity=False)

    torch.testing.assert_close(arm.get_qpos()[staged], arm.get_qpos()[direct])
    torch.testing.assert_close(arm.get_qpos()[staged, hinge.qs_idx_local[0]], torch.full((2,), 0.5, device=env.device))
    torch.testing.assert_close(arm.get_dofs_pos(dofs_idx_local)[staged], position)
    torch.testing.assert_close(arm.get_links_pos()[staged], arm.get_links_pos()[direct])


# ═════════════════════════════════════════════════════════════════════════════════════════════════
# A GroupedEntityOptions variant's own values
#
# Genesis builds a heterogeneous entity from one material, so applying the group's geom and DOF settings
# to every variant would drop a variant's own value. Each env takes its variant's.
# Envs 0-1 hold variant 0 and envs 2-3 variant 1 throughout.
# ═════════════════════════════════════════════════════════════════════════════════════════════════

_CUBE = (0.1, 0.1, 0.1)  # 1 litre: its mass in kg is its density / 1000
_SOFT, _STIFF = (0.05, 1.0, 0.9, 0.95, 0.001, 0.5, 2.0), (0.01, 1.0, 0.9, 0.95, 0.001, 0.5, 2.0)


def _gizmo(asset, **options) -> EntityOptions:
    return EntityOptions(
        file=asset("gizmo"), is_articulated=True, is_actuated=True, dofs_name=["turntable_base_joint"], **options
    )


def _gizmo_bank(asset, armatures: tuple[float, float] = (0.02, 0.3)) -> GroupedEntityOptions:
    return GroupedEntityOptions(
        grouped_entities=[
            _gizmo(
                asset,
                default_dofs_kp={"*": 50.0},
                default_dofs_damping={"*": 0.1},
                default_dofs_pos={"*": 0.3},
                default_dofs_pos_limits={"*": (-1.0, 1.0)},
                default_armature=armatures[0],
            ),
            _gizmo(asset, default_dofs_kp={"*": 500.0}, default_dofs_pos={"*": -0.3}, default_armature=armatures[1]),
        ],
        default_dofs_damping={"*": 0.7},  # variant 1 takes the group's
    )


@pytest.fixture(scope="module")
def variant_env(mjcf_assets):
    return build_env(
        EdenConfig(
            # The defaults: batch_dofs_info is on, and Genesis stores a heterogeneous scene's link info per env
            # whatever batch_links_info says.
            env_options=EnvOptions(num_envs=4),
            scene_options=SceneOptions(
                boxes=GroupedEntityOptions(
                    material=RigidMaterialOptions(friction=0.5),
                    grouped_entities=[
                        BoxOptions(
                            size=_CUBE,
                            material=RigidMaterialOptions(rho=1000.0, friction=2.0),
                            default_geoms_sol_params={"*": _STIFF},
                        ),
                        BoxOptions(size=_CUBE),
                    ],
                    default_geoms_sol_params={"*": _SOFT},
                    default_root_pos=(0.0, 0.0, 0.5),
                ),
                # A body fixed to the world moves no mass, so its density is never read, as for an ungrouped one.
                **{
                    name: GroupedEntityOptions(
                        is_fixed_base=True,
                        grouped_entities=[
                            BoxOptions(size=_CUBE, material=RigidMaterialOptions(rho=rho)) for rho in rhos
                        ],
                        default_root_pos=(1.0, y, 0.5),
                    )
                    for name, rhos, y in (("fixed_boxes", (1000.0, 25.0), 0.0), ("fixed_plain", (200.0, 200.0), 1.0))
                },
                gizmos=_gizmo_bank(mjcf_assets),
                # Not grouped: it shares the solver whose link info Genesis batched for the groups.
                plain=BoxOptions(size=_CUBE, default_root_pos=(-1.0, 0.0, 0.5)),
            ),
        )
    )


def test_each_env_of_a_heterogeneous_entity_weighs_what_its_variants_density_makes_it(variant_env):
    boxes, fixed, plain = (variant_env.entities[name] for name in ("boxes", "fixed_boxes", "fixed_plain"))

    torch.testing.assert_close(
        boxes.get_links_inertial_mass()[:, 0].cpu(), torch.tensor([1.0, 1.0, 0.2, 0.2]), rtol=1e-5, atol=1e-7
    )
    # Left as Genesis built it, at the group's density (Genesis' ``set_mass`` refuses a body with no moving link).
    torch.testing.assert_close(fixed.get_links_inertial_mass(), plain.get_links_inertial_mass())


def test_each_variant_of_a_heterogeneous_entity_keeps_its_own_friction_and_sol_params(variant_env):
    """Variant 0 sets its own material friction and geom sol params; variant 1 takes the group's."""
    first, second = variant_env.entities["boxes"].geoms  # one geom per variant, active in its own envs

    assert (float(first.get_friction()), float(second.get_friction())) == (2.0, 0.5)
    torch.testing.assert_close(first.get_sol_params().cpu()[:2], torch.tensor(_STIFF[:2]))
    torch.testing.assert_close(second.get_sol_params().cpu()[:2], torch.tensor(_SOFT[:2]))


def test_each_env_of_an_articulated_group_takes_its_variants_dof_values(variant_env):
    gizmos = variant_env.entities["gizmos"]
    _, idx = gizmos.find_named_dofs_idx_local(["turntable_base_joint"])

    def per_env(values: torch.Tensor) -> list[float]:
        return [round(v, 4) for v in values.reshape(4).tolist()]

    assert per_env(gizmos.get_dofs_kp(dofs_idx_local=idx)) == [50.0, 50.0, 500.0, 500.0]
    assert per_env(gizmos.get_dofs_damping(dofs_idx_local=idx)) == [0.1, 0.1, 0.7, 0.7]
    assert per_env(gizmos.default_dofs_pos) == [0.3, 0.3, -0.3, -0.3]
    assert per_env(gizmos.get_dofs_pos()) == [0.3, 0.3, -0.3, -0.3]
    # Each variant's own default armature, which Genesis fills per env; post_build used to write env 0's over all.
    assert per_env(gizmos.get_dofs_armature(dofs_idx_local=idx)) == [0.02, 0.02, 0.3, 0.3]
    # Variant 1 keeps its asset's range; the soft range is shrunk about each env's own.
    assert gizmos.default_dofs_pos_limits[:, 0].tolist() == [[-1.0, 1.0]] * 2 + [[-3.0, 3.0]] * 2
    torch.testing.assert_close(
        gizmos.soft_dofs_pos_limits[:, 0].cpu(), torch.tensor([[-0.9, 0.9]] * 2 + [[-2.7, 2.7]] * 2)
    )


def test_an_ungrouped_entity_beside_a_group_reads_and_shifts_its_link_info_per_env(variant_env):
    """Genesis batches link info for a scene holding a heterogeneous entity whatever ``batch_links_info`` says;
    Eden's getters followed the option and broadcast the solver's per-env rows as if they were per link.
    """
    plain = variant_env.entities["plain"]
    shift = torch.tensor([[0.1], [0.2], [0.3], [0.4]], device=variant_env.device)

    assert tuple(plain.get_links_inertial_mass().shape) == (4, 1)
    plain.set_mass_shift(shift, ls_idx_local=[0])
    try:
        torch.testing.assert_close(plain.get_mass_shift(ls_idx_local=[0]), shift)
    finally:
        plain.set_mass_shift(torch.zeros_like(shift), ls_idx_local=[0])


def test_variants_that_differ_in_a_simulated_dof_value_build_without_batch_dofs_info(mjcf_assets):
    """Genesis stores a heterogeneous scene's DOF info per env whatever the option says.

    Eden refused this config before the build while Genesis kept one row for every env (until
    https://github.com/Genesis-Embodied-AI/Genesis/pull/3497).
    """
    bank = _gizmo_bank(mjcf_assets, armatures=(0.1, 0.1))
    env = build_env(
        EdenConfig(env_options=EnvOptions(num_envs=2, batch_dofs_info=False), scene_options=SceneOptions(gizmos=bank))
    )
    gizmos = env.entities["gizmos"]
    _, idx = gizmos.find_named_dofs_idx_local(["turntable_base_joint"])

    assert gizmos.has_batched_dofs_info
    assert gizmos.get_dofs_damping(dofs_idx_local=idx).reshape(2).tolist() == pytest.approx([0.1, 0.7])


def test_a_file_variant_with_its_own_density_fails_before_the_build(mjcf_assets):
    """Genesis reads an MJCF body's mass from its file, so no density rescale reaches it."""
    bank = GroupedEntityOptions(
        grouped_entities=[_gizmo(mjcf_assets), _gizmo(mjcf_assets, material=RigidMaterialOptions(rho=25.0))]
    )

    # One env holds only the first variant; the second is refused all the same, so no env count hides it.
    with pytest.raises(ValueError, match=r"grouped_entities\[1\] \(EntityOptions\) has its own rho=25.0.*MJCF"):
        build_env(_config(num_envs=1, bank=bank))


def test_a_variants_own_torsional_friction_needs_the_scene_switch():
    bank = GroupedEntityOptions(
        grouped_entities=[BoxOptions(), BoxOptions(material=RigidMaterialOptions(friction_torsional=0.01))]
    )

    with pytest.raises(ValueError, match=r"`material.friction_torsional` set but torsional friction is disabled"):
        build_env(_config(num_envs=1, bank=bank))


def test_variant_force_limits_reach_each_env_when_genesis_batches_the_dof_info_itself(mjcf_assets):
    """Variants whose default armatures differ make Genesis batch DOF info under ``batch_dofs_info=False``; the
    force range must follow the solver, not the option, or every env takes the first variant's.
    """
    bank = _gizmo_bank(mjcf_assets)  # default armatures 0.02 / 0.3
    for limit, variant in zip((5.0, 50.0), bank.grouped_entities):
        variant.default_dofs_force_limits = {"*": limit}
    gizmos = build_env(
        EdenConfig(env_options=EnvOptions(num_envs=2, batch_dofs_info=False), scene_options=SceneOptions(gizmos=bank))
    ).entities["gizmos"]
    _, idx = gizmos.find_named_dofs_idx_local(["turntable_base_joint"])

    assert gizmos.get_dofs_force_range(dofs_idx_local=idx)[1].reshape(-1).tolist() == [5.0, 50.0]


def test_a_spring_reference_moves_a_set_stiffness_onto_the_servo_with_no_kp_set(mjcf_assets):
    """An ungrouped entity under the default ``batch_dofs_info``; a per-env stiffness row once failed to fit kp."""
    options = _gizmo(mjcf_assets, default_dofs_stiffness={"*": 2.0}, default_dofs_spring_ref={"*": 0.5})
    gizmo = build_env(_config(num_envs=2, gizmo=options)).entities["gizmo"]
    _, idx = gizmo.find_named_dofs_idx_local(["turntable_base_joint"])

    assert gizmo.get_dofs_kp(dofs_idx_local=idx).reshape(-1).tolist() == [2.0, 2.0]
    assert gizmo.get_dofs_stiffness(dofs_idx_local=idx).reshape(-1).tolist() == [0.0, 0.0]
    assert gizmo.default_dofs_kp[:, 0].tolist() == [2.0] * gizmo.default_dofs_kp.shape[0]  # dofs_name order
