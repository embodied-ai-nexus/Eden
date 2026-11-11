"""The manipulation observation terms in ``eden/managers/terms/observations/manipulation.py``."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from eden.managers.command_manager import CommandManager
from eden.managers.terms.commands.oracle import UniformT3Command
from eden.managers.terms.observations.manipulation import (
    EndEffectorPose,
    EndEffectorToObjectDistance,
    GripperState,
    ObjectToOracleDistance,
    gripper_state,
    rotation_as,
)
from eden.options.managers.commands import CommandManagerOptions

pytestmark = pytest.mark.requires_genesis

FINGERS = {"left_ee_link_name": "left_finger", "right_ee_link_name": "right_finger"}


def _fingertip_midpoint(robot):
    _, idx = robot.find_named_links_idx_local(["left_finger", "right_finger"])
    return robot.get_links_pos(ls_idx_local=idx).mean(dim=1)


def test_ee_to_object_distance_is_the_object_minus_the_finger_midpoint(panda_env, make_term, restore_state):
    cube = restore_state(panda_env.entities["cube_a"])
    offset = torch.tensor([[0.1, 0.2, 0.3], [-0.3, 0.0, 0.1]])
    cube.set_pos(_fingertip_midpoint(panda_env.entities["robot"]) + offset)
    term = make_term(EndEffectorToObjectDistance, panda_env, entity_name="cube_a", frame="world", **FINGERS)
    torch.testing.assert_close(term.compute(), offset)


def test_ee_to_object_distance_follows_a_swapped_object(panda_env, make_term):
    """The object is resolved per ``compute()``, so a deployment-style swap of ``env.entities`` is seen."""
    term = make_term(EndEffectorToObjectDistance, panda_env, entity_name="cube_a", frame="world", **FINGERS)
    cube_a = panda_env.entities["cube_a"]
    panda_env.entities["cube_a"] = panda_env.entities["cube_b"]
    try:
        out = term.compute()
    finally:
        panda_env.entities["cube_a"] = cube_a
    expected = panda_env.entities["cube_b"].get_pos() - _fingertip_midpoint(panda_env.entities["robot"])
    torch.testing.assert_close(out, expected)


@pytest.fixture
def oracle_command(panda_env, install_manager):
    """A real ``UniformT3Command`` named ``command``, holding (1, 0, 0) in its frame entity's frame."""
    options = CommandManagerOptions(command=UniformT3Command.configure())
    manager = install_manager(panda_env, "command_manager", CommandManager(panda_env, options))
    manager.get_term("command").oracle_pos_b[:] = torch.tensor([1.0, 0.0, 0.0])


@pytest.mark.parametrize(("frame", "expected"), [("world", [1.0, 1.0, 0.0]), ("robot", [1.0, -1.0, 0.0])])
def test_object_to_oracle_distance_in_either_frame(
    panda_env, make_term, restore_state, oracle_command, frame, expected
):
    """``cube_b`` stands in as the robot base, at (1, 0, 0) yawed +90 deg, so the frames differ.

    The oracle (1, 0, 0) in that frame is world (1, 1, 0); the object sits at (0, 0, 0).
    """
    base, obj = restore_state(panda_env.entities["cube_b"]), restore_state(panda_env.entities["cube_a"])
    base.set_pos(torch.tensor([[1.0, 0.0, 0.0]] * 2))
    base.set_quat(torch.tensor([[math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)]] * 2))
    obj.set_pos(torch.zeros(2, 3))
    term = make_term(ObjectToOracleDistance, panda_env, entity_name="cube_a", robot_name="cube_b", frame=frame)
    torch.testing.assert_close(term.compute(), torch.tensor([expected] * 2), atol=1e-6, rtol=0.0)


@pytest.mark.parametrize(
    ("term_cls", "params"),
    [
        (EndEffectorToObjectDistance, FINGERS),
        (ObjectToOracleDistance, {"command_name": "command"}),
    ],
)
def test_a_missing_object_entity_fails_at_build(panda_env, make_term, oracle_command, term_cls, params):
    with pytest.raises(KeyError, match="nope"):
        make_term(term_cls, panda_env, entity_name="nope", **params)


# ---------------------------------------------------------------------------
# Policy-facing state terms: EndEffectorPose, GripperState (arm joints: DofsPos, test_proprio.py)
# ---------------------------------------------------------------------------

_ARM = [f"joint{i}" for i in range(1, 8)]
_ARM_Q = [0.3, -0.4, 0.2, -2.0, 0.1, 1.5, 0.6]
_FINGERS_Q = {"finger_joint1": 0.03, "finger_joint2": 0.01}


@pytest.fixture
def posed_panda(panda_env, restore_state):
    """``panda_env`` with the arm and fingers at a fixed, non-trivial configuration."""
    robot = restore_state(panda_env.entities["robot"])
    pos = robot.get_dofs_pos().clone()
    for name, value in [*zip(_ARM, _ARM_Q), *_FINGERS_Q.items()]:
        pos[:, robot.dofs_name.index(name)] = value
    robot.set_dofs_pos(pos)
    return panda_env


def _link_pose(robot, link_name):
    _, (idx,) = robot.find_named_links_idx_local([link_name])
    return robot.get_links_pos(ls_idx_local=[idx])[:, 0], robot.get_links_quat(ls_idx_local=[idx])[:, 0]


@pytest.mark.parametrize(
    ("params", "expected"),
    [
        (dict(normalize="raw"), [0.03, 0.01]),
        (dict(normalize="openness", open_pos=0.04, close_pos=0.0), [0.75, 0.25]),
        (dict(normalize="closedness", open_pos=0.04, close_pos=0.0), [0.25, 0.75]),
        (dict(scale=2.0), [0.06, 0.02]),
    ],
)
def test_gripper_state_normalises_the_measured_fingers(posed_panda, make_term, params, expected):
    term = make_term(GripperState, posed_panda, entity_name="robot", dofs_name=list(_FINGERS_Q), **params)
    torch.testing.assert_close(term.compute(), torch.tensor([expected] * 2))


def test_end_effector_pose_in_world_frame_with_a_tool_offset(posed_panda, make_term):
    pos_w, quat_w = _link_pose(posed_panda.entities["robot"], "hand")
    world = make_term(EndEffectorPose, posed_panda, entity_name="robot", ee_link_name="hand", frame="world")
    torch.testing.assert_close(world.compute(), torch.cat([pos_w, quat_w], dim=-1))
    shifted = make_term(
        EndEffectorPose, posed_panda, entity_name="robot", ee_link_name="hand", frame="world", offset_pos=(0.1, 0, 0)
    ).compute()
    link_rot = Rotation.from_quat(np.roll(quat_w[0].numpy(), -1))
    np.testing.assert_allclose(shifted[0, :3].numpy(), pos_w[0].numpy() + link_rot.apply([0.1, 0.0, 0.0]), atol=1e-6)


def test_end_effector_pose_in_base_frame_is_the_link_relative_to_the_robot_root(posed_panda, make_term):
    robot = posed_panda.entities["robot"]
    pos_w, quat_w = _link_pose(robot, "hand")
    base_rot = Rotation.from_quat(np.roll(robot.get_quat()[0].numpy(), -1))
    expected_pos = base_rot.inv().apply(pos_w[0].numpy() - robot.get_pos()[0].numpy())
    expected_rot = base_rot.inv() * Rotation.from_quat(np.roll(quat_w[0].numpy(), -1))
    out = make_term(
        EndEffectorPose, posed_panda, entity_name="robot", ee_link_name="hand", rotation="euler_xyz"
    ).compute()
    np.testing.assert_allclose(out[0, :3].numpy(), expected_pos, atol=1e-5)
    np.testing.assert_allclose(out[0, 3:].numpy(), expected_rot.as_euler("xyz"), atol=1e-5)


def test_rotation_formats_are_consistent():
    quat = torch.tensor(np.roll(Rotation.from_euler("xyz", [0.3, -0.2, 1.1]).as_quat(), 1)[None], dtype=torch.float32)
    np.testing.assert_allclose(rotation_as(quat, "euler_xyz")[0], [0.3, -0.2, 1.1], atol=1e-6)
    rotvec = Rotation.from_euler("xyz", [0.3, -0.2, 1.1]).as_rotvec()
    np.testing.assert_allclose(rotation_as(quat, "axis_angle")[0], rotvec, atol=1e-6)
    r = Rotation.from_euler("xyz", [0.3, -0.2, 1.1]).as_matrix()
    np.testing.assert_allclose(rotation_as(quat, "rot6d")[0], r[:2].reshape(6), atol=1e-6)
    with pytest.raises(ValueError, match="rotation format"):
        rotation_as(quat, "matrix")


def test_gripper_state_normalisation_covers_both_finger_conventions():
    widowx = torch.tensor([[0.037, 0.0, 0.0185]])
    torch.testing.assert_close(
        gripper_state(widowx, normalize="openness", open_pos=0.037, close_pos=0.0), torch.tensor([[1.0, 0.0, 0.5]])
    )
    robotiq = torch.tensor([[0.0, 0.8, 0.2]])
    torch.testing.assert_close(
        gripper_state(robotiq, normalize="closedness", open_pos=0.0, close_pos=0.8), torch.tensor([[0.0, 1.0, 0.25]])
    )
    with pytest.raises(ValueError, match="distinct open_pos and close_pos"):
        gripper_state(robotiq, normalize="closedness", open_pos=0.5, close_pos=0.5)


@pytest.mark.parametrize(
    ("cls", "params", "error", "match"),
    [
        ("EndEffectorPose", dict(ee_link_name="not_a_link"), ValueError, "not_a_link"),
        ("EndEffectorPose", dict(ee_link_name="hand", rotation="matrix"), ValueError, "rotation"),
        ("GripperState", dict(dofs_name=["finger_joint1"], normalize="percent"), ValueError, "normalize"),
        ("GripperState", dict(dofs_name=["finger_joint1"], normalize="closedness"), ValueError, "open_pos/close_pos"),
    ],
)
def test_state_terms_fail_fast_on_bad_names_and_options(panda_env, make_term, cls, params, error, match):
    term_cls = {"EndEffectorPose": EndEffectorPose, "GripperState": GripperState}
    with pytest.raises(error, match=match):
        make_term(term_cls[cls], panda_env, entity_name="robot", **params)


@pytest.fixture(scope="module")
def leap_env():
    """A LEAP hand: its declared ``dofs_name`` order differs from Genesis' kinematic-tree order."""
    import eden as en
    from eden.options import EnvOptions
    from eden.options.robots.leap_hand import LeapHand_R
    from eden.utils.configs import EdenConfig
    from tests.support.envs import build_env

    return build_env(EdenConfig(env_options=EnvOptions(num_envs=1), scene_options=en.scenes.Empty(robot=LeapHand_R())))


def test_gripper_state_reads_by_genesis_index_when_dofs_name_order_differs(leap_env, make_term, restore_state):
    hand = restore_state(leap_env.entities["robot"])
    _, genesis_idx = hand.find_named_dofs_idx_local(list(hand.dofs_name), preserve_order=True)
    assert list(genesis_idx) != list(range(len(genesis_idx))), "the LEAP hand no longer reorders its DOFs"
    values = torch.linspace(0.1, 0.1 * len(hand.dofs_name), len(hand.dofs_name)).unsqueeze(0)
    hand.set_dofs_pos(values)  # dofs_name order
    names = list(hand.dofs_name[:4])
    gripper = make_term(
        GripperState,
        leap_env,
        entity_name="robot",
        dofs_name=[names[1]],
        normalize="closedness",
        open_pos=0.0,
        close_pos=1.0,
    ).compute()
    torch.testing.assert_close(gripper, values[:, 1:2])
