"""``RobotStateEntity``: real-robot state behind the entity getters, synced into a real sim entity."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from eden.extensions.deployment.utils.state import RobotState, RobotStateEntity
from eden.options.entities import RESOLVED_DOF_TENSOR_FIELD_NAMES

FLOATING_TWO_LINK_MJCF = """
<mujoco model="floating_two_link">
  <worldbody>
    <body name="base" pos="0 0 0.74">
      <freejoint name="root"/>
      <geom name="root" type="sphere" size="0.05" mass="1.0"/>
      <body name="link1">
        <joint name="j1" type="hinge" axis="0 1 0"/>
        <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.03" mass="0.5"/>
        <body name="link2" pos="0.3 0 0">
          <joint name="j2" type="hinge" axis="0 1 0"/>
          <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.03" mass="0.5"/>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


@pytest.fixture(scope="module")
def env(tmp_path_factory, two_link_asset):
    """A floating two-link robot, and an articulated but non-actuated ``prop`` (no gains resolved)."""
    from eden.options.entities import EntityOptions, RobotOptions
    from eden.utils.configs import EdenConfig, EnvOptions, SceneOptions
    from tests.support.envs import build_env

    asset = tmp_path_factory.mktemp("floating") / "floating_two_link.xml"
    asset.write_text(FLOATING_TWO_LINK_MJCF)
    return build_env(
        EdenConfig(
            env_options=EnvOptions(num_envs=1),
            scene_options=SceneOptions(
                robot=RobotOptions(
                    file=str(asset),
                    dofs_name=["j1", "j2"],
                    default_dofs_pos={".*": 0.0},
                    default_dofs_kp={".*": 10.0},
                    default_dofs_kd={".*": 1.0},
                    default_root_pos=(0.0, 0.0, 0.74),
                ),
                prop=EntityOptions(
                    file=str(two_link_asset), is_articulated=True, is_fixed_base=True, dofs_name=["j1", "j2"]
                ),
            ),
        )
    )


@pytest.fixture
def sim(env):
    """The sim robot, restored to its built pose afterwards (the wrapper writes its state into it)."""
    robot = env.entities["robot"]
    pos, quat = robot.get_pos().clone(), robot.get_quat(relative=True).clone()
    yield robot
    robot.set_pos(pos)
    robot.set_quat(quat)
    robot.set_dofs_pos(torch.zeros(1, 2))
    robot.set_dofs_vel(torch.zeros(1, 2))


@pytest.fixture
def rse(sim) -> RobotStateEntity:
    return RobotStateEntity(sim, device=sim.device)


def _yaw(yaw: float) -> np.ndarray:
    return np.array([np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)])


def _state(base_quat=(1.0, 0.0, 0.0, 0.0), dofs_pos=(0.0, 0.0), **extra) -> RobotState:
    return RobotState(
        stamp=0.0,
        base_quat=np.asarray(base_quat, dtype=np.float64),
        base_ang_vel=np.zeros(3),
        base_lin_acc=np.zeros(3),
        dofs_pos=np.asarray(dofs_pos, dtype=np.float64),
        dofs_vel=np.zeros(2),
        dofs_torque=np.zeros(2),
        extra=extra,
    )


def _np(tensor) -> np.ndarray:
    return tensor[0].detach().cpu().numpy()


# --- yaw alignment --------------------------------------------------------------------------


def test_with_no_offset_the_reading_passes_through(rse):
    assert rse.yaw_offset == 0.0
    rse.update(_state(_yaw(0.7)))
    np.testing.assert_allclose(_np(rse.get_quat()), _yaw(0.7), atol=1e-6)


@pytest.mark.parametrize("yaw", [-1.2, -0.3, 0.5, 1.6])
def test_a_captured_yaw_is_removed_from_every_later_reading(rse, sim, yaw):
    """The world frame the policy trained in, and the one the sim entity's FK is computed in."""
    assert rse.capture_yaw_offset(_yaw(yaw)) == pytest.approx(yaw, abs=1e-6)

    rse.update(_state(_yaw(yaw), dofs_pos=(0.2, -0.1)))

    np.testing.assert_allclose(_np(rse.get_quat()), [1.0, 0.0, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(_np(sim.get_quat(relative=True)), [1.0, 0.0, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(_np(sim.get_dofs_pos()), [0.2, -0.1], atol=1e-6)


def test_capture_rejects_anything_but_one_quaternion(rse):
    with pytest.raises(ValueError, match="single wxyz quaternion"):
        rse.capture_yaw_offset(np.zeros((2, 4)))


def test_the_offset_rotates_world_frame_readings_but_not_the_body_rate(rse):
    """Base at world (1, 0, 0) moving along +x while facing +y: both read as -y once the yaw is gone."""
    rse.set_yaw_offset(np.pi / 2)
    state = _state(_yaw(np.pi / 2), base_pos=np.array([1.0, 0.0, 0.0]), base_lin_vel=np.array([1.0, 0.0, 0.0]))
    state.base_ang_vel = np.array([0.1, 0.2, 0.3])

    rse.update(state)

    np.testing.assert_allclose(_np(rse.get_pos()), [0.0, -1.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(_np(rse.get_vel(frame="world")), [0.0, -1.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(_np(rse.get_ang(frame="body")), [0.1, 0.2, 0.3], atol=1e-6)


def test_a_held_reading_is_not_rotated_again_through_an_estimator_dropout(rse, sim):
    rse.set_yaw_offset(np.pi / 2)
    rse.update(_state(_yaw(np.pi / 2), base_pos=np.array([1.0, 0.0, 0.74]), base_lin_vel=np.array([1.0, 0.0, 0.0])))

    for _ in range(3):  # no base_pos / base_lin_vel in these
        rse.update(_state(_yaw(np.pi / 2)))

    np.testing.assert_allclose(_np(rse.get_pos()), [0.0, -1.0, 0.74], atol=1e-6)
    np.testing.assert_allclose(_np(rse.get_vel(frame="world")), [0.0, -1.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(_np(sim.get_pos()), [0.0, -1.0, 0.74], atol=1e-6, err_msg="FK holds the last pose")


# --- the base position ------------------------------------------------------------------------


def test_the_position_is_the_spawn_pose_until_a_reading_supplies_one(rse, sim):
    """Pushing a synthetic origin into the FK would anchor every world-frame link read at the origin."""
    spawn = _np(sim.get_pos()).copy()
    assert not rse.base_pos_known
    np.testing.assert_allclose(_np(rse.get_pos()), spawn, atol=1e-6)

    rse.set_yaw_offset(np.pi / 2)
    rse.update(_state(_yaw(np.pi / 2), dofs_pos=(0.3, 0.3)))

    np.testing.assert_allclose(_np(sim.get_pos()), spawn, atol=1e-6, err_msg="the sim base was not moved")
    np.testing.assert_allclose(_np(rse.get_pos()), spawn, atol=1e-6, err_msg="and the spawn pose is not yaw-rotated")
    assert not rse.base_pos_known


def test_a_reading_with_a_position_moves_the_sim_base(rse, sim):
    rse.update(_state(base_pos=np.array([1.5, -0.2, 0.74])))

    assert rse.base_pos_known
    np.testing.assert_allclose(_np(rse.get_pos()), [1.5, -0.2, 0.74], atol=1e-6)
    np.testing.assert_allclose(_np(sim.get_pos()), [1.5, -0.2, 0.74], atol=1e-6)


# --- pose getters -----------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["get_pos", "get_quat", "get_T", "get_links_pos", "get_links_quat"])
def test_relative_pose_access_is_refused(rse, method):
    with pytest.raises(NotImplementedError, match="does not support relative=True pose access"):
        getattr(rse, method)(relative=True)


def test_world_frame_link_and_transform_reads_come_from_the_synced_sim_entity(rse, sim):
    rse.update(_state(_yaw(0.4), dofs_pos=(0.5, -0.5), base_pos=np.array([0.3, 0.1, 0.9])))
    links = torch.tensor([1, 2])

    torch.testing.assert_close(rse.get_T(relative=False), sim.get_T(relative=False))
    torch.testing.assert_close(rse.get_links_pos(ls_idx_local=links), sim.get_links_pos(ls_idx_local=links))
    torch.testing.assert_close(rse.get_links_quat(ls_idx_local=links), sim.get_links_quat(ls_idx_local=links))
    np.testing.assert_allclose(_np(rse.get_T(relative=False))[:3, 3], _np(sim.get_pos()), atol=1e-6)


# --- DOF reads by the sim entity's local indices ------------------------------------------------

#: ``j2`` then ``j1``: local 7 and 6 behind the free base, columns 1 and 0 of the wrapper's state.
_J2_J1 = {"tensor": torch.tensor([7, 6]), "list": [7, 6], "range": range(7, 5, -1)}


@pytest.mark.parametrize("form", list(_J2_J1))
def test_a_local_index_reads_its_joint_in_any_form(rse, form):
    rse.update(_state(dofs_pos=(0.2, -0.1)))
    for _ in range(2):  # the second read comes from the memo
        np.testing.assert_allclose(_np(rse.get_dofs_pos(dofs_idx_local=_J2_J1[form])), [-0.1, 0.2], atol=1e-6)


def test_an_index_tensor_written_in_place_is_read_again(rse):
    rse.update(_state(dofs_pos=(0.2, -0.1)))
    idx = torch.tensor([6, 7])
    np.testing.assert_allclose(_np(rse.get_dofs_pos(dofs_idx_local=idx)), [0.2, -0.1], atol=1e-6)
    idx[0] = 7
    np.testing.assert_allclose(_np(rse.get_dofs_pos(dofs_idx_local=idx)), [-0.1, -0.1], atol=1e-6)


def test_a_local_index_the_robot_does_not_control_raises(rse):
    """The free base's DOFs (local 0-5) have no state here: named, rather than a bare ``KeyError``."""
    with pytest.raises(ValueError, match=r"dofs_idx_local \[0\] are not DOFs this robot controls"):
        rse.get_dofs_pos(dofs_idx_local=[0, 6])


# --- the configured per-DOF defaults ----------------------------------------------------------


def test_every_resolved_per_dof_default_is_a_snapshot_of_the_sim_entity(rse, sim):
    """Env 0's row, cloned at connect time, so nothing read through the wrapper aliases the sim entity's memory."""
    for name in RESOLVED_DOF_TENSOR_FIELD_NAMES:
        ours, theirs = getattr(rse, name), getattr(sim, name)
        assert torch.is_tensor(ours), name
        torch.testing.assert_close(ours, theirs[:1], msg=name)
        assert ours.data_ptr() != theirs.data_ptr(), f"{name} aliases the sim entity"


def test_the_default_dofs_getters_read_the_wrappers_own_copies(rse, sim):
    """Not the sim entity's tensors: a later write there does not reach a read through the wrapper."""
    before = rse.get_default_dofs_kp(["j1"])
    torch.testing.assert_close(before, torch.tensor([[10.0]], device=before.device))
    sim.default_dofs_kp.add_(1.0)
    try:
        torch.testing.assert_close(rse.get_default_dofs_kp(["j1"]), before)
    finally:
        sim.default_dofs_kp.sub_(1.0)


@pytest.fixture(scope="module")
def two_env_robot(tmp_path_factory):
    """The floating two-link robot in two envs: its configured ``default_dofs_pos`` is stored one row per env."""
    from eden.options.entities import RobotOptions
    from eden.utils.configs import EdenConfig, EnvOptions, SceneOptions
    from tests.support.envs import build_env

    asset = tmp_path_factory.mktemp("two_envs") / "floating_two_link.xml"
    asset.write_text(FLOATING_TWO_LINK_MJCF)
    env = build_env(
        EdenConfig(
            env_options=EnvOptions(num_envs=2),
            scene_options=SceneOptions(
                robot=RobotOptions(
                    file=str(asset),
                    dofs_name=["j1", "j2"],
                    default_dofs_pos={"j1": 0.1, "j2": 0.2},
                    default_dofs_kp={".*": 10.0},
                    default_dofs_kd={".*": 1.0},
                    default_root_pos=(0.0, 0.0, 0.74),
                )
            ),
        )
    )
    return env.entities["robot"]


def test_the_wrapper_is_one_env_its_sim_entitys_env_0(two_env_robot):
    """Like its state getters (``get_pos`` is ``(1, 3)``), the wrapper's defaults have one env: env 0's row."""
    robot = two_env_robot
    robot.default_dofs_pos[1] += 0.5  # env 1 drifts, as RandomizeStartupDofsPosBias would
    try:
        rse = RobotStateEntity(robot, device=robot.device)
        assert tuple(rse.default_dofs_pos.shape) == (1, 2)
        torch.testing.assert_close(rse.get_default_dofs_pos(), robot.get_default_dofs_pos(envs_idx=[0]))
        torch.testing.assert_close(rse.get_default_dofs_kp(["j2"]), robot.get_default_dofs_kp(["j2"], envs_idx=[0]))
    finally:
        robot.default_dofs_pos[1] -= 0.5


def test_a_property_the_wrapper_holds_no_copy_of_raises_as_on_an_entity(rse):
    """The getters' ``ValueError``, not the wrapper's ``__getattr__`` ``AttributeError``.

    No sim entity a deployment accepts lacks one of the eleven (every actuated, simulated entity resolves all of
    them), so the copy is removed by hand, standing in for one the sim entity did not carry.
    """
    del rse.default_dofs_frictionloss
    with pytest.raises(ValueError, match="'robot' has no resolved default_dofs_frictionloss"):
        rse.get_default_dofs_frictionloss()


def test_an_entity_with_no_resolved_gains_is_refused_at_connect(env):
    """A non-actuated entity resolves no kp/kd, so it is not a robot a deployment can drive."""
    with pytest.raises(ValueError, match="no resolved default_dofs_kp"):
        RobotStateEntity(env.entities["prop"], device=env.device)


def test_everything_else_falls_through_to_the_sim_entity(rse, sim):
    assert rse.get_link("link2") is sim.get_link("link2")
    assert rse.find_named_links_idx_local(["link2"]) == sim.find_named_links_idx_local(["link2"])
