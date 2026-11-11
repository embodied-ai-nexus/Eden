"""The OmniReset tasks, built: the peg workspace, the composite training rig, and UWLab's rewards.

The envs restore rows written here (``omnireset.datasets`` layout), so nothing from UWLab is downloaded.
"""

from __future__ import annotations

import pytest
import torch

from eden.tasks.omnireset.config import scene_entity_options
from eden.tasks.omnireset.constants import (
    DECIMATION,
    GRIPPER,
    INSERTIVE,
    INSERTIVE_MASS,
    ISAAC_JOINT_ORDER,
    MOUNT,
    PAIRS,
    RECEPTIVE,
    RIGS,
    ROBOT,
    ROBOT_FRICTION,
    SIM_DT,
    SIM_SUBSTEPS,
    UR5E_SYSID_ARMATURE,
)
from eden.managers.terms.events.state_buffer import ResetFromStateBuffer
from eden.tasks.registry import TASK_REGISTRY
from eden.utils.buffers.reset_states import EntityStates, ResetStateBuffer
from tests.support.envs import build_env

IDENTITY = [1.0, 0.0, 0.0, 0.0]
YAW_90 = [0.70710678, 0.0, 0.0, 0.70710678]
ARM_HOME = [0.0, -1.5708, 1.5708, -1.5708, -1.5708, -1.5708]
HOME = ARM_HOME + [0.0] * 6
# Row 0 leaves the peg lying on the table (centre 15 mm above the top at z = -0.013); row 1 moves every placed body.
ROWS = ResetStateBuffer(
    {
        ROBOT: EntityStates(
            root_pos=torch.tensor([[0.0, -0.039, 0.0], [0.008, -0.02, -0.006]]),
            root_quat=torch.tensor([IDENTITY, IDENTITY]),
            joints_name=ISAAC_JOINT_ORDER,
            joints_n_dofs=(1,) * 12,
            dofs_pos=torch.tensor([HOME, [0.3, *HOME[1:]]]),
            dofs_vel=torch.zeros(2, 12),
        ),
        MOUNT: EntityStates(
            root_pos=torch.tensor([[0.0, -0.039, -0.0129], [0.008, -0.02, -0.0189]]),
            root_quat=torch.tensor([IDENTITY, IDENTITY]),
        ),
        RECEPTIVE: EntityStates(
            root_pos=torch.tensor([[0.4, 0.1, 0.0022], [0.5, 0.25, 0.0022]]), root_quat=torch.tensor([IDENTITY, YAW_90])
        ),
        INSERTIVE: EntityStates(
            root_pos=torch.tensor([[0.45, -0.15, 0.0021], [0.6, -0.3, 0.25]]),
            root_quat=torch.tensor([[0.70710678, 0.70710678, 0.0, 0.0], YAW_90]),  # row 0 lies on a long side
        ),
    }
)


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    from eden.options import EnvOptions, SceneOptions
    from eden.utils.configs import EdenConfig, EventManagerOptions

    path = tmp_path_factory.mktemp("omnireset") / "rows.pt"
    ROWS.save(path)
    return build_env(
        EdenConfig(
            env_options=EnvOptions(num_envs=2, sim_dt=SIM_DT, sim_substeps=SIM_SUBSTEPS),
            scene_options=SceneOptions(**scene_entity_options(PAIRS["peg"])),
            event_options=EventManagerOptions(restore=ResetFromStateBuffer.configure(path=str(path))),
        )
    )


@pytest.mark.requires_genesis
def test_a_reset_places_the_robot_base_mount_fixture_and_object_of_each_row(env):
    restore = env.event_manager.get_term("restore")
    env.reset()
    rows = restore.current_config.cpu()
    for name in (ROBOT, MOUNT, RECEPTIVE, INSERTIVE):
        landed = env.entities[name].get_pos(relative=True).cpu()
        torch.testing.assert_close(landed, ROWS.entities[name].root_pos[rows], atol=1e-5, rtol=0, msg=name)
    robot = env.entities[ROBOT]
    _, joints = robot.find_named_dofs_idx_local(list(ISAAC_JOINT_ORDER))
    torch.testing.assert_close(robot.get_dofs_pos(dofs_idx_local=joints).cpu(), ROWS.entities[ROBOT].dofs_pos[rows])


@pytest.mark.requires_genesis
def test_a_peg_restored_onto_the_table_rests_there(env):
    restore = env.event_manager.get_term("restore")
    torch.manual_seed(0)
    for _ in range(20):  # uniform rows: reset until an env starts from the resting row 0
        env.reset()
        if (restore.current_config == 0).any():
            break
    resting = int(torch.nonzero(restore.current_config == 0)[0])
    peg = env.entities[INSERTIVE]
    before = peg.get_pos(relative=True)[resting].clone()
    for _ in range(round(0.5 / SIM_DT)):
        env.scene.step()
    moved = (peg.get_pos(relative=True)[resting] - before).norm().item()
    assert moved < 2e-3, f"the peg moved {moved * 1e3:.1f} mm off its resting pose on the table in 0.5 s"


@pytest.mark.requires_genesis
def test_the_peg_weighs_the_midpoint_of_uwlabs_mass_draw(env):
    # Genesis reads an MJCF body's mass from the file, so the converted asset must carry it: without an <inertial> it
    # would take MuJoCo's default geom density over the visual and collision meshes, whatever the scene's material says.
    assert float(env.entities[INSERTIVE].get_mass()) == pytest.approx(INSERTIVE_MASS, abs=1e-6)


def _composite_rows() -> ResetStateBuffer:
    rig = RIGS["ur5e_2f85"]
    return ResetStateBuffer(
        {
            ROBOT: EntityStates(
                root_pos=torch.tensor([[0.0, -0.039, 0.0]]),
                root_quat=torch.tensor([IDENTITY]),
                joints_name=rig.arm_joints,
                joints_n_dofs=(1,) * 6,
                dofs_pos=torch.tensor([ARM_HOME]),
            ),
            GRIPPER: EntityStates(
                joints_name=rig.joint_observation[1][1], joints_n_dofs=(1,) * 6, dofs_pos=torch.zeros(1, 6)
            ),
            MOUNT: EntityStates(root_pos=torch.tensor([[0.0, -0.039, -0.0129]]), root_quat=torch.tensor([IDENTITY])),
            RECEPTIVE: EntityStates(root_pos=torch.tensor([[0.4, 0.1, 0.0022]]), root_quat=torch.tensor([IDENTITY])),
            INSERTIVE: EntityStates(root_pos=torch.tensor([[0.45, -0.15, 0.017]]), root_quat=torch.tensor([IDENTITY])),
        }
    )


@pytest.fixture(scope="module")
def composite_env(tmp_path_factory):
    path = tmp_path_factory.mktemp("omnireset") / "rows.pt"
    _composite_rows().save(path)
    config = TASK_REGISTRY.build("omnireset", robot="ur5e_2f85")
    config.env_options.num_envs = 2
    config.event_options.reset_states = ResetFromStateBuffer.configure(path=str(path))
    return build_env(config)


@pytest.mark.requires_genesis
def test_the_composite_arm_carries_the_sysid_armature_and_holds_still_at_120_hz(composite_env):
    env = composite_env
    env.reset()
    robot = env.entities[ROBOT]
    torch.testing.assert_close(robot.get_dofs_armature()[0].cpu(), torch.tensor(UR5E_SYSID_ARMATURE), atol=1e-5, rtol=0)
    arm = env.action_manager.get_term("arm")
    start = arm._ee_pose_in_base()[0].clone()
    for _ in range(10):  # 1 s at 10 Hz, UWLab's explicit controller at 120 Hz
        env.action_manager.compute(
            {"arm": torch.zeros(2, 6, device=env.device), "gripper": torch.ones(2, 1, device=env.device)}
        )
        for _ in range(DECIMATION):
            env.action_manager.apply_actions()
            env.scene.step()
    assert (arm._ee_pose_in_base()[0] - start).norm(dim=-1).max() < 1e-3


@pytest.mark.requires_genesis
def test_the_rewards_pay_for_assembly_and_for_reaching_the_object(composite_env):
    env = composite_env
    env.reset()
    pair, rig = PAIRS["peg"], RIGS["ur5e_2f85"]
    hole = env.entities[RECEPTIVE].get_pos(relative=True)
    # env 0 assembled; env 1 the peg 5 cm above its assembled pose.
    lift = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.05]], device=env.device)
    peg_pos = hole + torch.tensor([pair.assembled_pos], device=env.device) + lift
    env.entities[INSERTIVE].set_pos(peg_pos, relative=True)
    env.entities[INSERTIVE].set_quat(torch.tensor([IDENTITY] * 2, device=env.device), relative=True)
    env.metric_manager.compute()
    rewards = env.reward_manager
    success = rewards.get_term("success_reward").compute()
    dense = rewards.get_term("dense_success_reward").compute()
    assert success.tolist() == [1.0, 0.0]
    torch.testing.assert_close(dense[0].cpu(), torch.tensor(1.0), atol=1e-4, rtol=0)
    assert dense[1] < dense[0]
    # The grasp point (13.45 cm along the approach axis in UWLab's frame) is the peg's origin: distance reward 1.
    grip = env.entities[rig.grasp_entity]
    _, (link,) = grip.find_named_links_idx_local([rig.grasp_link])
    from eden.utils.geom import transform_by_quat

    point = grip.get_links_pos(ls_idx_local=[link])[:, 0] + transform_by_quat(
        torch.tensor([rig.grasp_offset], device=env.device).expand(2, 3), grip.get_links_quat(ls_idx_local=[link])[:, 0]
    )
    env.entities[INSERTIVE].set_pos(point, relative=True)
    torch.testing.assert_close(rewards.get_term("ee_asset_distance").compute().cpu(), torch.ones(2), atol=1e-4, rtol=0)


@pytest.mark.requires_genesis
def test_the_composite_gripper_pads_carry_the_robot_friction(composite_env):
    # Genesis takes a contact pair's max friction: pads left at their MJCF 1.2 would floor every grasp contact at
    # 1.2, under the insertive object's randomized 0.9-1.6.
    pads = [geom for geom in composite_env.entities[GRIPPER].geoms if geom.link.name.endswith("pad")]
    assert pads and all(float(geom.get_friction()) == pytest.approx(ROBOT_FRICTION) for geom in pads)


@pytest.mark.requires_genesis
def test_reward_handles_are_checked_at_build(composite_env):
    from eden.tasks.omnireset.custom_terms.rewards import JointVelL2Clamped, TerminationFired

    def built(term_cls, **params):
        term = term_cls(env=composite_env, options=term_cls.configure(**params))
        term.build()
        return term

    with pytest.raises(KeyError, match=r"No active term named 'abnormal_robt'.*abnormal_robot"):
        built(TerminationFired, termination_name="abnormal_robt")
    with pytest.raises(ValueError, match=rf"JointVelL2Clamped on entity '{ROBOT}' needs joints_name"):
        built(JointVelL2Clamped, entity_name=ROBOT)
    wrists = built(JointVelL2Clamped, entity_name=ROBOT, joints_name=["wrist_.*"])  # a pattern is a selection
    assert len(wrists._dofs_idx) == 3
