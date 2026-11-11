"""The OmniReset terms on a built workspace: the 43-D observation, the controller, the gripper and the grader.

The env is the task's workspace and the released expert's observation and action terms, with reset rows written
here, so nothing from UWLab is downloaded.
"""

from __future__ import annotations

import pytest
import torch

from eden.extensions.policies.omnireset.constants import POLICY_OBSERVATION_LAYOUT
from eden.extensions.policies.omnireset.policy import OmniResetPolicy
from eden.managers.terms.events.state_buffer import ResetFromStateBuffer
from eden.tasks.omnireset.config import scene_entity_options
from eden.tasks.omnireset.constants import (
    INSERTIVE,
    ISAAC_JOINT_ORDER,
    MOUNT,
    OSC_TORQUE_LIMIT,
    PAIRS,
    RECEPTIVE,
    ROBOT,
    SIM_DT,
    SIM_SUBSTEPS,
)
from eden.tasks.omnireset.custom_terms import OmniResetStates, assembly_errors
from eden.utils.buffers.reset_states import EntityStates, ResetStateBuffer
from tests.support.envs import build_env

IDENTITY = [1.0, 0.0, 0.0, 0.0]
# The gripper pointing down over the table, open, with the peg standing in front of it.
GRIP_DOWN = [0.0, -1.5708, 1.5708, -1.5708, -1.5708, 0.0] + [0.0] * 6
ROWS = ResetStateBuffer(
    {
        ROBOT: EntityStates(
            root_pos=torch.tensor([[0.0, -0.039, 0.0]]),
            root_quat=torch.tensor([IDENTITY]),
            joints_name=ISAAC_JOINT_ORDER,
            joints_n_dofs=(1,) * 12,
            dofs_pos=torch.tensor([GRIP_DOWN]),
        ),
        MOUNT: EntityStates(root_pos=torch.tensor([[0.0, -0.039, -0.0129]]), root_quat=torch.tensor([IDENTITY])),
        RECEPTIVE: EntityStates(root_pos=torch.tensor([[0.4, 0.1, 0.0022]]), root_quat=torch.tensor([IDENTITY])),
        INSERTIVE: EntityStates(root_pos=torch.tensor([[0.45, -0.15, 0.017]]), root_quat=torch.tensor([IDENTITY])),
    }
)


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    from eden.options import EnvOptions, SceneOptions
    from eden.tasks.omnireset.config import OmniResetConfig
    from eden.utils.configs import ActionManagerOptions, EdenConfig, EventManagerOptions, ObservationManagerOptions

    path = tmp_path_factory.mktemp("omnireset") / "rows.pt"
    ROWS.save(path)
    return build_env(
        EdenConfig(
            env_options=EnvOptions(num_envs=1, sim_dt=SIM_DT, sim_substeps=SIM_SUBSTEPS),
            scene_options=SceneOptions(**scene_entity_options(PAIRS["peg"])),
            observation_options=ObservationManagerOptions(**OmniResetPolicy.observation_options()),
            action_options=ActionManagerOptions(**OmniResetPolicy.action_options()),
            event_options=EventManagerOptions(restore=ResetFromStateBuffer.configure(path=str(path))),
            metric_options=OmniResetConfig().metric_options,
        )
    )


def _step(env, arm, gripper, physics_steps=96):
    env.action_manager.compute({"arm": torch.tensor([arm]), "gripper": torch.tensor([[gripper]])})
    for _ in range(physics_steps):
        env.action_manager.apply_actions()
        env.scene.step()


@pytest.mark.requires_genesis
def test_the_policy_group_is_the_checkpoints_43_values_in_layout_order(env):
    env.reset()
    obs = env.observation_manager.compute()["policy"]
    assert obs.shape == (1, sum(width for _, width in POLICY_OBSERVATION_LAYOUT))
    joints = obs[0, 7:19].cpu()
    torch.testing.assert_close(joints, torch.tensor(GRIP_DOWN), atol=1e-5, rtol=0)
    # insertive_asset_in_receptive_asset_frame: the peg 5 cm along +x and 25 cm along -y of the hole, unrotated.
    torch.testing.assert_close(obs[0, 37:].cpu(), torch.tensor([0.05, -0.25, 0.0148, 0, 0, 0]), atol=1e-4, rtol=0)


@pytest.mark.requires_genesis
def test_the_controller_holds_still_at_rest_and_moves_the_wrist_along_the_commanded_axis(env):
    arm = env.action_manager.get_term("arm")
    env.reset()
    start = arm._ee_pose_in_base()[0].clone()
    _step(env, [0.0] * 6, 1.0)
    held = arm._ee_pose_in_base()[0]
    assert (held - start).norm() < 1e-3, f"zero action drifted the wrist {(held - start).norm() * 1e3:.2f} mm"
    _step(env, [3.0, 0.0, 0.0, 0.0, 0.0, 0.0], 1.0, physics_steps=480)  # 0.5 s toward +x (6 cm target)
    moved = arm._ee_pose_in_base()[0] - held
    assert moved[0, 0] > 0.02 and moved[0, 1:].abs().max() < 0.25 * moved[0, 0], f"wrist moved {moved}"
    assert (arm.processed_action.abs() <= torch.tensor(OSC_TORQUE_LIMIT, device=env.device) + 1e-4).all()


@pytest.mark.requires_genesis
def test_a_negative_gripper_action_closes_the_gripper(env):
    env.reset()
    robot = env.entities[ROBOT]
    _, finger = robot.find_named_dofs_idx_local(["finger_joint"])
    _step(env, [0.0] * 6, -1.0, physics_steps=960)
    assert robot.get_dofs_pos(dofs_idx_local=finger)[0, 0] > 0.7
    _step(env, [0.0] * 6, 1.0, physics_steps=960)
    assert robot.get_dofs_pos(dofs_idx_local=finger)[0, 0] < 0.05


def _place_peg(env, offset=(0.0, 0.0, 0.0)):
    """The peg at its assembled pose in the hole, shifted by ``offset`` (m)."""
    hole = env.entities[RECEPTIVE].get_pos(relative=True)
    shift = torch.tensor([PAIRS["peg"].assembled_pos], device=env.device) + torch.tensor([offset], device=env.device)
    env.entities[INSERTIVE].set_pos(hole + shift, relative=True)
    env.entities[INSERTIVE].set_quat(torch.tensor([IDENTITY], device=env.device), relative=True)


@pytest.mark.requires_genesis
def test_the_grader_reads_zero_error_at_the_assembled_pose(env):
    env.reset()
    _place_peg(env)
    position_error, tilt_error = assembly_errors(env, PAIRS["peg"], INSERTIVE, RECEPTIVE)
    assert position_error.item() < 1e-5 and tilt_error.item() < 1e-5


@pytest.mark.requires_genesis
@pytest.mark.parametrize(
    "offset, success", [((0.0, 0.0, 0.0), 1.0), ((0.004, 0.0, 0.0), 0.0)], ids=["seated", "4mm-off"]
)
def test_the_task_grades_the_pair_on_the_last_state_of_the_episode(env, offset, success):
    # UWLab grades at reset: the metrics are judged on the value computed in the episode's last step.
    env.reset()
    _place_peg(env, offset)
    env.metric_manager.compute()
    extras = env.metric_manager.reset()
    assert extras["Metrics/Success"].item() == success


@pytest.mark.requires_genesis
def test_omnireset_states_refuses_a_path_before_it_converts_anything():
    from eden.options import EnvOptions, SceneOptions
    from eden.utils.configs import EdenConfig, EventManagerOptions

    config = EdenConfig(
        env_options=EnvOptions(num_envs=1),
        scene_options=SceneOptions(),
        event_options=EventManagerOptions(reset_states=OmniResetStates.configure(path="mine.pt")),
    )
    with pytest.raises(ValueError, match=r"got path='mine.pt'.*ResetFromStateBuffer.configure\(path='mine.pt'\)"):
        build_env(config)


def test_an_unknown_rig_is_refused_naming_the_rigs():
    term = OmniResetStates(env=None, options=OmniResetStates.configure(pair="peg", rig="ur5e"))
    with pytest.raises(ValueError, match=r"unknown rig 'ur5e'; rigs: \['uwlab', 'ur5e_2f85'\]"):
        term.build()


def test_the_composite_needs_its_rows_retargeted_first(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    term = OmniResetStates(env=None, options=OmniResetStates.configure(pair="peg", rig="ur5e_2f85"))
    with pytest.raises(FileNotFoundError, match=r"python -m eden.tasks.omnireset.retarget --pair peg"):
        term.build()
