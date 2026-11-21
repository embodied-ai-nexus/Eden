"""The supermimic observation terms against the robot's own state."""

import pytest
import torch

pytestmark = [pytest.mark.requires_genesis]


def test_root_joint_state_is_the_base_link_pose_twist_and_joints(env):
    env.reset()
    robot = env.entities["robot"]
    _, root = robot.find_named_links_idx_local(["pelvis"])
    state = env.observation_manager.compute()["full_state"]
    origins = env.motion_manager.env_origins
    root_pos = robot.get_links_pos(ls_idx_local=root)[:, 0] - (0 if origins is None else origins)
    expected = torch.cat(
        [
            root_pos,
            robot.get_links_quat(ls_idx_local=root)[:, 0],
            robot.get_dofs_pos(),
            robot.get_links_vel(ls_idx_local=root)[:, 0],
            robot.get_links_ang(ls_idx_local=root)[:, 0],
            robot.get_dofs_vel(),
        ],
        dim=-1,
    )
    torch.testing.assert_close(state, expected)


def test_the_biased_joint_reading_adds_the_encoder_bias(env):
    """The encoder-bias event draws the controller offset ``-b``; the reading is ``q - q0 + b``."""
    env.reset()
    robot = env.entities["robot"]
    controller = env.action_manager.get_term("joint_pos")
    bias = robot.default_dofs_pos - controller.effective_offset
    assert bias.abs().max() > 0, "the startup event drew no encoder bias"
    assert bias.abs().max() <= 0.01 + 1e-6
    reading = env.observation_manager.compute()["policy"][:, 6 : 6 + len(robot.dofs_name)]
    torch.testing.assert_close(reading, robot.get_dofs_pos() - robot.default_dofs_pos + bias)


def test_clip_end_names_the_frame_count_it_needs(lafan1_library):
    from eden.tasks import TASK_REGISTRY
    from tests.support.envs import build_env

    cfg = TASK_REGISTRY.build("supermimic", raw_data_path=str(lafan1_library))
    cfg.env_options.num_envs = 1
    cfg.motion_options.num_frames = 1
    with pytest.raises(ValueError, match="num_frames=1. Set motion_options.num_frames >= 2"):
        build_env(cfg, reset=True)


def test_the_biased_joint_reading_refuses_a_controller_with_a_per_step_reference(lafan1_library):
    from eden.constants import ReferenceSource
    from eden.tasks import TASK_REGISTRY
    from tests.support.envs import build_env

    cfg = TASK_REGISTRY.build("supermimic", raw_data_path=str(lafan1_library))
    cfg.env_options.num_envs = 1
    cfg.action_options.joint_pos.reference_source = ReferenceSource.ZERO
    with pytest.raises(
        ValueError, match=r"action term 'joint_pos' has reference_source=.*ZERO.*ReferenceSource.DEFAULT"
    ):
        build_env(cfg, reset=True)
