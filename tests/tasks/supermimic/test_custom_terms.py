"""The supermimic observation terms against the robot's own state."""

import pytest
import torch

pytestmark = [pytest.mark.requires_genesis]


def test_root_joint_state_is_the_base_link_pose_twist_and_joints(env):
    env.reset()
    robot = env.entities["robot"]
    _, root = robot.find_named_links_idx_local(["pelvis"])
    state = env.observation_manager.compute()["full_state"]
    origins = getattr(env, "env_origins", None)
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
