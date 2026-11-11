"""The stateful observation terms behind ``ProtoMotionV3Policy``'s env spec, on the env that spec builds.

The policy once kept its heading alignment and its previous joint targets itself; they now live on
observation terms, and these tests pin when each is latched and cleared.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from genesis.utils import geom as gu

from eden.extensions.policies.protomotionv3.policy import ProtoMotionV3Policy

FRAMES = 30


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    """The env a task-less ``eden inference --policy protomotionv3`` builds, on a clip that turns in place."""
    from eden.utils.cli.tasks import _build_extension_env, _extension_command_options

    yaw = np.linspace(0.0, 0.3, FRAMES)
    quat = np.repeat(np.stack([gu.xyz_to_quat(np.array([0.0, 0.0, y]), rpy=True) for y in yaw])[:, None], 2, axis=1)
    pos = np.zeros((FRAMES, 2, 3), np.float32)
    pos[:, 0, 2], pos[:, 1, 2] = 0.79, 0.99
    motion_file = tmp_path_factory.mktemp("protomotionv3") / "motion.npz"
    np.savez(
        motion_file,
        joint_pos=np.zeros((FRAMES, 29), np.float32),
        joint_vel=np.zeros((FRAMES, 29), np.float32),
        body_pos_w=pos,
        body_quat_w=quat.astype(np.float32),
        body_lin_vel_w=np.zeros((FRAMES, 2, 3), np.float32),
        body_ang_vel_w=np.zeros((FRAMES, 2, 3), np.float32),
        body_names=np.array(["pelvis", "torso_link"]),
    )
    command = _extension_command_options(ProtoMotionV3Policy, "robot", str(motion_file))
    return _build_extension_env(ProtoMotionV3Policy, "G1Dofs29", "robot", False, command)


def _term(env, name: str):
    return {(group, key): term for group, key, term in env.observation_manager.iter_terms()}[("policy", name)]


@pytest.mark.requires_genesis
def test_previous_targets_read_zero_after_a_reset_then_the_applied_targets(env):
    obs, _ = env.reset()
    prev_targets = slice(64 + 256, 64 + 256 + 29)
    assert not obs["policy"][:, prev_targets].any()

    obs, *_ = env.step({"dofs_pos_controller": torch.full((1, 29), 0.1)})
    applied = env.action_manager.get_term("dofs_pos_controller").processed_action
    torch.testing.assert_close(obs["policy"][:, prev_targets], applied, rtol=0, atol=0)


@pytest.mark.requires_genesis
def test_relatching_aligns_the_reference_heading_with_a_turned_robot(env):
    """Turn the robot after the reset: the relative target rotations change until the heading is re-latched."""
    term = _term(env, "target_poses")
    env.reset()
    aligned = term.compute()[:, :6].clone()  # the first future frame's relative anchor rotation

    robot = env.entities["robot"]
    turn = torch.tensor(gu.xyz_to_quat(np.array([0.0, 0.0, math.pi / 3]), rpy=True), dtype=torch.float32)
    robot.set_quat(gu.transform_quat_by_quat(robot.get_quat()[0].cpu(), turn).unsqueeze(0))
    assert not torch.allclose(term.compute()[:, :6], aligned, atol=1e-3)

    term.post_reset(slice(None))  # what an env reset, or a deployment's observation restart, does
    torch.testing.assert_close(term.compute()[:, :6], aligned, atol=1e-5, rtol=0)
    env.reset()
