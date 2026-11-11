"""``Twist2Policy``'s env spec reproduces the frozen actor's input and action transform bit for bit.

The actor once built its 1524-dim input by hand from raw robot and motion state inside ``step()``.
Its spec now declares that input as an observation group with an 11-frame history, and these tests
pin the env those options build against the hand-built formula, kept here as the oracle.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from genesis.utils import geom as gu
from genesis.utils.geom import inv_transform_by_quat

from eden.extensions.policies.twist2.constants import (
    ACTOR_CURRENT_DIM,
    ACTOR_HISTORY_LENGTH,
    ACTOR_OBS_DIM,
    BASE_ANG_VEL_SCALE,
    DEFAULT_POS,
    JOINT_SCALES,
    JOINT_VEL_SCALES,
    NUM_JOINTS,
    POLICY_JOINT_NAMES,
)
from eden.extensions.policies.twist2.policy import Twist2Policy
from eden.utils.geom import euler_xyz_from_quat

FRAMES = 60


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    """The env a task-less ``eden inference --policy twist2`` builds, tracking a clip that moves, turns and tilts."""
    from eden.utils.cli.tasks import _build_extension_env, _extension_command_options

    t = np.arange(FRAMES, dtype=np.float32) / 50.0
    yaw, roll = 0.4 * t, 0.05 * np.sin(3.0 * t)
    quat = np.stack([gu.xyz_to_quat(np.array([r, 0.02, y]), rpy=True) for r, y in zip(roll, yaw)])[:, None]
    pos = np.stack([0.3 * t, 0.1 * t, 0.79 + 0.01 * np.sin(2.0 * t)], -1)[:, None]
    motion_file = tmp_path_factory.mktemp("twist2") / "motion.npz"
    np.savez(
        motion_file,
        joint_pos=(np.asarray(DEFAULT_POS) + 0.1 * np.sin(t[:, None] + np.arange(NUM_JOINTS))).astype(np.float32),
        joint_vel=np.zeros((FRAMES, NUM_JOINTS), np.float32),
        body_pos_w=pos.astype(np.float32),
        body_quat_w=quat.astype(np.float32),
        body_lin_vel_w=np.broadcast_to([[[0.3, 0.1, 0.0]]], (FRAMES, 1, 3)).astype(np.float32),
        body_ang_vel_w=np.broadcast_to([[[0.0, 0.0, 0.4]]], (FRAMES, 1, 3)).astype(np.float32),
        body_names=np.array(["pelvis"]),
    )
    command = _extension_command_options(Twist2Policy, "robot", str(motion_file))
    return _build_extension_env(Twist2Policy, "G1Dofs29", "robot", False, command)


def _hand_built_frame(env, last_action: torch.Tensor) -> torch.Tensor:
    """One 127-dim frame, as ``Twist2Policy._build_mimic`` / ``_build_proprio`` assembled it from raw state."""
    motion = env.command_manager.get_term("motion")
    root_quat = motion.base_quat_w
    roll, pitch, _ = euler_xyz_from_quat(root_quat)
    mimic = torch.cat(
        [
            inv_transform_by_quat(motion.base_lin_vel_w, root_quat)[:, :2],
            motion.base_pos_w[:, 2:3],
            torch.stack([roll, pitch], dim=-1),
            inv_transform_by_quat(motion.base_ang_vel_w, root_quat)[:, 2:3],
            motion.dofs_pos,
        ],
        dim=-1,
    )
    robot = env.entities["robot"]
    _, dofs_idx = robot.find_named_dofs_idx_local(list(POLICY_JOINT_NAMES), preserve_order=True)
    dofs_idx = torch.as_tensor(dofs_idx, dtype=torch.long, device=env.device)
    roll, pitch, _ = euler_xyz_from_quat(robot.get_quat())
    proprio = torch.cat(
        [
            robot.get_ang(frame="body") * BASE_ANG_VEL_SCALE,
            torch.stack([roll, pitch], dim=-1),
            robot.get_dofs_pos(dofs_idx_local=dofs_idx) - torch.tensor(DEFAULT_POS, device=env.device),
            robot.get_dofs_vel(dofs_idx_local=dofs_idx) * torch.tensor(JOINT_VEL_SCALES, device=env.device),
            last_action,
        ],
        dim=-1,
    )
    return torch.cat([mimic, proprio], dim=-1)


def _history(frames: list[torch.Tensor]) -> torch.Tensor:
    """The last 11 frames, oldest first and flattened, backfilled with the first frame after a reset."""
    window = ([frames[0]] * ACTOR_HISTORY_LENGTH + frames)[-ACTOR_HISTORY_LENGTH:]
    return torch.cat(window, dim=-1)


@pytest.mark.requires_genesis
def test_the_env_publishes_the_hand_built_history_and_action(env):
    generator = torch.Generator(device=env.device).manual_seed(0)
    term = env.action_manager.get_term("dofs_pos_controller")
    default = torch.tensor(DEFAULT_POS, device=env.device)
    scale = torch.tensor(JOINT_SCALES, device=env.device)

    obs, _ = env.reset()
    last_action = torch.zeros(env.num_envs, NUM_JOINTS, device=env.device)
    frames = [_hand_built_frame(env, last_action)]
    torch.testing.assert_close(obs["policy"], _history(frames), rtol=0, atol=0)

    for _ in range(ACTOR_HISTORY_LENGTH + 2):  # past the window, so the backfill has rolled out
        last_action = 0.3 * torch.randn(env.num_envs, NUM_JOINTS, device=env.device, generator=generator)
        obs, *_ = env.step({"dofs_pos_controller": last_action})
        torch.testing.assert_close(term.processed_action, default + last_action * scale, rtol=0, atol=0)
        frames.append(_hand_built_frame(env, last_action))
        torch.testing.assert_close(obs["policy"], _history(frames), rtol=0, atol=0)


@pytest.mark.requires_genesis
def test_step_feeds_the_actor_the_current_frame_then_the_history(env, tmp_path, write_onnx, onnx_runtime, monkeypatch):
    """A pass-through actor returns the first 29 columns of its input, which belong to the current frame."""
    monkeypatch.setenv("ONNX_CPU", "1")
    starts, ends, axes = np.array([0], np.int64), np.array([NUM_JOINTS], np.int64), np.array([1], np.int64)
    actor = write_onnx(
        tmp_path / "actor.onnx",
        [("Slice", ["obs", "starts", "ends", "axes"], ["actions"], {})],
        [("obs", [1, ACTOR_OBS_DIM])],
        [("actions", [1, NUM_JOINTS])],
        {"starts": starts, "ends": ends, "axes": axes},
    )
    policy_cls = type("LocalTwist2Policy", (Twist2Policy,), {})
    policy_cls.use_local_checkpoint(actor)
    policy = policy_cls(env=env, options=policy_cls.configure())

    obs, _ = env.reset()
    obs, *_ = env.step(policy.step(obs))
    action = policy.step(obs)["dofs_pos_controller"]
    torch.testing.assert_close(action, obs["policy"][:, -ACTOR_CURRENT_DIM:][:, :NUM_JOINTS])
