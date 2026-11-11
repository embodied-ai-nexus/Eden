"""``HuskyPolicy``'s env spec reproduces the frozen actor's input and action transform bit for bit.

The actor once built its 395-dim input by hand from raw robot state inside ``step()``. Its spec now
declares that input as the training task's observation group, held command and action term, and
these tests pin the env those options build against the hand-built formula, kept here as the oracle.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from genesis.utils.geom import inv_transform_by_quat

import eden as en
from eden.extensions.policies.husky.constants import (
    ACTION_SCALE,
    ACTOR_OBS_DIM,
    ANG_VEL_SCALE,
    COMMAND_SCALE,
    CYCLE_TIME_S,
    DEFAULT_DOFS_POS,
    DOFS_VEL_SCALE,
    HEADING_SCALE,
    HISTORY_LENGTH,
    NUM_JOINTS,
    OBS_BLOCK_DIMS,
    POLICY_JOINT_NAMES,
)
from eden.extensions.policies.husky.policy import HuskyPolicy
from eden.options import ActionManagerOptions, CommandManagerOptions, EnvOptions, ObservationManagerOptions
from eden.options.managers.observations import ObservationTermOptions
from eden.tasks import TASK_REGISTRY
from eden.utils.configs import EdenRLConfig
from eden.utils.geom import heading_from_quat
from tests.support.envs import build_env

COMMAND = (1.0, 0.3)


@pytest.fixture(scope="module")
def env():
    """The env a task-less ``eden inference --policy husky`` builds, with two envs."""
    env = build_env(
        EdenRLConfig(
            env_options=EnvOptions(num_envs=2, episode_length_s=1e9, **HuskyPolicy.env_options()),
            scene_options=en.scenes.Empty(robot=HuskyPolicy.robot_options(), **HuskyPolicy.scene_entity_options()),
            action_options=ActionManagerOptions(**HuskyPolicy.action_options()),
            command_options=CommandManagerOptions(**HuskyPolicy.command_options()),
            observation_options=ObservationManagerOptions(**HuskyPolicy.observation_options()),
        )
    )
    env.command_manager.get_term("skate").set_command(torch.tensor(COMMAND))
    return env


def _hand_built_frame(env, last_action: torch.Tensor, steps: int) -> torch.Tensor:
    """One 79-dim frame, as ``HuskyPolicy._build_frame`` assembled it from raw state (moving command only)."""
    robot = env.entities["robot"]
    _, dofs_idx = robot.find_named_dofs_idx_local(list(POLICY_JOINT_NAMES), preserve_order=True)
    dofs_idx = torch.as_tensor(dofs_idx, dtype=torch.long, device=env.device)
    commands = torch.tensor(COMMAND, device=env.device).expand(env.num_envs, 2)
    control_dt = env.env_options.sim_dt * env.env_options.decimation  # 0.02 s, as the source's 50 Hz
    phase = (torch.full((env.num_envs,), steps, device=env.device).float() * (control_dt / CYCLE_TIME_S)) % 1.0
    base_quat = robot.get_quat(relative=True)
    ang_vel_b = robot.get_ang(frame="body")
    gravity = torch.tensor([0.0, 0.0, -1.0], device=env.device).expand_as(ang_vel_b)
    return torch.cat(
        [
            commands * torch.tensor(COMMAND_SCALE, device=env.device),
            (heading_from_quat(base_quat) * HEADING_SCALE).unsqueeze(-1),
            ang_vel_b * ANG_VEL_SCALE,
            inv_transform_by_quat(gravity, base_quat),
            robot.get_dofs_pos(dofs_idx_local=dofs_idx) - torch.tensor(DEFAULT_DOFS_POS, device=env.device),
            robot.get_dofs_vel(dofs_idx_local=dofs_idx) * DOFS_VEL_SCALE,
            last_action,
            phase.unsqueeze(-1),
        ],
        dim=-1,
    )


def _per_term_blocks(history: list[torch.Tensor]) -> torch.Tensor:
    """The actor's term-major layout: each term's last 5 frames, oldest first, backfilled at reset."""
    window = torch.stack(([history[0]] * HISTORY_LENGTH + history)[-HISTORY_LENGTH:], dim=1)
    bounds = torch.tensor((0, *OBS_BLOCK_DIMS)).cumsum(0).tolist()
    return torch.cat([window[:, :, lo:hi].reshape(window.shape[0], -1) for lo, hi in zip(bounds, bounds[1:])], -1)


@pytest.mark.requires_genesis
def test_the_env_publishes_the_hand_built_actor_input_and_action(env):
    generator = torch.Generator(device=env.device).manual_seed(0)
    term = env.action_manager.get_term("dofs_pos_controller")
    default = torch.tensor(DEFAULT_DOFS_POS, device=env.device)
    scale = torch.tensor(ACTION_SCALE, device=env.device)

    obs, _ = env.reset()
    last_action = torch.zeros(env.num_envs, NUM_JOINTS, device=env.device)
    history = [_hand_built_frame(env, last_action, steps=0)]
    torch.testing.assert_close(obs["policy"], _per_term_blocks(history), rtol=0, atol=0)

    for steps in range(1, HISTORY_LENGTH + 3):  # past the window, so the backfill has rolled out
        last_action = 0.3 * torch.randn(env.num_envs, NUM_JOINTS, device=env.device, generator=generator)
        obs, *_ = env.step({"dofs_pos_controller": last_action})
        torch.testing.assert_close(term.processed_action, default + last_action * scale, rtol=0, atol=0)
        history.append(_hand_built_frame(env, last_action, steps))
        torch.testing.assert_close(obs["policy"], _per_term_blocks(history), rtol=0, atol=0)
    assert obs["policy"].shape == (env.num_envs, ACTOR_OBS_DIM)


def test_observation_spec_is_the_training_actor_group_without_its_noise():
    """One layout for the released checkpoint and for policies trained on ``--task husky``."""
    task_config = TASK_REGISTRY.build("husky")
    task = task_config.observation_options.policy
    spec = HuskyPolicy.observation_options()["policy"]

    def terms(group) -> dict:
        return {
            name: term.model_dump(exclude={"noise"})
            for name, term in group.__pydantic_extra__.items()
            if isinstance(term, ObservationTermOptions)
        }

    assert terms(spec) == terms(task)
    assert all(getattr(spec, name).noise is None for name in terms(spec))
    assert HuskyPolicy.action_options()["dofs_pos_controller"] == task_config.action_options.dofs_pos_controller


@pytest.fixture(scope="module")
def task_env():
    """The training task's env, which ``eden inference --task husky --policy husky`` runs the policy on."""
    config = TASK_REGISTRY.build("husky")
    config.env_options.num_envs = 1
    return build_env(config, reset=True)


@pytest.mark.requires_genesis
def test_on_the_training_task_the_task_command_drives_the_policy(task_env, tmp_path, write_onnx, onnx_runtime):
    """The task's actor group is the policy's input, and its resampled ``SkateCommand`` holds no command to set."""
    starts, ends, axes = np.array([0], np.int64), np.array([NUM_JOINTS], np.int64), np.array([1], np.int64)
    actor = write_onnx(
        tmp_path / "actor.onnx",
        [("Slice", ["obs", "starts", "ends", "axes"], ["actions"], {})],
        [("obs", [1, ACTOR_OBS_DIM])],
        [("actions", [1, NUM_JOINTS])],
        {"starts": starts, "ends": ends, "axes": axes},
    )
    policy_cls = type("LocalHuskyPolicy", (HuskyPolicy,), {})
    policy_cls.use_local_checkpoint(actor)
    policy = policy_cls(env=task_env, options=policy_cls.configure())

    assert policy._commands.shape == (1, 2)  # what the CLI sizes --base_command from
    with pytest.raises(TypeError, match="SkateCommand, which samples its own command"):
        policy.set_command(torch.tensor([1.0, 0.0]))
    obs, _ = task_env.reset()
    assert policy.step(obs)["dofs_pos_controller"].shape == (1, NUM_JOINTS)
