"""Tests for ``eden.managers.terms.rewards.tracking``: motion-tracking rewards.

The rewards score a built G1 and Panda against a reference written into the env's motion buffer
relative to the entities' real state, so every expected value is a known distance.
"""

from __future__ import annotations

import math

import pytest
import torch

from eden.managers.reward_manager import RewardManager
from eden.managers.terms.rewards.tracking import (
    MotionRelativeBodyOrientationErrorExp,
    MotionRelativeBodyPositionErrorExp,
    exp_kernel,
    motion_dofs_pos_tracking_exp,
    motion_link_position_tracking_exp,
    validity_weighted_frame_mean,
)
from eden.options.managers.rewards import RewardManagerOptions
from tests.support.motion import build_tracking_env, set_reference, write_tracking_library

# Four of the G1's joints, and the Panda's first four.
DOFS = {
    "robot": ["left_hip_pitch_joint", "left_hip_roll_joint", "left_knee_joint", "waist_yaw_joint"],
    "arm": ["joint1", "joint2", "joint3", "joint4"],
}
RAMP = torch.tensor([0.0, 0.1, 0.2, 0.3])


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    return build_tracking_env(write_tracking_library(tmp_path_factory.mktemp("tracking")))


def _links_pos(env, names):
    robot = env.entities["robot"]
    return robot.get_links_pos(ls_idx_local=robot.find_named_links_idx_local(names, preserve_order=True)[1])


def _hold_dofs(env, entity, values):
    """Set ``entity``'s ``DOFS[entity]`` joints to ``values`` in every env."""
    ent = env.entities[entity]
    ent.set_dofs_pos(
        values.expand(env.num_envs, -1).to(env.device),
        dofs_idx_local=ent.find_named_dofs_idx_local(DOFS[entity], preserve_order=True)[1],
    )


def _motion_dofs(env, entity):
    names = env.motion_manager.get_target_dofs_name(entity)
    return [names.index(n) for n in DOFS[entity]]


def test_link_position_reward_reads_only_the_current_frame_by_default(env):
    links = ["pelvis", "torso_link"]
    target = _links_pos(env, links)[:, None].repeat(1, 9, 1, 1)
    target[:, 1:] += 10.0  # far away on every later frame
    target[:, 0, 0, 0] += 1.0
    target[:, 0, 1, 1] += 1.0
    full = env.motion_manager.data["robot"].links_pos_w.clone()
    full[:, :, :2] = target  # the reference lists pelvis and torso first
    set_reference(env, "robot", links_pos_w=full, validity=torch.ones(4, 9, dtype=torch.bool))

    reward = motion_link_position_tracking_exp(env, entity_name="robot", links_name=links, std=1.0)

    torch.testing.assert_close(reward.cpu(), torch.full((4,), math.exp(-1.0)))


def test_frames_past_the_clip_end_are_left_out_of_a_multi_frame_reward(env):
    links = ["pelvis"]
    full = env.motion_manager.data["robot"].links_pos_w.clone()
    full[:, :, 0] = _links_pos(env, links)[:, None, 0]
    full[:, 1:3, 0, 0] += 10.0  # frames 1 and 2 are terrible, and invalid
    set_reference(env, "robot", links_pos_w=full, validity=torch.tensor([True, False, False, True] + [True] * 5))

    reward = motion_link_position_tracking_exp(
        env, entity_name="robot", links_name=links, std=1.0, frame_offsets=(0, 1, 2, 3)
    )

    torch.testing.assert_close(reward.cpu(), torch.ones(4))


def test_frame_offsets_past_the_buffer_raise(env):
    with pytest.raises(ValueError, match="num_frames"):
        motion_link_position_tracking_exp(
            env, entity_name="robot", links_name=["pelvis"], std=1.0, frame_offsets=(0, 9)
        )


def test_body_rewards_in_one_reward_pass_score_what_each_scores_alone(env, install_manager):
    """In a pass, the position and orientation rewards over one link set share one anchor-relative warp.

    Each term's ``compute()`` on its own, outside a pass, warps afresh: that is what the pass must match.
    """
    set_reference(env, "robot", validity=torch.ones(4, 9, dtype=torch.bool))
    link_sets = {"upper": ["pelvis", "torso_link"], "feet": ["left_ankle_roll_link", "right_ankle_roll_link"]}
    terms = {
        f"{kind}_{links}": cls.configure(weight=1.0, anchor_link_name="torso_link", links_name=names, std=0.5)
        for kind, cls in (("pos", MotionRelativeBodyPositionErrorExp), ("ori", MotionRelativeBodyOrientationErrorExp))
        for links, names in link_sets.items()
    }
    manager = install_manager(env, "reward_manager", RewardManager(env, RewardManagerOptions(**terms)))

    manager.compute(dt=1.0)  # a new manager's raw episode sums then hold exactly this pass's term values

    for name in terms:
        torch.testing.assert_close(manager.get_episode_sum(name), manager.get_term(name).compute(), msg=name)
    assert not torch.allclose(manager.get_episode_sum("pos_upper"), manager.get_episode_sum("pos_feet"))


@pytest.mark.parametrize("entity", ["robot", "arm"], ids=["floating-base", "fixed-base"])
def test_dofs_reward_reads_each_joint_by_name(env, entity):
    """``get_robot_dofs_idx`` gives Genesis-local indices; gathering the full-width qpos with them is out
    of bounds on a floating base and a silent permutation on a fixed one.
    """
    _hold_dofs(env, entity, RAMP)
    dofs = env.motion_manager.data[entity].dofs_pos.clone()
    dofs[:, :, _motion_dofs(env, entity)] = 0.5
    dofs[:, 1:] += 10.0  # only frame 0 is scored by default
    set_reference(env, entity, dofs_pos=dofs, validity=torch.ones(4, 9, dtype=torch.bool))

    total = motion_dofs_pos_tracking_exp(env, entity_name=entity, std=1.0, dofs_name=DOFS[entity])
    # A subset in another order must not be permuted: per-DOF credit for (d3, d0).
    per_dof = motion_dofs_pos_tracking_exp(
        env, entity_name=entity, std=1.0, dofs_name=[DOFS[entity][3], DOFS[entity][0]], reduction="per_dof_mean"
    )

    # sum over (q - 0.5)^2 for q in 0, 0.1, 0.2, 0.3 = 0.54
    torch.testing.assert_close(total.cpu(), torch.full((4,), math.exp(-0.54)))
    torch.testing.assert_close(per_dof.cpu(), torch.full((4,), 0.5 * (math.exp(-0.04) + math.exp(-0.25))))


def test_invalid_frames_are_left_out_of_a_multi_frame_dofs_reward(env):
    _hold_dofs(env, "robot", RAMP)
    dofs = env.motion_manager.data["robot"].dofs_pos.clone()
    dofs[:, 0, _motion_dofs(env, "robot")] = RAMP.to(dofs.device)
    dofs[:, 1:3] += 10.0
    set_reference(env, "robot", dofs_pos=dofs, validity=torch.tensor([True, False, False] + [True] * 6))

    reward = motion_dofs_pos_tracking_exp(
        env, entity_name="robot", std=1.0, dofs_name=DOFS["robot"], frame_offsets=(0, 1, 2)
    )

    torch.testing.assert_close(reward.cpu(), torch.ones(4))


def test_dofs_reward_on_an_entity_without_dofs_says_to_use_a_link_reward(env):
    with pytest.raises(ValueError, match="'object' has no DOFs in the motion reference"):
        motion_dofs_pos_tracking_exp(env, entity_name="object", std=0.1)


# ---------------------------------------------------------------------------
# Reduction primitives
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mode, expected",
    [
        ("mean_of_exp", (math.exp(-1.0) + math.exp(-3.0)) / 2),
        ("exp_of_mean", math.exp(-2.0)),
        ("exp_of_sum", math.exp(-4.0)),
    ],
)
def test_exp_kernel_modes(mode, expected):
    torch.testing.assert_close(exp_kernel(torch.tensor([[1.0, 3.0]]), 1.0, mode=mode), torch.tensor([expected]))
    torch.testing.assert_close(exp_kernel(torch.zeros(2, 3, 4), 0.5, mode=mode), torch.ones(2, 3))


def test_exp_kernel_rejects_an_unknown_mode():
    with pytest.raises(ValueError, match="exp_kernel mode"):
        exp_kernel(torch.zeros(2, 3), std=0.5, mode="sum_of_exp")


@pytest.mark.parametrize(
    "per_frame, valid, expected",
    [
        ([[1.0, 0.0], [0.5, 0.5]], [[1.0, 0.0], [1.0, 1.0]], [1.0, 0.5]),
        ([[0.7, 0.3]], [[0.0, 0.0]], [0.0]),  # all invalid: 0, not NaN
        ([[0.2, 0.8], [1.0, 0.0]], None, [0.5, 0.5]),
    ],
    ids=["masked", "all-invalid", "no-validity"],
)
def test_validity_weighted_frame_mean(per_frame, valid, expected):
    valid = None if valid is None else torch.tensor(valid)

    torch.testing.assert_close(validity_weighted_frame_mean(torch.tensor(per_frame), valid), torch.tensor(expected))
