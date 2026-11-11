"""The shared reward terms in ``eden/managers/terms/rewards/common.py``."""

from __future__ import annotations

import pytest
import torch

from eden.managers.action_manager import ActionManager
from eden.managers.terms.actions.joint_actions import ImplicitPDController
from eden.managers.terms.rewards.common import (
    ActionRateL2Smooth,
    DofsAccL2,
    MetricReward,
    TaskSuccess,
    UndesiredContacts,
)
from eden.options.managers.actions import ActionManagerOptions

pytestmark = pytest.mark.requires_genesis


def test_dofs_acc_l2_penalises_the_change_since_its_own_last_read(panda_env, make_term, restore_state):
    """The term owns its previous-velocity buffer, so the penalty is live with no external wiring."""
    robot = restore_state(panda_env.entities["robot"])
    term = make_term(DofsAccL2, panda_env, weight=-1.0, entity_name="robot")
    vel = torch.zeros(2, 9)
    vel[:, :2] = torch.tensor([[1.0, 1.0], [2.0, 0.0]])
    robot.set_dofs_vel(vel)
    torch.testing.assert_close(term.compute(), torch.tensor([2.0, 4.0]))  # against the zeroed buffer
    vel[:, 0] += 2.0
    robot.set_dofs_vel(vel)
    torch.testing.assert_close(term.compute(), torch.tensor([4.0, 4.0]))  # against the last read

    term.reset(envs_idx=torch.tensor([0]))  # env 0 forgets; env 1 keeps its buffer
    torch.testing.assert_close(term.compute(), torch.tensor([10.0, 0.0]))  # 3^2 + 1^2 from zero


@pytest.fixture
def arm_actions(panda_env, install_manager):
    """A real action manager with one two-DOF term, ``arm``, wired in as ``env.action_manager``."""
    arm = ImplicitPDController.configure(entity_name="robot", dofs_name=["joint1", "joint2"])
    return install_manager(panda_env, "action_manager", ActionManager(panda_env, ActionManagerOptions(arm=arm)))


@pytest.mark.parametrize("action_term_name", [None, "arm"], ids=["whole-action", "one-term"])
def test_action_rate_l2_smooth_keeps_its_second_difference_live(panda_env, make_term, arm_actions, action_term_name):
    """``||a_t - a_{t-1}||^2 + ||a_t - 2 a_{t-1} + a_{t-2}||^2`` with ``a_{t-2}`` held by the term.

    Sized at build, so the first step already sees a zero two steps back.
    """
    term = make_term(ActionRateL2Smooth, panda_env, weight=-1.0, action_term_name=action_term_name)
    arm_actions.compute(torch.zeros(2, 2))
    arm_actions.compute(torch.tensor([[1.0, 0.0]] * 2))
    torch.testing.assert_close(term.compute(), torch.full((2,), 2.0))  # a_{t-2} = 0: 1 + 1
    arm_actions.compute(torch.tensor([[2.0, 0.0]] * 2))
    torch.testing.assert_close(term.compute(), torch.full((2,), 1.0))  # 1 + (2 - 2 + 0)^2

    term.reset(envs_idx=None)
    arm_actions.compute(torch.tensor([[2.0, 0.0]] * 2))
    torch.testing.assert_close(term.compute(), torch.full((2,), 4.0))  # 0 + (2 - 4 + 0)^2 after the reset


def test_action_rate_l2_smooth_built_under_inference_mode_resets_from_a_plain_step_loop(
    panda_env, make_term, arm_actions
):
    """``RewardManager.compute`` runs in inference mode and ``reset()`` may not (``eden smoke``'s plain loop).

    A buffer born under inference mode made that reset raise "Inplace update to inference tensor outside
    InferenceMode"; the term's buffer is a normal tensor however it is built, and a reset clears it.
    """
    with torch.inference_mode():
        term = make_term(ActionRateL2Smooth, panda_env, weight=-1.0)
    arm_actions.compute(torch.ones(2, 2))
    arm_actions.compute(torch.full((2, 2), 3.0))
    with torch.inference_mode():
        term.compute()  # the two-steps-old buffer now holds the first action
    term.reset(envs_idx=torch.tensor([0]))
    term.reset()
    arm_actions.compute(torch.full((2, 2), 3.0))
    torch.testing.assert_close(
        term.compute(), torch.full((2,), 18.0)
    )  # 0 + (3 - 6 + 0)^2 per DOF: nothing two steps back


def test_action_rate_l2_smooth_rejects_an_unknown_action_term_at_build(panda_env, make_term, arm_actions):
    with pytest.raises(KeyError, match="nope"):
        make_term(ActionRateL2Smooth, panda_env, weight=-1.0, action_term_name="nope")


def test_undesired_contacts_counts_the_selected_links_over_the_threshold(contact_env, make_term):
    """The stack rests on its ``foot`` alone; ``torso`` and ``shin`` touch nothing."""
    for links_name, count in ((["foot"], 1.0), ([".*"], 1.0), (["shin", "torso"], 0.0)):
        term = make_term(UndesiredContacts, contact_env, entity_name="stack", links_name=links_name, threshold=1.0)
        assert term.compute().tolist() == [count, count], links_name
    with pytest.raises(ValueError, match="Not all query patterns are matched"):
        make_term(UndesiredContacts, contact_env, entity_name="stack", links_name=["nonexistent_link"])


def test_metric_reward_is_the_metric_value(panda_env, make_term, score_metric):
    drive = score_metric(panda_env)
    term = make_term(MetricReward, panda_env, metric_name="score")
    drive([0.5, 2.0])
    torch.testing.assert_close(term.compute(), torch.tensor([0.5, 2.0]))
    drive([1.0, 3.0])
    torch.testing.assert_close(term.compute(), torch.tensor([1.0, 3.0]))


def test_metric_reward_takes_a_cumulative_metric_only_when_asked(panda_env, make_term, score_metric):
    """A cumulative value is the running episode sum; paying it per step integrates it twice."""
    drive = score_metric(panda_env, is_cumulative=True)
    with pytest.raises(ValueError, match="cumulative"):
        make_term(MetricReward, panda_env, metric_name="score")
    term = make_term(MetricReward, panda_env, metric_name="score", allow_cumulative=True)
    drive([1.0, 2.0])
    drive([1.0, 2.0])
    torch.testing.assert_close(term.compute(), torch.tensor([2.0, 4.0]))


def test_task_success_pays_one_where_every_success_metric_holds(panda_env, make_term, score_metric):
    """``score`` succeeds at 1.0 (higher is better): env 0 falls short, env 1 makes it."""
    drive = score_metric(panda_env, success_threshold=1.0)
    term = make_term(TaskSuccess, panda_env)
    drive([0.5, 2.0])
    torch.testing.assert_close(term.compute(), torch.tensor([0.0, 1.0]))


def test_task_success_refuses_a_task_with_no_per_step_success_metric(panda_env, make_term, score_metric):
    score_metric(panda_env, contributes_to_success=False)
    with pytest.raises(ValueError, match="no INTERVAL metric that contributes to success"):
        make_term(TaskSuccess, panda_env)
