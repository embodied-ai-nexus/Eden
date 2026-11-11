"""The shared termination terms in ``eden/managers/terms/terminations/common.py``."""

from __future__ import annotations

import pytest
import torch

from eden.managers.terms.rewards.common import UndesiredContacts
from eden.envs.base import env_class_for_config
from eden.managers.reward_manager import REWARD_TERM_REGISTRY
from eden.managers.terms.terminations.common import (
    IllegalLinkContact,
    MetricAbove,
    OnMetricSuccess,
    cumulative_reward_below_threshold,
    dofs_pos_limit_exceeded,
    dofs_torque_limit_exceeded,
    dofs_vel_limit_exceeded,
)
from eden.options import BoxOptions, EnvOptions, SceneOptions
from eden.options.managers.rewards import RewardManagerOptions, RewardTermOptions
from eden.options.robots import FrankaEmikaPanda
from eden.utils.configs import EdenRLConfig

pytestmark = pytest.mark.requires_genesis

STACK_WEIGHT_N = 4.8 * 9.81  # what the stack's ``foot`` carries once settled


@pytest.mark.parametrize(
    ("params", "error", "match"),
    [
        ({"links_name": []}, ValueError, "IllegalLinkContact: links_name=.*matched no links"),
        ({"links_name": ["nonexistent_link"]}, ValueError, "Not all query patterns are matched"),
        ({"entity_name": "nope", "links_name": [".*"]}, KeyError, "No scene entity named 'nope'"),
    ],
    ids=["empty-links", "unmatched-links", "unknown-entity"],
)
def test_illegal_link_contact_rejects_a_bad_selection_at_build(contact_env, make_term, params, error, match):
    with pytest.raises(error, match=match):
        make_term(IllegalLinkContact, contact_env, **{"entity_name": "stack", **params})


@pytest.mark.parametrize(
    ("links_name", "fires"),
    [(["foot"], True), (["torso"], False), (["shin", "foot"], True), (["shin", "torso"], False)],
)
def test_illegal_link_contact_reads_only_the_selected_links(contact_env, make_term, links_name, fires):
    """Only ``foot`` is loaded: a term that ignored its link subset would fire for every selection."""
    done = make_term(IllegalLinkContact, contact_env, entity_name="stack", links_name=links_name).compute()
    assert done.dtype == torch.bool and done.tolist() == [fires, fires]


def test_illegal_link_contact_gates_the_selected_links_on_force(contact_env, make_term):
    def done(links_name, force_threshold):
        term = make_term(
            IllegalLinkContact, contact_env, entity_name="stack", links_name=links_name, force_threshold=force_threshold
        )
        return term.compute().tolist()

    assert done(["foot"], 0.5 * STACK_WEIGHT_N) == [True, True]
    assert done(["foot"], 2.0 * STACK_WEIGHT_N) == [False, False]
    assert done(["torso"], 1e-9) == [False, False]  # a clear link reads 0 N under any threshold


def test_illegal_link_contact_agrees_with_undesired_contacts(contact_env, make_term):
    """Terminating on a link set and penalising it at one threshold are the same predicate."""
    for links_name in (["foot"], ["torso"], [".*"]):
        params = {"entity_name": "stack", "links_name": links_name}
        done = make_term(IllegalLinkContact, contact_env, force_threshold=1.0, **params).compute()
        penalty = make_term(UndesiredContacts, contact_env, threshold=1.0, **params).compute()
        assert torch.equal(done, penalty > 0), links_name


def test_on_metric_success_uses_the_metric_success_check(contact_env, make_term, score_metric):
    drive = score_metric(contact_env, success_threshold=1.0)
    term = make_term(OnMetricSuccess, contact_env, metric_name="score")
    drive([0.5, 1.0])
    assert term.compute().tolist() == [False, True]  # higher-is-better, threshold inclusive
    drive([2.0, 0.0])
    assert term.compute().tolist() == [True, False]


@pytest.mark.parametrize(
    ("fields", "match"),
    [({"metric_mode": "reset"}, "INTERVAL"), ({"contributes_to_success": False}, "contributes_to_success=False")],
)
def test_on_metric_success_rejects_a_metric_without_a_step_success_at_build(
    contact_env, make_term, score_metric, fields, match
):
    score_metric(contact_env, **fields)
    with pytest.raises(ValueError, match=match):
        make_term(OnMetricSuccess, contact_env, metric_name="score")


def test_metric_above_is_strictly_above_the_threshold(contact_env, make_term, score_metric):
    # contributes_to_success=False is accepted: a collision flag may end the episode outside the success count.
    drive = score_metric(contact_env, contributes_to_success=False)
    at_one = make_term(MetricAbove, contact_env, metric_name="score", threshold=1.0)
    positive = make_term(MetricAbove, contact_env, metric_name="score")  # the default threshold, 0.0
    drive([1.0, 1.5])
    assert at_one.compute().tolist() == [False, True]
    drive([0.0, 0.25])
    assert positive.compute().tolist() == [False, True]


def test_metric_above_on_a_cumulative_metric_bounds_the_episode_sum(contact_env, make_term, score_metric):
    drive = score_metric(contact_env, is_cumulative=True)
    term = make_term(MetricAbove, contact_env, metric_name="score", threshold=1.5)
    drive([1.0, 0.5])
    assert term.compute().tolist() == [False, False]
    drive([1.0, 0.5])
    assert term.compute().tolist() == [True, False]  # sums 2.0 and 1.0


# ---------------------------------------------------------------------------
# Reward floor and DOF-limit terminations on an RL env
# ---------------------------------------------------------------------------


# Not ``box_height_reward``: a term's registry key is its name, and test_reward_manager.py registers that one.
@REWARD_TERM_REGISTRY.register()
def box_z_reward(env) -> torch.Tensor:
    return env.entities["box"].get_pos()[:, 2]


def _build(num_envs: int, *, robot: bool, **env_options):
    scene = dict(box=BoxOptions(size=(0.1, 0.1, 0.1), default_root_pos=(1.0, 0.0, 0.5)))
    if robot:
        scene["robot"] = FrankaEmikaPanda()
    config = EdenRLConfig(
        env_options=EnvOptions(num_envs=num_envs, **env_options),
        scene_options=SceneOptions(**scene),
        reward_options=RewardManagerOptions(task_reward=RewardTermOptions(name="box_z_reward", weight=1.0)),
    )
    env = env_class_for_config(config).from_config(config, show_viewer=False)
    env.build()
    env.reset()
    return env


@pytest.fixture(scope="module")
def rl_env():
    """Five envs at dt = 0.02, a Franka and a box, with a reward manager paying the box's height."""
    return _build(5, robot=True)


@pytest.fixture(scope="module")
def fine_env():
    """One env at dt = 0.005."""
    return _build(1, robot=False, sim_dt=0.005, decimation=1)


def _episode(env, cumulative, episode_lengths) -> None:
    """Give every env a raw ``task_reward`` episode sum and an episode length."""
    pos = torch.zeros(env.num_envs, 3)
    pos[:, 0] = 1.0
    pos[:, 2] = torch.tensor(cumulative)
    env.entities["box"].set_pos(pos)
    env.reward_manager.reset()
    env.reward_manager.compute(dt=1.0)  # raw sum = height * 1.0
    env.episode_length_buf[:] = torch.tensor(episode_lengths)


@pytest.fixture(autouse=True)
def zero_episode_lengths(request):
    """The limit and reward terms read ``episode_length_buf``; leave it at zero for the next test."""
    yield
    for name in ("rl_env", "fine_env"):
        if name in request.fixturenames:
            request.getfixturevalue(name).episode_length_buf[:] = 0


# ---------------------------------------------------------------------------
# cumulative_reward_below_threshold: cum < per_step_threshold * dt * completed_steps
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs, cumulative, lengths, fired",
    [
        # completed = N - 1; floor = 0.5 * 0.02 * completed: 0, 0, 0.04, 0.09, 0.20
        ({}, [0.0, 0.0, 0.10, 0.05, 0.10], [0, 1, 5, 10, 21], [False, False, False, True, True]),
        # check_interval=5: the largest rung strictly below N - 1, ((N - 1) // 5) * 5; strict "<" at a rung
        (dict(check_interval=5), [0.0, 0.10, 0.05, 0.04, 0.10], [1, 5, 6, 10, 21], [False, False, False, True, True]),
        # grace_steps=10 silences every rl_env with N <= 10
        (dict(grace_steps=10), [0.0] * 5, [1, 5, 10, 11, 21], [False, False, False, True, True]),
    ],
    ids=["continuous", "interval-ladder", "grace-steps"],
)
def test_the_floor_grows_with_the_completed_steps(rl_env, kwargs, cumulative, lengths, fired):
    _episode(rl_env, cumulative, lengths)

    out = cumulative_reward_below_threshold(rl_env, reward_term="task_reward", per_step_threshold=0.5, **kwargs)

    assert out.dtype == torch.bool
    assert out.tolist() == fired


def test_the_floor_is_in_raw_per_step_reward_units_so_it_scales_with_dt(rl_env, fine_env):
    """0.5 * 0.02 * 20 = 0.2 > 0.1 fires; 0.5 * 0.005 * 20 = 0.05 < 0.1 does not."""
    _episode(rl_env, [0.10] * 5, [21] * 5)
    _episode(fine_env, [0.10], [21])

    assert (
        cumulative_reward_below_threshold(rl_env, reward_term="task_reward", per_step_threshold=0.5).tolist()
        == [True] * 5
    )
    assert cumulative_reward_below_threshold(fine_env, reward_term="task_reward", per_step_threshold=0.5).tolist() == [
        False
    ]


# ---------------------------------------------------------------------------
# DOF limits
# ---------------------------------------------------------------------------


def test_a_dof_outside_its_soft_position_limits_ends_the_episode(rl_env):
    robot = rl_env.entities["robot"]
    soft = robot.soft_dofs_pos_limits[0]
    pos = soft.mean(dim=-1).expand(rl_env.num_envs, -1).clone()  # every DOF mid-range
    pos[1, 0] = soft[0, 1] + 0.05  # joint1 past its upper soft limit
    pos[3, 3] = soft[3, 0] - 0.05  # joint4 past its lower one
    robot.set_dofs_pos(pos)
    try:
        assert dofs_pos_limit_exceeded(rl_env, entity_name="robot").tolist() == [False, True, False, True, False]
    finally:
        rl_env.reset()


def test_a_dof_faster_than_max_vel_ends_the_episode(rl_env):
    robot = rl_env.entities["robot"]
    vel = torch.zeros(rl_env.num_envs, 9)
    vel[0, 2] = 0.5
    vel[2, 2] = -1.1
    robot.set_dofs_vel(vel)
    try:
        assert dofs_vel_limit_exceeded(rl_env, entity_name="robot", max_vel=1.0).tolist() == [
            False,
            False,
            True,
            False,
            False,
        ]
    finally:
        rl_env.reset()


def test_a_dof_force_past_its_effort_limit_ends_the_episode(rl_env):
    robot = rl_env.entities["robot"]
    force = torch.zeros(rl_env.num_envs, 9)
    force[4] = 500.0  # far past every Panda effort limit (87 / 12 / 100 N·m)
    robot.control_dofs_force(force)
    rl_env.scene.step()
    try:
        assert dofs_torque_limit_exceeded(rl_env, entity_name="robot").tolist() == [False, False, False, False, True]
    finally:
        rl_env.reset()


@pytest.mark.parametrize("termination", [dofs_pos_limit_exceeded, dofs_torque_limit_exceeded], ids=lambda f: f.__name__)
def test_an_entity_without_limits_never_ends_the_episode(rl_env, termination):
    """A free box has no soft position limits and no effort limits."""
    out = termination(rl_env, entity_name="box")

    assert out.dtype == torch.bool and out.tolist() == [False] * rl_env.num_envs
