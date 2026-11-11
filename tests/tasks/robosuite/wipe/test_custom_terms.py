"""Wipe: the marker latch and its reward, the unsafe-step gate, the wrist force, and the dirt path."""

from __future__ import annotations

# isort: off
import eden  # noqa: F401 - isolate the DDP rank before importing Torch
# isort: on

import pytest
import torch

from eden.extensions.scripted_demos.rig import RigDriver
from eden.tasks import TASK_REGISTRY
from eden.tasks.robosuite.wipe.config import (
    _PATH_X_HALF,
    _PATH_Y_HALF,
    _TABLE_TOP_Z,
    _TABLE_XY,
    MARKER_NAMES,
    NUM_MARKERS,
)
from eden.utils.smoke import build_and_step
from tests.support.envs import build_env

#: Every tool target the tests command stays over the table, between its top and well above the arm's reach.
_WORKSPACE = ((-0.1, -0.4, _TABLE_TOP_Z), (0.4, 0.4, _TABLE_TOP_Z + 0.4))


@pytest.fixture(scope="module")
def env():
    config = TASK_REGISTRY.build("robosuite_wipe")
    config.env_options.num_envs = 2
    return build_env(config, reset=True)


def _drive(env, driver: RigDriver, target: torch.Tensor, steps: int, *, check=None) -> None:
    """Step the env toward a tool target with the fingers closed, calling ``check(before)`` after each step.

    The hand keeps the orientation it starts with: the Panda's home heading is about pi from the
    driver's heading 0, and asking for 0 would wind the last joint into its stop.
    """
    closed = torch.ones(env.num_envs, device=env.device)
    _, rot = driver.tool_pose()
    for _ in range(steps):
        before = {
            "wiped": env.metric_manager.get_value("markers_wiped").clone(),
            "paid": env.reward_manager.get_episode_sum("unit_wiped", raw=True).clone(),
        }
        env.step(driver.command(target, rot, closed))
        assert not env.reset_buf.any(), "an episode ended under the test: the hand hit something it should not"
        if check is not None:
            check(before)


@pytest.mark.requires_genesis
def test_a_wiped_marker_latches_and_pays_in_the_step_it_is_wiped(env):
    """The latch is the whole reward signal: count a marker once, in the step the pads pass over it, never again."""
    env.reset(seed=0)
    driver = RigDriver(env, "robot", workspace=_WORKSPACE)
    # The marker nearest the robot's side of the table, per env: within easy reach with the hand down.
    markers = torch.stack([env.entities[name].get_pos() for name in MARKER_NAMES], dim=1)
    nearest = markers[torch.arange(env.num_envs), markers[..., 0].argmin(dim=1)]
    on_marker = torch.cat([nearest[:, :2], nearest[:, 2:] + driver.tip_reach + 0.004], dim=-1)
    above = on_marker + torch.tensor([0.0, 0.0, 0.08], device=env.device)

    def wiped_count_reaches_the_reward(before):
        newly = (env.metric_manager.get_value("markers_wiped") - before["wiped"]) * NUM_MARKERS
        paid = (env.reward_manager.get_episode_sum("unit_wiped", raw=True) - before["paid"]) / env.dt
        assert (newly >= -1e-6).all(), "a wiped marker came back"
        torch.testing.assert_close(paid, newly, atol=1e-4, rtol=0)

    _drive(env, driver, above, 50, check=wiped_count_reaches_the_reward)
    _drive(env, driver, on_marker, 30, check=wiped_count_reaches_the_reward)
    wiped = env.metric_manager.get_value("markers_wiped").clone()
    assert (wiped >= 1.0 / NUM_MARKERS).all(), "the pads sat on a marker and did not wipe it"

    lifted = on_marker + torch.tensor([0.0, 0.0, 0.15], device=env.device)
    _drive(env, driver, lifted, 20, check=wiped_count_reaches_the_reward)
    torch.testing.assert_close(env.metric_manager.get_value("markers_wiped"), wiped)


@pytest.mark.requires_genesis
def test_the_wrist_force_reads_the_table_pushing_back(env):
    env.reset(seed=1)
    assert env.metric_manager.get_value("gripper_force").tolist() == [0.0, 0.0]
    # A floor below the table top lets the tool target sink into it, so the pads press on the table.
    driver = RigDriver(env, "robot", workspace=((-0.1, -0.4, _TABLE_TOP_Z - 0.1), _WORKSPACE[1]))
    centre = torch.tensor([[_TABLE_XY[0], _TABLE_XY[1], _TABLE_TOP_Z + driver.tip_reach - 0.01]], device=env.device)
    _drive(env, driver, centre.expand(env.num_envs, 3), 40)
    force = env.metric_manager.get_value("gripper_force")
    assert (force > 1.0).all(), f"pads pressed 1 cm into the table, wrist force {force.tolist()} N"
    contact = env.observation_manager.get_term("robot0_contact").compute()
    assert contact.tolist() == [[1.0], [1.0]]
    assert env.reward_manager.get_term("contact").compute().tolist() == [1.0, 1.0]


@pytest.mark.requires_genesis
def test_an_unsafe_step_pays_the_penalty_alone_and_ends_the_episode(env):
    """As in robosuite, the step pays ``arm_limit_collision_penalty`` *instead of* every other term, then terminates."""
    env.reset(seed=2)
    arm = env.entities["robot"]
    shoulder = arm.find_named_dofs([arm.dofs_name[0]])
    _, high = (limit[:, shoulder.columns] for limit in arm.get_dofs_limit())
    q = arm.get_dofs_pos(dofs_idx_local=shoulder.dofs_idx_local)
    q[0] = high[0] - 0.05  # env 0 within 0.1 rad of its first joint's stop
    arm.set_dofs_pos(q, dofs_idx_local=shoulder.dofs_idx_local)

    _, reward, terminated, _, _ = env.step(
        torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
    )

    assert terminated.tolist() == [True, False]
    torch.testing.assert_close(reward[0], torch.tensor(-10.0 * env.dt, device=env.device))
    assert reward[1] > 0.0, "the safe env still earns its distance shaping"


@pytest.mark.requires_genesis
def test_the_wipe_path_is_sampled_inside_the_table_box_and_cached_per_reset(env):
    """The markers are kinematic, so their positions are read once per reset and reused each step.

    Both halves are load-bearing: the cache is what keeps the metric off a per-marker read in the
    hot loop, and it is only correct because ``SampleWipePath`` (a RESET event) is the one thing that
    moves a marker. If that ever stops being true the cache goes stale silently, so this pins the
    cache against the live poses right after a reset — and the sampled path against its box.
    """
    env.reset(seed=3)
    live = torch.stack([env.entities[name].get_pos() for name in MARKER_NAMES], dim=1)
    # Private on purpose: the cache is the contract under test, and it has no public reading.
    cached = env.metric_manager.get_term("markers_wiped")._markers_w
    assert torch.equal(cached, live), "the wiped-marker metric is reading a stale marker cache"
    assert (live[..., 0] - _TABLE_XY[0]).abs().max() <= _PATH_X_HALF + 1e-4
    assert (live[..., 1] - _TABLE_XY[1]).abs().max() <= _PATH_Y_HALF + 1e-4
    assert (live[..., 2] - _TABLE_TOP_Z).abs().max() < 0.01, "markers must lie on the table top"


@pytest.mark.requires_genesis
def test_a_partial_reset_resamples_only_the_resetting_envs_path(env):
    env.reset(seed=4)
    before = torch.stack([env.entities[name].get_pos() for name in MARKER_NAMES], dim=1)
    env.reset(torch.tensor([1], device=env.device))
    after = torch.stack([env.entities[name].get_pos() for name in MARKER_NAMES], dim=1)

    assert torch.equal(after[0], before[0]), "env 0 did not reset, so its path must not move"
    assert not torch.allclose(after[1], before[1]), "env 1 reset onto a fresh path"
    assert (after[1, :, 0] - _TABLE_XY[0]).abs().max() <= _PATH_X_HALF + 1e-4
    assert (after[1, :, 1] - _TABLE_XY[1]).abs().max() <= _PATH_Y_HALF + 1e-4


def test_a_bad_metric_name_fails_at_env_construction():
    """Metric handles resolve in ``build()``, not on the first step.

    The metric manager is constructed ahead of the reward and observation managers
    (``RLEnvBase._manager_order``), so a term that names a metric that does not exist must raise
    while the env is being built — not survive construction and ``KeyError`` mid-rollout.
    """
    cfg = TASK_REGISTRY.build("robosuite_wipe")
    cfg.reward_options.distance.metric_name = "no_such_metric"
    with pytest.raises(KeyError, match="no_such_metric"):
        build_and_step(cfg, num_envs=2, num_steps=0)
