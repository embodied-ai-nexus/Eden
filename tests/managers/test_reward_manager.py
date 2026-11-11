"""Tests for ``RewardManager``: the scalar reward, reward groups over the flat term set, episode sums,
reset logging, term lookup, and the ``compute_once`` pass around the term loop.

Grouping must never change the scalar reward, and the per-group readout must reconstruct it: the
invariant CMDP-style consumers (task/style/penalty channels) rely on.
"""

from __future__ import annotations

import pytest
import torch

from eden.managers.base import FlatManagerBase
from eden.managers.reward_manager import DEFAULT_REWARD_GROUP, REWARD_TERM_REGISTRY, RewardManager, RewardTerm
from eden.options.managers.rewards import RewardGroupOptions, RewardManagerOptions, RewardTermOptions

NUM_ENVS = 4


@REWARD_TERM_REGISTRY.register()
def const_reward(env, value: float = 1.0) -> torch.Tensor:
    return torch.full((env.num_envs,), float(value), device=env.device)


@REWARD_TERM_REGISTRY.register()
def box_height_reward(env) -> torch.Tensor:
    return env.entities["box"].get_pos()[:, 2]


@REWARD_TERM_REGISTRY.register()
def random_reward(env) -> torch.Tensor:
    return torch.rand(env.num_envs, device=env.device)


@REWARD_TERM_REGISTRY.register()
class ConstRewardTerm(RewardTerm):
    value: float = 1.0

    def compute(self) -> torch.Tensor:
        return torch.full((self.num_envs,), self.value, device=self.device)


def _shared_draw(env) -> torch.Tensor:
    """A fresh random draw per production, so sharing is visible as equality."""
    return env.reward_manager.compute_once(("reward_test_shared",), lambda: torch.rand(env.num_envs))


@REWARD_TERM_REGISTRY.register()
def shared_draw(env) -> torch.Tensor:
    return _shared_draw(env)


@REWARD_TERM_REGISTRY.register()
def shared_draw_doubled(env) -> torch.Tensor:
    return 2.0 * _shared_draw(env)


@REWARD_TERM_REGISTRY.register()
class SharedDrawThenMaybeRaises(RewardTerm):
    """Reads the shared draw, then raises while ``fail`` is set."""

    def build(self) -> None:
        self.fail = False

    def compute(self) -> torch.Tensor:
        shared = _shared_draw(self._env)
        if self.fail:
            raise RuntimeError("term failed mid-pass")
        return shared


def _const(weight: float, value: float, **kwargs) -> RewardTermOptions:
    return RewardTermOptions(name="const_reward", weight=weight, params={"value": value}, **kwargs)


def _grouped(env) -> RewardManager:
    """rg: 2*3 + 1*5 = 11, rs: 4*0.5 = 2, rp: -1*2 = -2 (per step, before dt)."""
    return RewardManager(
        env=env,
        options=RewardManagerOptions(
            rg=RewardGroupOptions(track_pos=_const(2.0, 3.0), track_rot=_const(1.0, 5.0)),
            rs=RewardGroupOptions(style=_const(4.0, 0.5)),
            rp=RewardGroupOptions(qvel_penalty=_const(-1.0, 2.0)),
        ),
    )


def _full(value: float) -> torch.Tensor:
    return torch.full((NUM_ENVS,), value)


# ---------------------------------------------------------------------------
# The scalar reward and its partition into groups
# ---------------------------------------------------------------------------


def test_flat_terms_form_the_default_group_and_sum_weighted_times_dt(box_env):
    rm = RewardManager(env=box_env, options=RewardManagerOptions(a=_const(1.0, 2.0), b=_const(0.5, 4.0)))

    reward = rm.compute(dt=box_env.dt)

    assert rm.group_term_names == {DEFAULT_REWARD_GROUP: ["a", "b"]}
    torch.testing.assert_close(reward, _full(4.0 * box_env.dt))  # (1 * 2 + 0.5 * 4) * dt
    torch.testing.assert_close(rm.get_group_reward(DEFAULT_REWARD_GROUP), reward)


def test_an_env_genesis_flags_as_corrupt_gets_zero_reward_instead_of_the_nan_its_state_produced(
    box_env, flag_solver_error
):
    """It is reset this same step; its NaN must not reach the learner's NaN check or the episode sums."""
    rm = RewardManager(
        env=box_env,
        options=RewardManagerOptions(ok=_const(1.0, 2.0), reads_corrupt_state=_const(1.0, float("nan"))),
    )
    flag_solver_error(box_env, [0, 1, 2, 3])

    reward = rm.compute(dt=box_env.dt)

    torch.testing.assert_close(reward, _full(0.0))
    torch.testing.assert_close(rm.get_group_episode_sum(DEFAULT_REWARD_GROUP, raw=False), _full(0.0))
    torch.testing.assert_close(rm.get_group_episode_sum(DEFAULT_REWARD_GROUP, raw=True), _full(0.0))


def test_only_the_flagged_env_loses_its_reward(box_env, flag_solver_error):
    rm = RewardManager(env=box_env, options=RewardManagerOptions(a=_const(1.0, 2.0)))
    flag_solver_error(box_env, [2])

    reward = rm.compute(dt=box_env.dt)

    torch.testing.assert_close(reward, torch.tensor([2.0, 2.0, 0.0, 2.0]) * box_env.dt)


def test_the_groups_partition_the_scalar_reward(box_env):
    rm = _grouped(box_env)
    assert torch.all(rm.get_group_reward("rg") == 0.0)  # before the first compute

    reward = rm.compute(dt=box_env.dt)

    assert tuple(rm.group_term_names) == ("rg", "rs", "rp")
    torch.testing.assert_close(sum(rm.get_group_reward(g) for g in rm.group_term_names), reward)
    for group, per_step in (("rg", 11.0), ("rs", 2.0), ("rp", -2.0)):
        torch.testing.assert_close(rm.get_group_reward(group), _full(per_step * box_env.dt))


def test_term_value_is_one_terms_unweighted_output(box_env):
    rm = _grouped(box_env)
    rm.compute(dt=box_env.dt)
    torch.testing.assert_close(rm.get_term_value("track_pos"), _full(3.0))
    torch.testing.assert_close(rm.get_term_value("qvel_penalty"), _full(2.0))


def test_weighted_term_values_reconstruct_the_scalar_reward_and_each_group(box_env):
    rm = _grouped(box_env)
    reward = rm.compute(dt=box_env.dt)

    def share(name):
        return rm.get_term_options(name).weight * rm.get_term_value(name) * box_env.dt

    torch.testing.assert_close(sum(share(name) for name in rm.active_terms), reward)
    for group, names in rm.group_term_names.items():
        torch.testing.assert_close(sum(share(name) for name in names), rm.get_group_reward(group))


def test_term_values_sum_to_the_raw_episode_sum_without_dt(box_env):
    """The episode sums accumulate ``value * dt``; the step readout is the value itself."""
    rm = RewardManager(env=box_env, options=RewardManagerOptions(r=_const(2.0, 3.0)))
    total = _full(0.0)
    for _ in range(3):
        rm.compute(dt=box_env.dt)
        total += rm.get_term_value("r")
    torch.testing.assert_close(total, rm.get_episode_sum("r", raw=True) / box_env.dt)


def test_term_value_is_a_live_view_zero_before_the_first_compute_and_kept_across_a_reset(box_env):
    """Like ``MetricManager.get_value``: held once, read every step; a caller clones to keep a step."""
    rm = RewardManager(env=box_env, options=RewardManagerOptions(r=RewardTermOptions(name="random_reward")))
    view = rm.get_term_value("r")
    torch.testing.assert_close(view, _full(0.0))

    rm.compute(dt=box_env.dt)
    kept = view.clone()
    rm.reset()  # auto-reset runs after reward: the readout still holds the step that was rewarded
    torch.testing.assert_close(rm.get_term_value("r"), kept)

    rm.compute(dt=box_env.dt)
    assert not torch.equal(view, kept)
    torch.testing.assert_close(view, rm.get_term_value("r"))


def test_a_term_a_curriculum_zeroes_still_reports_its_output(box_env):
    """Why the output has its own buffer: it cannot be recovered by dividing the weighted one by the weight."""
    rm = _grouped(box_env)
    rm.get_term("track_pos").weight = 0.0
    rm.compute(dt=box_env.dt)
    torch.testing.assert_close(rm.get_term_value("track_pos"), _full(3.0))
    torch.testing.assert_close(rm.get_group_reward("rg"), _full(5.0 * box_env.dt))


def test_the_term_value_is_temporally_masked_and_zero_for_a_corrupt_env(box_env, flag_solver_error):
    """[0.5 s, 1.0 s] at dt = 0.02 is steps [25, 50]; env 3 is in the window but flagged corrupt."""
    rm = RewardManager(env=box_env, options=RewardManagerOptions(gated=_const(2.0, 3.0, range_s=[(0.5, 1.0)])))
    flag_solver_error(box_env, [3])
    try:
        box_env.episode_length_buf[:] = torch.tensor([24, 25, 50, 30])
        rm.compute(dt=box_env.dt)
        torch.testing.assert_close(rm.get_term_value("gated"), torch.tensor([0.0, 3.0, 3.0, 0.0]))
    finally:
        box_env.episode_length_buf[:] = 0


def test_the_term_value_of_a_corrupt_env_is_zero_not_the_nan_its_state_produced(box_env, flag_solver_error):
    """The learner stores this value per transition, so it needs the same guard as the scalar reward."""
    rm = RewardManager(env=box_env, options=RewardManagerOptions(reads_corrupt_state=_const(1.0, float("nan"))))
    flag_solver_error(box_env, [0, 1, 2, 3])

    rm.compute(dt=box_env.dt)

    torch.testing.assert_close(rm.get_term_value("reads_corrupt_state"), _full(0.0))


def test_flat_and_grouped_terms_mix(box_env):
    rm = RewardManager(
        env=box_env,
        options=RewardManagerOptions(alive=_const(1.0, 1.0), rg=RewardGroupOptions(track=_const(2.0, 3.0))),
    )

    reward = rm.compute(dt=box_env.dt)

    assert tuple(rm.group_term_names) == (DEFAULT_REWARD_GROUP, "rg")
    torch.testing.assert_close(rm.get_group_reward(DEFAULT_REWARD_GROUP), _full(box_env.dt))
    torch.testing.assert_close(rm.get_group_reward(DEFAULT_REWARD_GROUP) + rm.get_group_reward("rg"), reward)


def test_a_group_named_like_the_default_merges_with_the_flat_terms_and_warns(box_env, eden_caplog):
    rm = RewardManager(
        env=box_env,
        options=RewardManagerOptions(
            alive=_const(1.0, 1.0), **{DEFAULT_REWARD_GROUP: RewardGroupOptions(bonus=_const(1.0, 2.0))}
        ),
    )

    reward = rm.compute(dt=box_env.dt)

    assert rm.group_term_names == {DEFAULT_REWARD_GROUP: ["alive", "bonus"]}
    torch.testing.assert_close(rm.get_group_reward(DEFAULT_REWARD_GROUP), reward)
    assert any("declared explicitly alongside flat terms" in r.getMessage() for r in eden_caplog.records)


def _gated(term_name: str, *windows: tuple[float, float]) -> RewardGroupOptions:
    """A group holding one constant term gated to ``windows``."""
    return RewardGroupOptions(**{term_name: _const(1.0, 1.0, range_s=list(windows))})


@pytest.mark.parametrize(
    "group, steps, active",
    [
        # [0.5 s, 1.0 s] at dt = 0.02 is steps [25, 50], both ends inclusive; a raw-step reading fired only at 1.
        ("single", [24, 25, 50, 51], [False, True, True, False]),
        # OR over both windows, per env: [25, 50] or [100, 150].
        ("multi", [40, 75, 100, 151], [True, False, True, False]),
        # 0.51 / dt = 25.5 -> 26 and 0.99 / dt = 49.5 -> 50 (round half to even).
        ("fractional", [25, 26, 50, 51], [False, True, True, False]),
    ],
)
def test_range_s_gates_a_term_on_seconds_scaled_to_steps(box_env, group, steps, active):
    """Read back per window through ``get_group_reward``, one group per window."""
    assert box_env.dt == pytest.approx(0.02)
    rm = RewardManager(
        env=box_env,
        options=RewardManagerOptions(
            single=_gated("one_window", (0.5, 1.0)),
            multi=_gated("two_windows", (0.5, 1.0), (2.0, 3.0)),
            fractional=_gated("fractional_window", (0.51, 0.99)),
        ),
    )
    try:
        box_env.episode_length_buf[:] = torch.tensor(steps)
        rm.compute(dt=box_env.dt)
        assert (rm.get_group_reward(group) != 0.0).tolist() == active
    finally:
        box_env.episode_length_buf[:] = 0


def test_the_weights_follow_a_term_whose_weight_changes_after_build(box_env):
    """A curriculum moves ``term.weight``; ``compute()`` must pick it up without a rebuild."""
    rm = RewardManager(env=box_env, options=RewardManagerOptions(a=_const(1.0, 2.0)))
    rm.get_term("a").weight = 3.0

    torch.testing.assert_close(rm.compute(dt=box_env.dt), _full(6.0 * box_env.dt))


# ---------------------------------------------------------------------------
# Episode sums and reset
# ---------------------------------------------------------------------------


def test_group_episode_sums_accumulate_weighted_and_raw_and_reset_logs_their_mean(box_env):
    rm = _grouped(box_env)
    for _ in range(3):
        rm.compute(dt=box_env.dt)

    torch.testing.assert_close(rm.get_group_episode_sum("rg", raw=False), _full(3 * 11.0 * box_env.dt))
    torch.testing.assert_close(rm.get_group_episode_sum("rg", raw=True), _full(3 * 8.0 * box_env.dt))  # 3 + 5

    extras = rm.reset()

    assert extras["Episode_Reward_Group/rg"] == pytest.approx(3 * 11.0 * box_env.dt / box_env.max_episode_length_s)
    assert {"Episode_Reward_Group/rs", "Episode_Reward_Group/rp"} <= extras.keys()
    assert torch.all(rm.get_group_episode_sum("rg", raw=False) == 0.0)


def test_the_group_episode_sum_takes_raw_by_keyword_only_with_no_default(box_env):
    """``get_episode_sum`` defaults to raw; a silent opposite default here would flip ported code."""
    rm = _grouped(box_env)
    with pytest.raises(TypeError):
        rm.get_group_episode_sum("rg")
    with pytest.raises(TypeError):
        rm.get_group_episode_sum("rg", False)


@pytest.mark.parametrize(
    "envs_idx, cleared, logged_steps",
    [
        (torch.tensor([0, 2]), [True, False, True, False], 1.0),
        (torch.tensor([], dtype=torch.long), [False] * NUM_ENVS, 0.0),  # empty selection: guarded mean
        (torch.tensor(1), [False, True, False, False], 1.0),  # 0-dim, as on the per-term path
    ],
    ids=["index", "empty", "zero-dim"],
)
def test_a_partial_reset_clears_and_logs_only_the_selected_envs(box_env, envs_idx, cleared, logged_steps):
    rm = _grouped(box_env)
    rm.compute(dt=box_env.dt)

    extras = rm.reset(envs_idx=envs_idx)

    assert extras["Episode_Reward_Group/rg"] == pytest.approx(
        logged_steps * 11.0 * box_env.dt / box_env.max_episode_length_s
    )
    assert (rm.get_group_episode_sum("rg", raw=False) == 0.0).tolist() == cleared


def test_a_masked_reset_logs_each_terms_episode_mean_over_the_reset_envs_and_clears_only_them(box_env, set_box_heights):
    """The auto-reset form: a bool mask. Weighted and raw means over the masked envs, over ``max_episode_length_s``."""
    set_box_heights(box_env, [0.0, 0.2, 0.4, 0.6])
    rm = RewardManager(
        env=box_env,
        options=RewardManagerOptions(
            a=RewardTermOptions(name="box_height_reward", weight=2.0),
            b=RewardTermOptions(name="box_height_reward", weight=-0.5),
        ),
    )
    for _ in range(3):
        rm.compute(dt=box_env.dt)
    mask = torch.tensor([True, False, True, True])
    per_env = 3 * box_env.dt / box_env.max_episode_length_s  # an episode's raw sum per unit height, normalized

    extras = rm.reset(mask)

    mean_height = (0.0 + 0.4 + 0.6) / 3
    for key, weight in (("a", 2.0), ("b", -0.5)):
        assert extras[f"Episode_Reward/{key}"].item() == pytest.approx(weight * per_env * mean_height)
        assert extras[f"Episode_Reward_Raw/{key}"].item() == pytest.approx(per_env * mean_height)
    # The env still running kept its sums.
    assert rm.reset(~mask)["Episode_Reward/a"].item() == pytest.approx(2.0 * per_env * 0.2)


def test_after_a_reset_the_group_readout_still_matches_the_step_that_was_rewarded(box_env):
    """Auto-reset runs after reward: ``reset()`` clears the sums, not the step buffer."""
    rm = _grouped(box_env)
    reward = rm.compute(dt=box_env.dt)

    rm.reset()

    torch.testing.assert_close(sum(rm.get_group_reward(g) for g in rm.group_term_names), reward)


def test_an_all_weight_zero_group_stays_declared_and_reads_zero(box_env):
    rm = RewardManager(
        env=box_env,
        options=RewardManagerOptions(
            rg=RewardGroupOptions(active=_const(1.0, 1.0), dead=_const(0.0, 1.0)),
            empty=RewardGroupOptions(dead_too=_const(0.0, 1.0)),
        ),
    )
    rm.compute(dt=box_env.dt)

    assert rm.group_term_names == {"rg": ["active"], "empty": []}
    assert torch.all(rm.get_group_reward("empty") == 0.0)
    assert torch.all(rm.get_group_episode_sum("empty", raw=False) == 0.0)
    assert rm.reset()["Episode_Reward_Group/empty"] == pytest.approx(0.0)


@pytest.mark.parametrize(
    "options, present, absent",
    [
        (dict(a=_const(1.0, 1.0)), ["Episode_Reward/a", "Episode_Reward_Raw/a"], ["Episode_Reward_Group/"]),
        (dict(rg=RewardGroupOptions(a=_const(1.0, 1.0))), ["Episode_Reward_Group/rg"], ["Episode_Reward/"]),
        (
            dict(rg=RewardGroupOptions(a=_const(1.0, 1.0)), log_per_term_rewards=True),
            ["Episode_Reward/a", "Episode_Reward_Raw/a", "Episode_Reward_Group/rg"],
            [],
        ),
        (dict(a=_const(1.0, 1.0), log_per_term_rewards=False), [], ["Episode_Reward"]),
    ],
    ids=["flat-default", "grouped-default", "grouped-per-term-on", "flat-per-term-off"],
)
def test_reset_logs_per_term_keys_only_when_asked_or_ungrouped(box_env, options, present, absent):
    """A grouped config defaults to the group sums; per-term keys dominated per-iteration log volume."""
    rm = RewardManager(env=box_env, options=RewardManagerOptions(**options))
    rm.compute(dt=box_env.dt)

    keys = rm.reset().keys()

    assert set(present) <= keys
    assert not [key for key in keys for prefix in absent if key.startswith(prefix)]


@pytest.mark.parametrize("explicit", [False, True], ids=["defaulted", "set-explicitly"])
def test_grouping_announces_the_per_term_keys_it_drops_unless_the_flag_was_set(box_env, eden_caplog, explicit):
    """The flip is silent otherwise: every ``Episode_Reward/<term>`` key disappears from wandb."""
    extra = dict(log_per_term_rewards=False) if explicit else {}
    RewardManager(env=box_env, options=RewardManagerOptions(rg=RewardGroupOptions(a=_const(2.0, 3.0)), **extra))

    announced = [r.getMessage() for r in eden_caplog.records if "Episode_Reward/<term>" in r.getMessage()]
    assert (announced == []) is explicit
    assert explicit or "log_per_term_rewards=True" in announced[0]


def test_the_summary_has_a_group_column_only_when_groups_are_declared(box_env):
    summary = _grouped(box_env).summary()
    assert "Group" in summary and all(group in summary for group in ("rg", "rs", "rp"))
    assert "Group" not in RewardManager(env=box_env, options=RewardManagerOptions(a=_const(1.0, 1.0))).summary()


# ---------------------------------------------------------------------------
# Terms and lookup
# ---------------------------------------------------------------------------


def test_the_grouped_manager_keeps_the_flat_term_api(box_env):
    rm = _grouped(box_env)

    assert isinstance(rm, FlatManagerBase)  # groups label one flat set; the API is inherited, not rebuilt
    assert rm.active_terms == ["track_pos", "track_rot", "style", "qvel_penalty"]
    assert dict(rm.iter_terms()).keys() == set(rm.active_terms)
    assert rm.get_term("style").weight == 4.0
    assert rm.get_term_options("track_pos").weight == 2.0


def test_group_term_names_is_a_copy(box_env):
    rm = _grouped(box_env)
    view = rm.group_term_names
    view.pop("rs")
    view["rg"].append("bogus")

    assert rm.group_term_names == {"rg": ["track_pos", "track_rot"], "rs": ["style"], "rp": ["qvel_penalty"]}


def test_get_episode_sum_is_a_live_view_of_one_terms_raw_or_weighted_sum(box_env):
    """The view is the contract ``cumulative_reward_below_threshold`` relies on: held once, read every step."""
    rm = RewardManager(
        env=box_env,
        options=RewardManagerOptions(first=_const(1.0, 10.0), skipped=_const(0.0, 1.0), third=_const(2.0, 30.0)),
    )
    raw_third = rm.get_episode_sum("third")

    rm.compute(dt=box_env.dt)

    torch.testing.assert_close(rm.get_episode_sum("first"), _full(10.0 * box_env.dt))
    torch.testing.assert_close(raw_third, _full(30.0 * box_env.dt))
    torch.testing.assert_close(rm.get_episode_sum("third", raw=False), _full(60.0 * box_env.dt))


@pytest.mark.parametrize("name", ["not_a_term", "skipped"], ids=["unknown", "weight-zero"])
def test_get_episode_sum_of_an_inactive_name_lists_the_active_terms(box_env, name):
    rm = RewardManager(env=box_env, options=RewardManagerOptions(first=_const(1.0, 1.0), skipped=_const(0.0, 1.0)))

    with pytest.raises(KeyError, match=rf"'{name}' not active. Active terms: \['first'\]"):
        rm.get_episode_sum(name)


@pytest.mark.parametrize("readout", ["get_episode_sum", "get_term_value"])
def test_only_a_weight_zero_name_is_told_it_was_never_built(box_env, readout):
    """A typo must not send the reader looking for a weight-0 entry that is not there."""
    rm = RewardManager(
        env=box_env,
        options=RewardManagerOptions(first=_const(1.0, 1.0), rg=RewardGroupOptions(skipped=_const(0.0, 1.0))),
    )
    with pytest.raises(KeyError, match=r"reward_options\.rg\.skipped has weight 0\.0, so it is never built"):
        getattr(rm, readout)("skipped")
    with pytest.raises(KeyError) as typo:
        getattr(rm, readout)("skiped")
    assert "weight 0.0" not in str(typo.value)


def test_tags_reach_both_function_and_class_terms(box_env):
    rm = RewardManager(
        env=box_env,
        options=RewardManagerOptions(
            func=_const(-1.0, 1.0, tags=["penalty", "effort"]),
            cls=ConstRewardTerm.configure(weight=2.0, tags=["tracking"]),
            untagged=_const(1.0, 1.0),
        ),
    )

    assert {name: term.tags for name, term in rm.iter_terms()} == {
        "func": ["penalty", "effort"],
        "cls": ["tracking"],
        "untagged": [],
    }


def test_an_unknown_group_names_the_active_groups(box_env):
    rm = _grouped(box_env)
    with pytest.raises(KeyError, match="'typo'.*rg"):
        rm.get_group_reward("typo")
    with pytest.raises(KeyError, match="'typo'.*rg"):
        rm.get_group_episode_sum("typo", raw=False)


def test_active_names_are_unique_across_groups_and_a_weight_zero_term_reserves_none(box_env):
    """They share one flat buffer and one log namespace; a term never built occupies neither."""
    with pytest.raises(ValueError, match="Duplicate reward term name 'track'"):
        RewardManager(
            env=box_env,
            options=RewardManagerOptions(
                rg=RewardGroupOptions(track=_const(1.0, 1.0)), rs=RewardGroupOptions(track=_const(1.0, 2.0))
            ),
        )

    rm = RewardManager(
        env=box_env,
        options=RewardManagerOptions(
            rg=RewardGroupOptions(track=_const(0.0, 1.0)), rs=RewardGroupOptions(track=_const(1.0, 2.0))
        ),
    )
    assert rm.group_term_names == {"rg": [], "rs": ["track"]}
    assert rm.get_term("track").weight == 1.0


def test_a_weight_zero_warning_names_the_config_path(box_env, eden_caplog):
    """Where the zero was written, so the line to change can be found."""
    RewardManager(
        env=box_env,
        options=RewardManagerOptions(
            flat_dead=_const(0.0, 1.0),
            rg=RewardGroupOptions(active=_const(1.0, 1.0), grouped_dead=_const(0.0, 1.0)),
        ),
    )

    warnings = [r.getMessage() for r in eden_caplog.records if r.levelname == "WARNING"]
    assert any(m.startswith("reward_options.flat_dead has weight 0.0") for m in warnings), warnings
    assert any(m.startswith("reward_options.rg.grouped_dead has weight 0.0") for m in warnings), warnings


# ---------------------------------------------------------------------------
# compute_once
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sharing_env(make_box_env):
    """An env whose own reward manager holds the three ``compute_once`` terms, found as ``env.reward_manager``."""
    return make_box_env(
        reward_options=RewardManagerOptions(
            first=RewardTermOptions(name="shared_draw", weight=1.0),
            doubled=RewardTermOptions(name="shared_draw_doubled", weight=1.0),
            flaky=SharedDrawThenMaybeRaises.configure(weight=1.0),
        )
    )


def test_reward_terms_sharing_a_key_get_one_production_per_pass(sharing_env):
    manager = sharing_env.reward_manager
    manager.reset()

    manager.compute(dt=1.0)  # the raw episode sums then hold exactly this pass's term values
    first = manager.get_episode_sum("first").clone()
    torch.testing.assert_close(manager.get_episode_sum("doubled"), 2.0 * first)

    manager.reset()
    manager.compute(dt=1.0)  # a new pass starts empty
    assert not torch.equal(manager.get_episode_sum("first"), first)
    torch.testing.assert_close(manager.get_episode_sum("doubled"), 2.0 * manager.get_episode_sum("first"))


def test_a_reward_term_that_raises_mid_pass_does_not_leave_the_memo_behind(sharing_env):
    manager = sharing_env.reward_manager
    flaky = manager.get_term("flaky")
    flaky.fail = True
    try:
        with pytest.raises(RuntimeError, match="term failed mid-pass"):
            manager.compute(dt=1.0)
    finally:
        flaky.fail = False

    assert manager.compute_once(("reward_test_shared",), lambda: "fresh") == "fresh"


def test_compute_and_a_masked_reset_are_sync_free(cuda_stand_in_env, count_cuda_syncs):
    """Neither the per-step weight write nor the reset's gathers may sync once per term."""
    rm = RewardManager(
        env=cuda_stand_in_env,
        options=RewardManagerOptions(
            a=RewardTermOptions(name="random_reward", weight=2.0),
            b=RewardTermOptions(name="random_reward", weight=-0.5),
        ),
    )
    rm.compute(dt=0.02)
    mask = torch.tensor([True, False, True, True], device="cuda")

    assert count_cuda_syncs(lambda: rm.compute(dt=0.02)) == 0
    assert count_cuda_syncs(lambda: rm.reset(mask)) == 0
