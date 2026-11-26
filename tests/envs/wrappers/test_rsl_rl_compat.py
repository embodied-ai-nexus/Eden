"""The rsl-rl-lib 6.0 contract helpers, under whichever rsl-rl-lib is installed (3.x to 6.x)."""

from __future__ import annotations

import contextlib
import io
import logging
import re
import types
from collections import deque

import pytest
import torch
from tensordict import TensorDict

from eden.envs.wrappers.rsl_rl_compat import (
    FINAL_OBS_KEY,
    MEAN_REWARD_KEY,
    env_step_result,
    episode_buffers,
    log_env_step,
    log_iteration,
    loss_dict_of,
    ppo_stats,
    raise_if_nonfinite,
    record_env_step_nonfinite,
    runner_expects_stats,
    stock_nan_check_reads_nested_groups,
    update_result,
)
from eden.options.learning.rsl_rl.translate import detect_rsl_rl_version
from eden.utils.warning import warn_once
from tests.support.rsl_rl import runner_unpacks_stats


def test_the_logger_signature_probe_agrees_with_the_installed_release_number():
    assert runner_expects_stats() == runner_unpacks_stats()


def test_a_bare_loss_dict_is_its_own_loss_dict():
    losses = {"value": 1.0}
    assert loss_dict_of(losses) is losses


def test_a_key_written_into_the_pairs_loss_dict_reaches_the_pair():
    losses, stats = {"value": 1.0}, {"Optimizer/learning_rate": 1e-3}
    result = (losses, stats)
    loss_dict_of(result)["explained_variance"] = 0.5
    assert result[0] is losses
    assert result == ({"value": 1.0, "explained_variance": 0.5}, {"Optimizer/learning_rate": 1e-3})


@pytest.mark.parametrize(
    ("result", "got"),
    [
        (({"value": 1.0}, [0.1]), "a tuple of (dict, list)"),
        (({"value": 1.0}, {}, {}), "a tuple of (dict, dict, dict)"),
        (None, "a NoneType"),
    ],
)
def test_any_other_update_result_is_refused_by_what_it_is(result, got):
    with pytest.raises(TypeError, match=rf"update\(\) returned {re.escape(got)};.*loss_dict_of\(result\)"):
        loss_dict_of(result)


def test_an_update_result_takes_the_installed_releases_shape():
    losses, stats = {"value": 1.0}, ppo_stats(1e-3, torch.tensor([0.5, 1.5]))
    assert stats == {"Optimizer/learning_rate": 1e-3, "Model/mean_policy_std": 1.0}
    assert ppo_stats(1e-3, None) == {"Optimizer/learning_rate": 1e-3}
    result = update_result(losses, stats)
    assert result == ((losses, stats) if runner_unpacks_stats() else losses)
    assert loss_dict_of(result) is losses


def test_an_env_step_result_is_the_rewards_from_6_0_and_none_before():
    rewards = torch.tensor([1.0, 2.0])
    result = env_step_result(rewards)
    if runner_unpacks_stats():
        assert set(result) == {MEAN_REWARD_KEY} and result[MEAN_REWARD_KEY] is rewards
    else:
        assert result is None


def test_a_driven_logger_sums_finished_episodes_and_prints_the_releases_stats(tmp_path):
    """Two steps through rsl_rl's own ``Logger``: envs 0 and 2 finish on the second, env 1 never does."""
    from rsl_rl.utils.logger import Logger

    cfg = {"logger": "tensorboard", "num_steps_per_env": 2, "algorithm": {"rnd_cfg": None}}
    logger = Logger(str(tmp_path), cfg, {}, 3, False, 1, 0, "cpu")
    if hasattr(logger, "init_logging_writer"):  # rsl-rl-lib 3.x opens its writer in __init__
        logger.init_logging_writer()
    rewards = torch.tensor([1.0, 2.0, 3.0])
    log_env_step(logger, rewards, torch.tensor([0, 0, 0]), {})
    log_env_step(logger, rewards, torch.tensor([1, 0, 1]), {})
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        log_iteration(
            logger,
            it=0,
            start_it=0,
            total_it=1,
            collect_time=0.1,
            learn_time=0.1,
            loss_dict={"value": 1.0},
            learning_rate=1e-3,
            policy_std=torch.tensor([0.5, 1.5]),
        )

    episode_rewards, episode_lengths = episode_buffers(logger)
    assert (list(episode_rewards), list(episode_lengths)) == ([2.0, 6.0], [2.0, 2.0])
    text = printed.getvalue()
    assert re.search(r"Mean reward: +4\.00", text) and re.search(r"Mean value loss: +1\.0000", text)
    # rsl-rl-lib 6.0 prints the stats it is given; older releases print the std it reads itself.
    assert re.search(r"Mean policy std: +1\b" if runner_unpacks_stats() else r"Mean action (noise )?std: +1\.00", text)


def test_a_logger_whose_methods_are_wrapped_on_the_instance_still_gets_its_releases_call(tmp_path):
    """Inside ``learn()`` Eden's runner mixins replace ``logger.log`` with a ``*args, **kwargs`` wrapper (and a non-zero
    DDP rank replaces ``process_env_step``), so the helpers read the methods off the logger's class.
    """
    from rsl_rl.utils.logger import Logger

    cfg = {"logger": "tensorboard", "num_steps_per_env": 1, "algorithm": {"rnd_cfg": None}}
    logger = Logger(str(tmp_path), cfg, {}, 2, False, 1, 0, "cpu")
    if hasattr(logger, "init_logging_writer"):  # rsl-rl-lib 3.x opens its writer in __init__
        logger.init_logging_writer()
    calls = []
    for name in ("log", "process_env_step"):
        method = getattr(logger, name)

        def wrapper(*args, _method=method, _name=name, **kwargs):
            calls.append((_name, sorted(kwargs)))
            return _method(*args, **kwargs)

        setattr(logger, name, wrapper)

    log_env_step(logger, torch.ones(2), torch.ones(2), {})
    with contextlib.redirect_stdout(io.StringIO()):
        log_iteration(
            logger,
            it=0,
            start_it=0,
            total_it=1,
            collect_time=0.1,
            learn_time=0.1,
            loss_dict={"value": 1.0},
            learning_rate=1e-3,
            policy_std=None,
        )

    stats_or_std = "stats_dict" if runner_unpacks_stats() else "action_std"
    assert [name for name, _ in calls] == ["process_env_step", "log"] and stats_or_std in calls[1][1]


def test_episode_buffers_read_either_layout_and_name_both_when_neither_is_there():
    old = types.SimpleNamespace(rewbuffer=deque([1.0]), lenbuffer=deque([5.0]))
    new = types.SimpleNamespace(step_buffers={MEAN_REWARD_KEY: deque([2.0])}, length_buffer=deque([6.0]))
    before_any_episode = types.SimpleNamespace(step_buffers={}, length_buffer=deque())
    assert episode_buffers(old) == (old.rewbuffer, old.lenbuffer)
    assert episode_buffers(new) == (new.step_buffers[MEAN_REWARD_KEY], new.length_buffer)
    assert [list(buffer) for buffer in episode_buffers(before_any_episode)] == [[], []]
    with pytest.raises(AttributeError, match=r"SimpleNamespace holds no episode buffers.*'rewbuffer'.*'step_buffers'"):
        episode_buffers(types.SimpleNamespace())


def test_a_6_0_logger_whose_algorithm_reports_no_mean_reward_warns_once_and_reads_no_rewards(eden_caplog):
    """Episodes finished, but the algorithm's ``process_env_step()`` returned no ``Training/mean_reward`` to sum."""
    warn_once.cache_clear()
    logger = types.SimpleNamespace(step_buffers={"Custom/bonus": deque([0.5])}, length_buffer=deque([6.0]))

    for _ in range(2):
        rewards, lengths = episode_buffers(logger)
        assert (list(rewards), list(lengths)) == ([], [6.0])

    warnings = [r.getMessage() for r in eden_caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "sums no 'Training/mean_reward' per episode (it sums 'Custom/bonus')" in warnings[0]
    assert "mean_reward stays null" in warnings[0] and "env_step_result" in warnings[0]


def test_nonfinite_flags_add_up_over_a_rollout_and_raise_once_naming_every_output():
    obs = TensorDict({"policy": torch.zeros(2, 3)}, batch_size=[2])
    dones = torch.zeros(2)
    flags: dict[str, torch.Tensor] = {}
    record_env_step_nonfinite(flags, obs, torch.zeros(2), dones, {})
    raise_if_nonfinite(flags)  # a finite step raises nothing
    raise_if_nonfinite({})  # nor does an empty rollout

    record_env_step_nonfinite(flags, obs, torch.tensor([0.0, float("nan")]), dones, {})
    record_env_step_nonfinite(flags, obs, torch.zeros(2), dones, {})  # a finite step after it keeps the flag
    record_env_step_nonfinite(flags, obs.apply(lambda x: x / 0.0), torch.zeros(2), dones, {})
    with pytest.raises(
        ValueError,
        match=r"^The observation group 'policy', rewards returned by the environment "
        r"contained NaN or Inf values\..*eden smoke --task <task>",
    ):
        raise_if_nonfinite(flags)


def test_a_nonfinite_final_observation_is_flagged_only_for_an_env_that_is_done():
    obs = TensorDict({"policy": torch.zeros(2, 3)}, batch_size=[2])
    final = TensorDict({"policy": torch.tensor([[0.0, 0.0, 0.0], [float("inf"), 0.0, 0.0]])}, batch_size=[2])

    flags: dict[str, torch.Tensor] = {}
    record_env_step_nonfinite(flags, obs, torch.zeros(2), torch.tensor([1, 0]), {FINAL_OBS_KEY: final})
    raise_if_nonfinite(flags)  # env 1 is not done: its final observation is the cache's stale row

    record_env_step_nonfinite(flags, obs, torch.zeros(2), torch.tensor([0, 1]), {FINAL_OBS_KEY: final})
    with pytest.raises(ValueError, match=r"^The final observation group 'policy' returned by the environment"):
        raise_if_nonfinite(flags)


def test_a_group_whose_terms_are_not_concatenated_is_checked_term_by_term():
    """Such a group is a nested TensorDict; ``~torch.isfinite(td).all()`` on it is an ``int``, never a flag."""
    obs = TensorDict({"policy": torch.zeros(2, 3), "critic": {"dofs_pos": torch.zeros(2, 1)}}, batch_size=[2])
    dones = torch.tensor([1, 0])
    final = obs.clone()
    flags: dict[str, torch.Tensor] = {}
    record_env_step_nonfinite(flags, obs, torch.zeros(2), dones, {FINAL_OBS_KEY: final})
    assert all(isinstance(flag, torch.Tensor) for flag in flags.values())
    assert {"observation group 'critic/dofs_pos'", "final observation group 'critic/dofs_pos'"} <= set(flags)
    raise_if_nonfinite(flags)  # finite: nothing to raise

    obs["critic", "dofs_pos"][1] = float("nan")
    final["critic", "dofs_pos"][0] = float("inf")  # env 0 is done, so its final observation counts
    record_env_step_nonfinite(flags, obs, torch.zeros(2), dones, {FINAL_OBS_KEY: final})
    with pytest.raises(
        ValueError,
        match=r"^The observation group 'critic/dofs_pos', final observation group 'critic/dofs_pos' returned",
    ):
        raise_if_nonfinite(flags)


def test_the_stock_nan_check_probe_finds_rsl_rl_6_0_unable_to_read_a_nested_group():
    """6.0.0's ``find_nonfinite`` cannot; 5.x's ``check_nan`` can, and 3.x / 4.x runners check nothing. A release that
    fixes ``find_nonfinite`` turns this red, which is the signal to drop ``StockNanCheckGuardMixin``.
    """
    assert stock_nan_check_reads_nested_groups() == (detect_rsl_rl_version() < (6,))


def test_runner_expects_stats_reads_a_release_without_a_logger_module_as_older_than_6_0(monkeypatch):
    """rsl-rl-lib 3.0 and 3.1 ship no ``rsl_rl.utils.logger``; importing it there used to raise."""
    import importlib.util

    find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util, "find_spec", lambda name, *a: None if name == "rsl_rl.utils.logger" else find_spec(name, *a)
    )
    runner_expects_stats.cache_clear()
    try:
        assert runner_expects_stats() is False
        assert update_result({"value": 1.0}, {}) == {"value": 1.0}
        assert env_step_result(torch.ones(2)) is None
    finally:
        runner_expects_stats.cache_clear()
