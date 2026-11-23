"""The rsl-rl-lib 6.0 contract helpers, under whichever rsl-rl-lib is installed (3.x to 6.x)."""

from __future__ import annotations

import contextlib
import io
import re
import types
from collections import deque

import pytest
import torch
from tensordict import TensorDict

from eden.envs.wrappers.rsl_rl_compat import (
    MEAN_REWARD_KEY,
    env_step_result,
    episode_buffers,
    log_env_step,
    log_iteration,
    loss_dict_of,
    ppo_stats,
    raise_if_env_step_nonfinite,
    runner_expects_stats,
    update_result,
)
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


def test_a_nonfinite_env_step_raises_rsl_rls_own_error_naming_the_output():
    obs = TensorDict({"policy": torch.zeros(2, 3)}, batch_size=[2])
    dones = torch.zeros(2)
    raise_if_env_step_nonfinite(obs, torch.zeros(2), dones, {})
    with pytest.raises(ValueError, match=r"The rewards returned by the environment contain(ed)? NaN"):
        raise_if_env_step_nonfinite(obs, torch.tensor([0.0, float("nan")]), dones, {})
    with pytest.raises(ValueError, match=r"observation group 'policy'"):
        raise_if_env_step_nonfinite(obs.apply(lambda x: x / 0.0), torch.zeros(2), dones, {})
