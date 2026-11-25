"""The algorithm, runner and logger contracts that rsl-rl-lib 6.0 changed, behind one interface for 3.x to 6.x.

An algorithm, runner or runner mixin that Eden writes against rsl_rl goes through these helpers instead of calling the
changed API directly, so it runs under every rsl-rl-lib that Eden's ``rsl-rl-lib>=3.0`` pin admits. What 6.0 changed:

* ``update()`` returns ``(loss_dict, stats_dict)``. Before 6.0 it returned the loss dict, and the runner read the
  learning rate and the policy std off the algorithm itself; 6.0's runner logs whatever ``stats_dict`` holds, under the
  tags its keys spell (``Optimizer/learning_rate``, ``Model/mean_policy_std``).
* ``process_env_step()`` returns a dict of per-env values (``{"Training/mean_reward": rewards}`` at least) that the
  ``Logger`` sums over each episode. Before 6.0 it returned ``None``, and the runner handed the rewards to the ``Logger``.
* The ``Logger`` takes ``process_env_step(dones, extras, step_dict)`` and ``log(..., loss_dict, stats_dict)``, and keeps
  its episode sums in ``step_buffers["Training/mean_reward"]`` / ``length_buffer`` instead of ``rewbuffer`` /
  ``lenbuffer``. ``rsl_rl.utils.check_nan`` became ``find_nonfinite`` + ``raise_if_nonfinite``.

A PPO subclass that adds metrics passes its parent's results through (:func:`loss_dict_of`, and returning what
``super().process_env_step()`` returned). An algorithm or runner written from scratch builds the release's shapes with
:func:`update_result`, :func:`env_step_result`, :func:`log_env_step` and :func:`log_iteration`.

**How a helper tells the releases apart.** It inspects what it is handed, never the version number: the shape of the
parent's ``update()`` result, the parameters of the logger's methods, the names ``rsl_rl.utils`` exports. This is the
rule :func:`~eden.envs.wrappers.rsl_rl_env._runner_uses_split_policy` follows, for the same reason: a source checkout
without package metadata reads as version 3. :func:`update_result` and :func:`env_step_result` are handed nothing that
differs between releases, so they inspect the class the stock runner hands their results to,
``rsl_rl.utils.logger.Logger`` (:func:`runner_expects_stats`). A logger's methods are read off its class: a runner
mixin may replace ``logger.log`` on the instance with a ``*args, **kwargs`` wrapper while ``learn()`` runs.
"""

from __future__ import annotations

import functools
import importlib.util
import inspect
from collections.abc import Iterator, Sequence
from typing import Any, TypeAlias

import torch

from eden.utils.warning import warn_once

#: What an rsl-rl-lib algorithm's ``update()`` returns: the loss dict (< 6.0), or ``(loss_dict, stats_dict)`` (>= 6.0).
UpdateResult: TypeAlias = dict[str, float] | tuple[dict[str, float], dict[str, float]]

#: The ``step_dict`` key rsl-rl-lib >= 6.0's ``Logger`` reports as the mean episode reward.
MEAN_REWARD_KEY = "Training/mean_reward"

#: The ``extras`` key rsl-rl-lib >= 6.0 reads a done env's last observation from (its NaN check, SAC's bootstrap). Eden's
#: env publishes it as ``extras["final_observations"]``; :class:`~eden.envs.wrappers.rsl_rl_env.RslRlVecEnvWrapper`
#: adds this key for rsl_rl.
FINAL_OBS_KEY = "final_obs"


@functools.cache
def _takes(cls: type, method: str, parameter: str) -> bool:
    """Whether ``cls.method`` declares a parameter named ``parameter``.

    Cached: :func:`log_env_step` asks at every env step, and a class's signature does not change within a run.
    """
    return parameter in inspect.signature(getattr(cls, method)).parameters


@functools.cache
def runner_expects_stats() -> bool:
    """Whether the installed rsl_rl runner takes ``(loss_dict, stats_dict)`` from ``update()`` (rsl-rl-lib >= 6.0).

    The same runner then also takes a dict from ``process_env_step()``. Read off the ``Logger`` it hands both to: its
    ``log()`` takes ``stats_dict`` from 6.0 on, and ``learning_rate`` / ``action_std`` before. rsl-rl-lib 3.0 and 3.1
    have no ``rsl_rl.utils.logger`` module (their runner logs by itself), so a release without one is older than 6.0.
    Cached: :func:`env_step_result` asks at every env step, and the installed release does not change within a run.
    """
    if importlib.util.find_spec("rsl_rl.utils.logger") is None:
        return False
    from rsl_rl.utils.logger import Logger

    return _takes(Logger, "log", "stats_dict")


def loss_dict_of(result: UpdateResult) -> dict[str, float]:
    """The loss dict in an ``update()`` result, as the same object, so a key written into it reaches the runner.

    A PPO subclass writes its extra metrics into ``loss_dict_of(super().update())`` and returns that result object,
    which keeps the installed release's return shape without reading the version.

    Parameters
    ----------
    result : UpdateResult
        What the parent algorithm's ``update()`` returned.

    Returns
    -------
    dict[str, float]
        ``result`` itself when it is a dict, else its first element.

    Raises
    ------
    TypeError
        If ``result`` is neither a dict nor a pair of dicts.
    """
    if isinstance(result, dict):
        return result
    if isinstance(result, tuple) and len(result) == 2 and all(isinstance(part, dict) for part in result):
        return result[0]
    if isinstance(result, tuple):
        got = f"a tuple of ({', '.join(type(part).__name__ for part in result)})"
    else:
        got = f"a {type(result).__name__}"
    raise TypeError(
        f"update() returned {got}; rsl-rl-lib's algorithms return a loss dict (< 6.0) or a (loss_dict, stats_dict) "
        "pair of dicts (>= 6.0). Fix: make each update() override between rsl_rl's and this one return its parent's "
        "result object, with its own metrics written into loss_dict_of(result); rebuilding the result drops the "
        "release's shape."
    )


def update_result(loss_dict: dict[str, float], stats_dict: dict[str, float]) -> UpdateResult:
    """What an algorithm's ``update()`` returns under the installed rsl-rl-lib: ``loss_dict``, or both dicts on 6.0.

    Parameters
    ----------
    loss_dict : dict[str, float]
        Mean losses, logged under ``Loss/``.
    stats_dict : dict[str, float]
        Diagnostics keyed by their full tag (``Optimizer/learning_rate``). Only rsl-rl-lib >= 6.0 reads it; before
        that the runner reads the learning rate and the policy std off the algorithm.

    Returns
    -------
    UpdateResult
        ``(loss_dict, stats_dict)`` on rsl-rl-lib >= 6.0, else ``loss_dict``.
    """
    return (loss_dict, stats_dict) if runner_expects_stats() else loss_dict


def ppo_stats(learning_rate: float, policy_std: torch.Tensor | None) -> dict[str, float]:
    """The ``stats_dict`` rsl-rl-lib 6.0's PPO reports: the learning rate and, for a stochastic policy, its mean std.

    Parameters
    ----------
    learning_rate : float
        The current learning rate.
    policy_std : torch.Tensor or None
        The policy's action std, of any shape; ``None`` for a deterministic policy.

    Returns
    -------
    dict[str, float]
        ``Optimizer/learning_rate``, plus ``Model/mean_policy_std`` when ``policy_std`` is given.
    """
    stats = {"Optimizer/learning_rate": float(learning_rate)}
    if policy_std is not None:
        stats["Model/mean_policy_std"] = policy_std.mean().item()
    return stats


def env_step_result(rewards: torch.Tensor) -> dict[str, torch.Tensor] | None:
    """What an algorithm's ``process_env_step()`` returns: ``{"Training/mean_reward": rewards}`` on 6.0, else ``None``.

    Parameters
    ----------
    rewards : torch.Tensor
        The env's rewards for the step, shape ``(num_envs,)``, before any bootstrapping.

    Returns
    -------
    dict[str, torch.Tensor] or None
        The per-env values rsl-rl-lib >= 6.0's ``Logger`` sums per episode, or ``None`` for an older runner.
    """
    return {MEAN_REWARD_KEY: rewards} if runner_expects_stats() else None


def log_env_step(logger: Any, rewards: torch.Tensor, dones: torch.Tensor, extras: dict) -> None:
    """Record one env step in an rsl_rl ``Logger``, for a runner that drives the logger itself.

    Parameters
    ----------
    logger : rsl_rl.utils.logger.Logger
        The runner's logger.
    rewards, dones : torch.Tensor
        The step's rewards and dones, shape ``(num_envs,)``.
    extras : dict
        The env's extras for the step (its ``"log"`` / ``"episode"`` entries are recorded).
    """
    if _takes(type(logger), "process_env_step", "step_dict"):
        logger.process_env_step(dones, extras, {MEAN_REWARD_KEY: rewards})
    else:
        logger.process_env_step(rewards, dones, extras, None)


def log_iteration(
    logger: Any,
    *,
    it: int,
    start_it: int,
    total_it: int,
    collect_time: float,
    learn_time: float,
    loss_dict: dict[str, float],
    learning_rate: float,
    policy_std: torch.Tensor | None,
) -> None:
    """Log and print one learning iteration through an rsl_rl ``Logger``, for a runner that drives the logger itself.

    Parameters
    ----------
    logger : rsl_rl.utils.logger.Logger
        The runner's logger.
    it, start_it, total_it : int
        The iteration, the first of this ``learn()`` call, and the last.
    collect_time, learn_time : float
        Seconds spent on the rollout and on the update.
    loss_dict : dict[str, float]
        Mean losses, logged under ``Loss/``.
    learning_rate : float
        The current learning rate.
    policy_std : torch.Tensor or None
        The policy's action std; ``None`` for a deterministic policy, which logs no std on 6.0 and a std of 0 before.
    """
    timing = {"it": it, "start_it": start_it, "total_it": total_it, "collect_time": collect_time}
    if _takes(type(logger), "log", "stats_dict"):
        stats = ppo_stats(learning_rate, policy_std)
        logger.log(**timing, learn_time=learn_time, loss_dict=loss_dict, stats_dict=stats)
    else:
        std = policy_std if policy_std is not None else torch.zeros(1)
        logger.log(
            **timing,
            learn_time=learn_time,
            loss_dict=loss_dict,
            learning_rate=learning_rate,
            action_std=std,
            rnd_weight=None,
        )


def _terms(groups: Any, prefix: str = "") -> Iterator[tuple[str, torch.Tensor]]:
    """``(path, tensor)`` for each tensor in an observation TensorDict, through groups whose terms are not concatenated.

    ``torch.isfinite(td).all()`` on a nested TensorDict returns a Python ``bool``, which ``~`` turns into an ``int``
    (``~True == -2``), so a nested group is read term by term.
    """
    for key, value in groups.items():
        path = f"{prefix}{key}"
        if isinstance(value, torch.Tensor):
            yield path, value
        else:
            yield from _terms(value, f"{path}/")


@functools.cache
def stock_nan_check_reads_nested_groups() -> bool:
    """Whether the installed rsl_rl runner's NaN check reads an observation group whose terms are not concatenated.

    Such a group (``ObservationGroupOptions.concatenate_terms=False``) reaches rsl_rl as a nested TensorDict.
    rsl-rl-lib 6.0's ``find_nonfinite`` reduces it to a Python ``bool`` and inverts that into an ``int``, so its
    ``raise_if_nonfinite`` fails with ``AttributeError: 'int' object has no attribute 'to'`` at the end of the first
    rollout. 5.x's ``check_nan`` reads such a group, and the 3.x and 4.x runners make no check. Probed by running the
    release's own ``find_nonfinite`` on a one-env nested TensorDict, so a release that fixes it reads as fixed.
    """
    import rsl_rl.utils as rsl_rl_utils
    from tensordict import TensorDict

    if not hasattr(rsl_rl_utils, "find_nonfinite"):
        return True
    nested = TensorDict({"group": {"term": torch.zeros(1, 1)}}, batch_size=[1])
    flags = rsl_rl_utils.find_nonfinite(nested, torch.zeros(1), torch.zeros(1), None)
    return all(isinstance(flag, torch.Tensor) for flag in flags.values())


def record_env_step_nonfinite(
    flags: dict[str, torch.Tensor], obs: Any, rewards: torch.Tensor, dones: torch.Tensor, extras: dict
) -> None:
    """OR one env step's NaN / Inf flags into ``flags``, without a host sync; :func:`raise_if_nonfinite` reads them.

    The check rsl-rl-lib 6.0's runner makes (``rsl_rl.utils.find_nonfinite``), written here so a runner that drives
    its own loop gets it under every release: one flag per observation group, the rewards, the dones, and each group
    of ``extras["final_obs"]`` over the envs that are done. A group whose terms are not concatenated
    (``concatenate_terms=False``) is a nested TensorDict, and gets one flag per term (``'critic/dofs_pos'``); 6.0's own
    check cannot read it (:func:`stock_nan_check_reads_nested_groups`). Like 6.0's runner, record every step of a
    rollout into one dict and raise once after it, which costs one host sync per rollout instead of one per step.

    Parameters
    ----------
    flags : dict[str, torch.Tensor]
        The rollout's flags so far, keyed by output name; updated in place.
    obs : TensorDict
        The step's observations.
    rewards, dones : torch.Tensor
        The step's rewards and dones.
    extras : dict
        The step's extras. :class:`~eden.envs.wrappers.rsl_rl_env.RslRlVecEnvWrapper` publishes the done envs' last
        observations under ``"final_obs"`` when the env records them (``EnvOptions.record_final_observations``).
    """
    step = {f"observation group '{path}'": ~torch.isfinite(term).all() for path, term in _terms(obs)}
    step["rewards"] = ~torch.isfinite(rewards).all()
    step["dones"] = ~torch.isfinite(dones).all()
    final_obs = extras.get(FINAL_OBS_KEY)
    if final_obs is not None:
        done = dones.reshape(-1).bool()
        for path, term in _terms(final_obs):
            mask = done.to(term.device).reshape(-1, *([1] * (term.ndim - 1)))
            step[f"final observation group '{path}'"] = ~(torch.isfinite(term) | ~mask).all()
    for source, flag in step.items():
        flags[source] = flags[source] | flag if source in flags else flag


def raise_if_nonfinite(flags: dict[str, torch.Tensor]) -> None:
    """Raise a ``ValueError`` naming every output :func:`record_env_step_nonfinite` flagged, as rsl-rl-lib 6.0 does.

    Parameters
    ----------
    flags : dict[str, torch.Tensor]
        The rollout's flags, keyed by output name.

    Raises
    ------
    ValueError
        If any flag is set.
    """
    if not flags:
        return
    device = next(iter(flags.values())).device
    values = torch.stack([flag.to(device) for flag in flags.values()]).tolist()
    sources = [source for source, value in zip(flags, values) if value]
    if sources:
        raise ValueError(
            f"The {', '.join(sources)} returned by the environment contained NaN or Inf values. This usually indicates "
            "a bug in the environment's step() or reset() function, such as a reward or observation term. Fix: find "
            "the term with `eden smoke --task <task>`; setting runner_options.check_for_nan=False only hides it."
        )


def episode_buffers(logger: Any) -> tuple[Sequence[float], Sequence[float]]:
    """The last completed episodes' rewards and lengths an rsl_rl ``Logger`` holds (up to 100 each, oldest first).

    Read by structure: ``rewbuffer`` / ``lenbuffer`` before rsl-rl-lib 6.0 (and on an Eden runner's own logger that
    keeps them), ``step_buffers["Training/mean_reward"]`` / ``length_buffer`` from 6.0. Read them after ``Logger.log``
    has run for the iteration: 6.0 moves the finished episodes into its buffers there.

    Parameters
    ----------
    logger : Any
        The runner's logger.

    Returns
    -------
    tuple[Sequence[float], Sequence[float]]
        The episode rewards and the episode lengths; empty before the first episode ends.

    Raises
    ------
    AttributeError
        If ``logger`` holds neither layout.
    """
    if hasattr(logger, "rewbuffer") and hasattr(logger, "lenbuffer"):
        return logger.rewbuffer, logger.lenbuffer
    if hasattr(logger, "step_buffers") and hasattr(logger, "length_buffer"):
        rewards = logger.step_buffers.get(MEAN_REWARD_KEY)
        if rewards is None and (logger.step_buffers or logger.length_buffer):
            summed = ", ".join(map(repr, logger.step_buffers)) or "nothing"
            warn_once(
                f"The rsl_rl Logger has recorded env steps but sums no {MEAN_REWARD_KEY!r} per episode (it sums "
                f"{summed}), so the mean episode reward reads as none: progress.json's mean_reward stays null. "
                "rsl-rl-lib >= 6.0's Logger sums only what the algorithm's process_env_step() returns. Fix: return the "
                f"env's rewards under {MEAN_REWARD_KEY!r} from process_env_step(), as rsl_rl's PPO does "
                "(eden.envs.wrappers.rsl_rl_compat.env_step_result builds that dict for an algorithm written from "
                "scratch)."
            )
        return (rewards if rewards is not None else ()), logger.length_buffer
    raise AttributeError(
        f"{type(logger).__module__}.{type(logger).__qualname__} holds no episode buffers: neither 'rewbuffer' / "
        "'lenbuffer' (rsl-rl-lib < 6.0) nor 'step_buffers' / 'length_buffer' (>= 6.0). Fix: a runner with its own "
        "logger gives it 'rewbuffer' and 'lenbuffer' deques of finished-episode rewards and lengths, as "
        "eden.learning.wrappers.rsl_rl.supermimic.logger.RunLogger does."
    )
