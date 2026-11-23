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
``super().process_env_step()`` returned), so it never needs the version. An algorithm or runner written from scratch
builds the release's shapes with :func:`update_result`, :func:`env_step_result`, :func:`log_env_step` and
:func:`log_iteration`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, TypeAlias

import torch

from eden.options.learning.rsl_rl.translate import detect_rsl_rl_version

#: What an rsl-rl-lib algorithm's ``update()`` returns: the loss dict (< 6.0), or ``(loss_dict, stats_dict)`` (>= 6.0).
UpdateResult: TypeAlias = dict[str, float] | tuple[dict[str, float], dict[str, float]]

#: The ``step_dict`` key rsl-rl-lib >= 6.0's ``Logger`` reports as the mean episode reward.
MEAN_REWARD_KEY = "Training/mean_reward"


def has_stats_api() -> bool:
    """Whether the installed rsl-rl-lib is 6.0 or newer: ``update()`` returns stats and ``process_env_step()`` a dict."""
    return detect_rsl_rl_version() >= (6,)


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
    return (loss_dict, stats_dict) if has_stats_api() else loss_dict


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
    return {MEAN_REWARD_KEY: rewards} if has_stats_api() else None


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
    if has_stats_api():
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
    if has_stats_api():
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


def raise_if_env_step_nonfinite(obs: Any, rewards: torch.Tensor, dones: torch.Tensor, extras: dict) -> None:
    """Raise a ``ValueError`` naming the outputs of an env step that hold NaN or Inf, as rsl_rl's ``check_for_nan`` does.

    Uses the installed release's own check: ``find_nonfinite`` + ``raise_if_nonfinite`` from 6.0, ``check_nan`` in
    5.x. Releases before ``check_nan`` have no check of their own, and get the same one written here.

    Parameters
    ----------
    obs : TensorDict
        The step's observations.
    rewards, dones : torch.Tensor
        The step's rewards and dones.
    extras : dict
        The step's extras; rsl-rl-lib >= 6.0 also checks the done envs' ``"final_obs"`` in it.
    """
    import rsl_rl.utils as rsl_rl_utils

    if hasattr(rsl_rl_utils, "find_nonfinite"):
        rsl_rl_utils.raise_if_nonfinite(rsl_rl_utils.find_nonfinite(obs, rewards, dones, extras))
    elif hasattr(rsl_rl_utils, "check_nan"):
        rsl_rl_utils.check_nan(obs, rewards, dones)
    else:
        outputs = {f"observation group '{key}'": value for key, value in obs.items()}
        outputs |= {"rewards": rewards, "dones": dones}
        nonfinite = [name for name, value in outputs.items() if not torch.isfinite(value).all()]
        if nonfinite:
            raise ValueError(
                f"The {', '.join(nonfinite)} returned by the environment contain NaN or Inf values. This usually "
                "indicates a bug in the environment's step() or reset() function."
            )


def ensure_logger_writer(logger: Any) -> None:
    """Give an rsl-rl-lib 4.x ``Logger`` the ``writer = None`` that every other release sets in its ``__init__``.

    rsl-rl-lib 4.x creates ``Logger.writer`` only in ``init_logging_writer()``, which ``learn()`` calls, while its
    ``OnPolicyRunner.save()`` reads it to decide on an upload. So a save before ``learn()`` (a script writing out an
    initial checkpoint) raises ``AttributeError`` there and nowhere else. ``None`` is the value 5.x and 6.x start
    with: no writer yet, so nothing to upload. ``init_logging_writer()`` replaces it as usual.

    Parameters
    ----------
    logger : Any
        The runner's logger; one that already has a ``writer`` is left alone.
    """
    if not hasattr(logger, "writer"):
        logger.writer = None


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
        return logger.step_buffers.get(MEAN_REWARD_KEY, ()), logger.length_buffer
    raise AttributeError(
        f"{type(logger).__module__}.{type(logger).__qualname__} holds no episode buffers: neither 'rewbuffer' / "
        "'lenbuffer' (rsl-rl-lib < 6.0) nor 'step_buffers' / 'length_buffer' (>= 6.0). Fix: a runner with its own "
        "logger gives it 'rewbuffer' and 'lenbuffer' deques of finished-episode rewards and lengths, as "
        "eden.learning.wrappers.rsl_rl.supermimic.logger.RunLogger does."
    )
