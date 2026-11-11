"""What the gymnasium, SB3 and skrl wrappers add on top of :mod:`eden.envs.wrappers.core`.

:class:`VecEnvMixin` turns an :class:`RLEnvBase` into what every such library needs. What the rsl_rl
and rl_games wrappers need too — the build, env ownership, clamped actions, gymnasium spaces read off a
real observation, the time-out split — is :mod:`eden.envs.wrappers.core`'s; the mixin adds what only
these libraries need: final observations, per-episode statistics and copies of the env's reused
buffers. Each wrapper mixes it in beside its library's base class and adds only that library's
interface:

- :class:`~eden.envs.wrappers.gymnasium_env.GymnasiumVecEnvWrapper`: a gymnasium ``VectorEnv`` of torch
  tensors, for CleanRL or a loop of your own;
- :class:`~eden.envs.wrappers.sb3_env.Sb3VecEnvWrapper`: a stable-baselines3 ``VecEnv`` of numpy arrays;
- :class:`~eden.envs.wrappers.skrl_env.SkrlVecEnvWrapper`: an skrl ``Wrapper`` of flattened tensors.
"""

from __future__ import annotations

from collections.abc import Collection
from typing import TYPE_CHECKING, Any, ClassVar, Self

import gymnasium as gym
import numpy as np
import torch

from eden.envs.wrappers.core import EnvWrapperMixin, action_space, observation_space

if TYPE_CHECKING:
    from eden.envs.base import RLEnvBase
    from eden.utils.configs import EdenRLConfig


class VecEnvMixin(EnvWrapperMixin):
    """The library-independent half of a vec-env wrapper around an :class:`RLEnvBase`.

    A wrapper calls :meth:`_init_vec_env` from its ``__init__`` and builds its library's ``reset``,
    ``step`` and ``close`` on :meth:`_reset_env`, :meth:`_step_env` and :meth:`_close_env`; its
    ``from_config`` calls :meth:`_from_config`. The mixin defines no ``__init__``, ``num_envs`` or
    ``device``: those belong to the library's base class (skrl's are read-only properties).

    :meth:`_step_env` returns ``(obs, reward, terminated, truncated, info)``: ``obs`` a dict of the
    observation groups the wrapper reads, the other three ``(num_envs,)`` tensors. ``terminated`` and
    ``truncated`` are the env's termination terms split by ``time_out``; an env can be both, and is then
    terminal. Done envs reset inside the step, so ``obs`` already starts their next episode. ``info``
    holds:

    - ``"final_obs"`` / ``"_final_obs"`` (with ``final_observations``): the observation each done env
      ended its episode on, and the mask of the envs it is valid for — done, and not flagged by Genesis
      as a corrupt state (a solver error, reported as truncated). Rows outside the mask are zero, never
      the corrupt state's NaNs;
    - ``"episode"`` / ``"_episode"``: ``{"r": return, "l": length}`` of each episode that ended this
      step (zero elsewhere), and the done mask — gymnasium's ``RecordEpisodeStatistics`` layout, kept
      on device;
    - ``"log"``: the managers' episode statistics, as the rsl_rl and rl_games wrappers log them.

    Everything :meth:`_step_env` and :meth:`_reset_env` return is a copy: the env reuses its buffers in
    place, and a learner that keeps a step's tensors past the next one (SB3 and skrl both do) must not
    see them change. That is one copy of the groups the wrapper reads per step (two with ``final_obs``);
    the groups it does not read are neither copied nor returned.

    Attributes
    ----------
    env:
        The wrapped env. Wrapping it resets every env, since the spaces are read off a real observation.
    clip_actions:
        Actions are clamped to ``[-clip_actions, clip_actions]`` before the action manager reads them,
        and the action space is bounded by it. ``None`` leaves actions unclamped and the space unbounded.
    final_observations:
        Report ``final_obs`` / ``_final_obs`` in ``info``. The env must then record final observations
        (``env_options.record_final_observations``), which costs one extra observation compute per step
        — a second render for a camera group.
    random_episode_starts:
        On every reset, start each env's episode clock at a random step in ``[0, max_episode_length)``,
        as ``eden train`` does (rsl_rl's ``init_at_random_ep_len``). Without it every env that does not
        terminate early times out on the same step, so episode ends — and a learner's per-episode work —
        arrive in bursts. The first episodes are then shorter; their time-outs are real ``truncated`` ends.
    single_observation_space:
        A :class:`gymnasium.spaces.Dict` with one entry per observation group of the env, read or not:
        a ``Box`` for a group that concatenates its terms, a ``Dict`` of its terms for one that does not.
    single_action_space:
        A ``Box`` of the action manager's width and dtype, bounded by ``clip_actions``.
    """

    #: How to do without final observations, ending the error raised for an env that does not record
    #: them. A wrapper that cannot do without them says why instead.
    _without_final_observations: ClassVar[str] = "Pass `final_observations=False` if nothing reads them."

    final_observations: bool
    random_episode_starts: bool
    single_observation_space: gym.spaces.Dict
    single_action_space: gym.spaces.Box

    def _init_vec_env(
        self,
        env: RLEnvBase,
        *,
        clip_actions: float | None,
        final_observations: bool,
        random_episode_starts: bool,
        observation_groups: tuple[str, ...] | None,
    ) -> None:
        """Validate ``env`` and the options, reset every env, and read the spaces off the observation.

        ``observation_groups`` are the groups the wrapper reads, the only ones :meth:`_reset_env` and
        :meth:`_step_env` return; ``None`` reads them all.
        """
        from eden.envs.base import RLEnvBase

        name = type(self).__name__
        if not isinstance(env, RLEnvBase):
            raise TypeError(
                f"{name} wraps an RLEnvBase, which has rewards and terminations, but got a {type(env).__name__}. "
                "Fix: build the task from an `EdenRLConfig` (`env_class_for_config(cfg)` picks the env class from "
                "the config)."
            )
        if final_observations and not env.env_options.record_final_observations:
            raise ValueError(
                f"{name} reports the observation each done env ended its episode on, but "
                f"'{type(env.config).__name__}' was built with `env_options.record_final_observations=False`, so "
                f"Eden discards it at the auto-reset. Fix: build through `{name}.from_config(cfg)`, which turns it "
                "on, or set `cfg.env_options.record_final_observations = True` before building the env. "
                f"{self._without_final_observations}"
            )
        self._check_clip_actions(clip_actions)
        self.env = env
        self.clip_actions = clip_actions
        self.final_observations = final_observations
        self.random_episode_starts = random_episode_starts
        self._sim_device = torch.device(env.device)

        # Spaces from a real observation: the manager's output is the one source of shapes and dtypes.
        obs, _ = env.reset()
        self.single_observation_space = observation_space(obs)
        for group in observation_groups or ():
            _check_group(group, obs)
        self._observation_groups = tuple(obs) if observation_groups is None else observation_groups
        self._action_dtype = env.action_manager.action.dtype
        self.single_action_space = action_space(env.action_manager.total_action_dim, clip_actions, self._action_dtype)
        self._episode_return = torch.zeros(env.num_envs, device=self._sim_device)
        self._episode_length = torch.zeros(env.num_envs, dtype=torch.long, device=self._sim_device)

    @classmethod
    def _from_config(
        cls,
        cfg: EdenRLConfig,
        *,
        clip_actions: float | None,
        observation_groups: tuple[str, ...] | None,
        record_final_observations: bool,
        seed: int | None,
        show_viewer: bool,
        eval_mode: bool,
        **init_kwargs: Any,
    ) -> Self:
        """Build the env ``cfg`` describes and wrap it as ``cls(env, clip_actions=..., **init_kwargs)``, owning the env.

        ``clip_actions`` and the ``observation_groups`` the wrapper reads (``None``: all of them) are
        checked against ``cfg`` before the build, which can take minutes; the build itself is
        :meth:`~eden.envs.wrappers.core.EnvWrapperMixin._build_and_wrap`'s. The env is built from a copy
        of ``cfg`` whose ``env_options`` (a copy too) set ``record_final_observations``; ``cfg`` itself
        is left as it was. ``seed`` seeds every RNG before the build; ``None`` leaves the process RNG as
        it is (``cfg.runner_options.seed`` belongs to ``eden train``'s runners and is not read here).
        """
        from eden.utils.configs import EdenRLConfig

        if not isinstance(cfg, EdenRLConfig):
            raise TypeError(
                f"'{type(cfg).__name__}' is not an EdenRLConfig, so its env has no rewards or terminations to "
                f"train on. {cls.__name__} needs an RL task; build this one with "
                "`env_class_for_config(cfg).from_config(cfg)` instead."
            )
        cls._check_clip_actions(clip_actions)
        for group in observation_groups or ():
            _check_group(group, cfg.observation_options.term_keys())
        env_options = cfg.env_options.model_copy(update={"record_final_observations": record_final_observations})
        cfg = cfg.model_copy(update={"env_options": env_options})
        return cls._build_and_wrap(
            cfg, seed=seed, show_viewer=show_viewer, eval_mode=eval_mode, clip_actions=clip_actions, **init_kwargs
        )

    def group_space(self, group: str) -> gym.Space:
        """The per-env space of observation group ``group``, raising, naming the env's groups, if there is none."""
        _check_group(group, self.single_observation_space.spaces)
        return self.single_observation_space[group]

    def _reset_env(self, seed: int | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reset every env, seeding the RNGs first with ``seed``; return the observation and the reset log.

        With ``random_episode_starts``, each env's episode clock then starts at a random step, drawn
        after the seeding, so it is reproducible.
        """
        obs, extras = self.env.reset(seed=seed)
        if self.random_episode_starts:
            clock = self.env.episode_length_buf
            clock.copy_(torch.randint_like(clock, high=int(self.env.max_episode_length)))
            # Re-read (compute() is a read, not a tick), so a term that reads the clock sees the new one.
            obs = self.env.observation_manager.compute()
        self._episode_return.zero_()
        self._episode_length.zero_()
        return self._copy_read_groups(obs), extras["log"]

    def _step_env(
        self, actions: torch.Tensor | np.ndarray
    ) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Clamp ``actions`` (a tensor or numpy array), step every env, and return copies (see the class docstring)."""
        actions = self._clamp_actions(torch.as_tensor(actions, dtype=self._action_dtype, device=self._sim_device))
        obs, reward, terminated, truncated, extras = self.env.step(actions)
        done = terminated | truncated

        self._episode_return += reward
        self._episode_length += 1
        info = {}
        if self.final_observations:
            final_obs_valid = done & ~self.env.termination_manager.solver_errors
            final_obs = extras["final_observations"]
            info["final_obs"] = {
                group: _rows_where(final_obs_valid, final_obs[group]) for group in self._observation_groups
            }
            info["_final_obs"] = final_obs_valid
        info |= {
            "episode": {
                "r": torch.where(done, self._episode_return, 0.0),
                "l": torch.where(done, self._episode_length, 0),
            },
            "_episode": done,
            "log": extras["log"],
        }
        self._episode_return.masked_fill_(done, 0.0)
        self._episode_length.masked_fill_(done, 0)
        return self._copy_read_groups(obs), reward.clone(), terminated.clone(), truncated.clone(), info

    def _copy_read_groups(self, obs: dict[str, Any]) -> dict[str, Any]:
        """A copy of the groups the wrapper reads, detached from the env's reused buffers."""
        return {group: _copy(obs[group]) for group in self._observation_groups}


def _check_group(group: str, groups: Collection[str]) -> None:
    """Raise, naming ``groups``, unless observation group ``group`` is one of them."""
    if group not in groups:
        raise KeyError(
            f"Observation group '{group}' is not one of the task's observation groups {sorted(groups)}. Fix: pass one of "
            f"those, or add an `ObservationGroupOptions` named '{group}' to the task's `observation_options`."
        )


def _copy(value: torch.Tensor | dict[str, Any]) -> torch.Tensor | dict[str, Any]:
    """A copy of an observation (group, or dict of groups/terms), detached from the env's buffers."""
    if isinstance(value, dict):
        return {key: _copy(item) for key, item in value.items()}
    return value.clone()


def _rows_where(mask: torch.Tensor, value: torch.Tensor | dict[str, Any]) -> torch.Tensor | dict[str, Any]:
    """A new observation (tree) holding ``value``'s rows where ``mask`` is set and zeros elsewhere."""
    if isinstance(value, dict):
        return {key: _rows_where(mask, item) for key, item in value.items()}
    rows = mask.view(mask.shape[0], *([1] * (value.ndim - 1)))
    return torch.where(rows, value, torch.zeros((), dtype=value.dtype, device=value.device))
