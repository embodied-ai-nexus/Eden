"""skrl environment wrapper for Eden.

Presents an :class:`RLEnvBase` as an skrl (torch) :class:`~skrl.envs.wrappers.torch.Wrapper`:
observations, rewards and dones stay torch tensors on the env's device, so an skrl agent trains without
a host round-trip. The shared core (spaces, action clamping, episode statistics) is
:class:`~eden.envs.wrappers.vec_env_mixin.VecEnvMixin`.

Usage::

    from skrl.agents.torch.ppo import PPO
    from skrl.trainers.torch import SequentialTrainer
    from eden.envs.wrappers.skrl_env import SkrlVecEnvWrapper

    env = SkrlVecEnvWrapper.from_config(cfg, state_group="critic")
    agent = PPO(models=..., memory=..., observation_space=env.observation_space,
                state_space=env.state_space, action_space=env.action_space, device=env.device)
    SequentialTrainer(env=env, agents=agent, cfg={"timesteps": 10_000}).train()
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from skrl.envs.wrappers.torch.base import Wrapper
from skrl.utils.spaces.torch import flatten_tensorized_space, tensorize_space

from eden.envs.wrappers.core import time_outs
from eden.envs.wrappers.vec_env_mixin import VecEnvMixin

if TYPE_CHECKING:
    import gymnasium as gym

    from eden.envs.base import RLEnvBase
    from eden.utils.configs import EdenRLConfig


class SkrlVecEnvWrapper(VecEnvMixin, Wrapper):
    """An :class:`RLEnvBase` as skrl's single-agent torch :class:`Wrapper`.

    The policy observes ``obs_group``; ``state_group``, when given, is the state skrl hands an
    asymmetric critic (:meth:`state`, :attr:`state_space`). Both are flattened to ``(num_envs, D)``
    as skrl's models expect. Rewards and dones are ``(num_envs, 1)``, and ``truncated`` excludes an env
    that also terminated (it is terminal). skrl values a time-out from the next observation ``step``
    returns, which in Eden (as in Isaac Lab) already starts the next episode — with
    ``time_limit_bootstrap`` on or off — so skrl's agents never read ``info["final_obs"]``, and
    ``final_observations`` defaults to off here. ``info`` is :class:`VecEnvMixin`'s, except that every
    ``log`` value — the managers' episode statistics — is a 0-d tensor. Most already are; a curriculum
    logs a plain float, which skrl's trainers would skip, since they record only one-element tensors.
    With ``environment_info: "log"`` in the trainer config, all of them reach skrl's TensorBoard.
    :meth:`close` destroys the env only when :meth:`from_config` built it.

    skrl's :class:`Wrapper` answers any attribute this class lacks from the wrapped env, so
    ``wrapper.episode_length_buf`` reads the :class:`RLEnvBase`'s, where the gymnasium and SB3 wrappers
    raise ``AttributeError``. Reach the env through :attr:`env` to keep code portable between them.

    Parameters
    ----------
    env:
        The Eden RL environment, already built.
    obs_group:
        The observation group the policy observes.
    state_group:
        The observation group the critic reads as skrl's state, or ``None`` for no separate state.
    clip_actions, final_observations, random_episode_starts:
        See :class:`~eden.envs.wrappers.vec_env_mixin.VecEnvMixin`.
    """

    def __init__(
        self,
        env: RLEnvBase,
        obs_group: str = "policy",
        state_group: str | None = None,
        *,
        clip_actions: float | None = None,
        final_observations: bool = False,
        random_episode_starts: bool = False,
    ):
        # skrl's Wrapper first: it routes unknown attribute lookups to the env it stores.
        super().__init__(env)
        self._init_vec_env(
            env,
            clip_actions=clip_actions,
            final_observations=final_observations,
            random_episode_starts=random_episode_starts,
            observation_groups=_read_groups(obs_group, state_group),
        )
        self.obs_group = obs_group
        self.state_group = state_group
        self._observation_space = self.group_space(obs_group)
        self._state_space = self.group_space(state_group) if state_group is not None else None
        self._state: torch.Tensor | None = None

    @classmethod
    def from_config(
        cls,
        cfg: EdenRLConfig,
        *,
        obs_group: str = "policy",
        state_group: str | None = None,
        clip_actions: float | None = None,
        final_observations: bool = False,
        random_episode_starts: bool = False,
        seed: int | None = None,
        show_viewer: bool = False,
        eval_mode: bool = False,
    ) -> SkrlVecEnvWrapper:
        """Construct and **build** the env ``cfg`` (any :class:`EdenRLConfig`) describes, and wrap it.

        The env records final observations as ``final_observations`` says, on a copy of ``cfg``;
        ``seed`` seeds every RNG before the build (``None`` leaves them alone); ``show_viewer`` and
        ``eval_mode`` go to the env's ``from_config``. :meth:`close` destroys the env.
        """
        return cls._from_config(
            cfg,
            clip_actions=clip_actions,
            observation_groups=_read_groups(obs_group, state_group),
            record_final_observations=final_observations,
            seed=seed,
            show_viewer=show_viewer,
            eval_mode=eval_mode,
            obs_group=obs_group,
            state_group=state_group,
            final_observations=final_observations,
            random_episode_starts=random_episode_starts,
        )

    @property
    def observation_space(self) -> gym.Space:
        return self._observation_space

    @property
    def state_space(self) -> gym.Space | None:
        return self._state_space

    @property
    def action_space(self) -> gym.Space:
        return self.single_action_space

    def reset(self) -> tuple[torch.Tensor, dict[str, Any]]:
        obs, log = self._reset_env()
        return self._split(obs), {"log": log}

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Any]:
        with torch.no_grad():
            obs, reward, terminated, truncated, info = self._step_env(actions)
        info = {**info, "log": {key: torch.as_tensor(value) for key, value in info["log"].items()}}
        # An env that both terminated and timed out is terminal: skrl must not bootstrap it as a time-out.
        time_out = time_outs(terminated, truncated)
        return self._split(obs), reward.view(-1, 1), terminated.view(-1, 1), time_out.view(-1, 1), info

    def state(self) -> torch.Tensor | None:
        return self._state

    def render(self, *args, **kwargs) -> None:
        return None

    def close(self) -> None:
        self._close_env()

    def _split(self, obs: dict[str, Any]) -> torch.Tensor:
        """The policy's flattened observation; keeps the critic's state for :meth:`state`."""
        if self.state_group is not None:
            self._state = flatten_tensorized_space(tensorize_space(self._state_space, obs[self.state_group]))
        return flatten_tensorized_space(tensorize_space(self._observation_space, obs[self.obs_group]))


def _read_groups(obs_group: str, state_group: str | None) -> tuple[str, ...]:
    """The observation groups the wrapper reads: the policy's, and the critic's state when there is one."""
    return (obs_group,) if state_group is None else (obs_group, state_group)
