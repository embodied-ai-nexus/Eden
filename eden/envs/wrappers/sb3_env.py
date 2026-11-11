"""stable-baselines3 environment wrapper for Eden.

Presents an :class:`RLEnvBase` as an SB3 :class:`VecEnv`: one observation group, as numpy arrays on
the CPU, which is what every SB3 algorithm consumes. The shared core (spaces, action clamping, final
observations, episode statistics) is :class:`~eden.envs.wrappers.vec_env_mixin.VecEnvMixin`.

Usage::

    from stable_baselines3 import PPO
    from eden.envs.wrappers.sb3_env import Sb3VecEnvWrapper

    env = Sb3VecEnvWrapper.from_config(cfg, clip_actions=1.0)
    model = PPO("MlpPolicy", env, n_steps=24, batch_size=env.num_envs * 6)
    model.learn(total_timesteps=1_000_000)

SB3 is single-device and numpy-based: every step copies the observation, reward and done to the CPU,
so expect a fraction of the throughput of the on-device rsl_rl and rl_games runners.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from stable_baselines3.common.vec_env import VecEnv

from eden.envs.wrappers.core import time_outs
from eden.envs.wrappers.vec_env_mixin import VecEnvMixin

if TYPE_CHECKING:
    import gymnasium as gym
    from stable_baselines3.common.vec_env.base_vec_env import VecEnvIndices, VecEnvObs, VecEnvStepReturn

    from eden.envs.base import RLEnvBase
    from eden.utils.configs import EdenRLConfig


class Sb3VecEnvWrapper(VecEnvMixin, VecEnv):
    """An :class:`RLEnvBase` as a stable-baselines3 :class:`VecEnv`.

    The policy observes one observation group, ``obs_group``: a ``Box`` group trains with
    ``"MlpPolicy"``, a group that keeps its terms apart (``concatenate_terms=False``) is a ``Dict``
    and needs ``"MultiInputPolicy"``. Every done env's info carries what SB3 reads at an episode end:
    ``"terminal_observation"`` (for time-out bootstrapping and off-policy replay),
    ``"TimeLimit.truncated"`` and ``"episode"`` (``{"r": return, "l": length}``, which SB3 logs as
    ``rollout/ep_rew_mean`` — no ``VecMonitor`` needed). An env Genesis flagged as corrupt (a solver
    error) gets no ``terminal_observation``: its state is not one to value, so SB3 does not bootstrap it.

    So ``env`` must record final observations (:meth:`from_config` turns that on), and, since every
    SB3 algorithm asserts finite action bounds, actions are clipped to ``clip_actions``; SB3 clips each
    sampled action to that bound before stepping. :meth:`close` destroys the env only when
    :meth:`from_config` built it.

    Parameters
    ----------
    env:
        The Eden RL environment, already built, with ``env_options.record_final_observations``.
    clip_actions, random_episode_starts:
        See :class:`~eden.envs.wrappers.vec_env_mixin.VecEnvMixin`; ``clip_actions`` is required.
    obs_group:
        The observation group the policy observes.
    """

    _without_final_observations = (
        "SB3 cannot do without them: PPO bootstraps time-outs from each done env's `terminal_observation`, and "
        "the off-policy algorithms store it as the transition's next observation. Do NOT drop the key: SB3 would "
        "silently replay a done env's reset observation as its next observation."
    )

    def __init__(
        self,
        env: RLEnvBase,
        *,
        clip_actions: float,
        obs_group: str = "policy",
        random_episode_starts: bool = False,
    ):
        self._init_vec_env(
            env,
            clip_actions=clip_actions,
            final_observations=True,
            random_episode_starts=random_episode_starts,
            observation_groups=(obs_group,),
        )
        self.obs_group = obs_group
        self._actions: np.ndarray | None = None
        super().__init__(env.num_envs, self.group_space(obs_group), self.single_action_space)

    @classmethod
    def from_config(
        cls,
        cfg: EdenRLConfig,
        *,
        clip_actions: float,
        obs_group: str = "policy",
        random_episode_starts: bool = False,
        seed: int | None = None,
        show_viewer: bool = False,
        eval_mode: bool = False,
    ) -> Sb3VecEnvWrapper:
        """Construct and **build** the env ``cfg`` (any :class:`EdenRLConfig`) describes, and wrap it.

        The env records final observations, on a copy of ``cfg``; ``seed`` seeds every RNG before the
        build (``None`` leaves them alone); ``show_viewer`` and ``eval_mode`` go to the env's
        ``from_config``. :meth:`close` destroys the env.
        """
        return cls._from_config(
            cfg,
            clip_actions=clip_actions,
            observation_groups=(obs_group,),
            record_final_observations=True,
            seed=seed,
            show_viewer=show_viewer,
            eval_mode=eval_mode,
            obs_group=obs_group,
            random_episode_starts=random_episode_starts,
        )

    @classmethod
    def _check_clip_actions(cls, clip_actions: float | None, field: str = "clip_actions") -> None:
        if clip_actions is None:
            raise ValueError(
                "Sb3VecEnvWrapper needs a bounded action space — every SB3 algorithm asserts finite action bounds — "
                "but got `clip_actions=None`. Fix: pass a bound, e.g. `Sb3VecEnvWrapper.from_config(cfg, "
                "clip_actions=1.0)`; use the task's `runner_options.clip_actions` when it has one."
            )
        super()._check_clip_actions(clip_actions, field)

    def reset(self) -> VecEnvObs:
        if any(self._options):
            raise TypeError(
                f"set_options({self._options[0]!r}): the Eden env takes no reset options. Configure the task "
                "through its config (or its TaskMod kwargs) instead."
            )
        # SB3 hands out one seed per env (``seed + i``); Eden's envs share their RNGs, so the first seeds them all.
        obs, _ = self._reset_env(self._seeds[0])
        self._reset_seeds()
        return _to_numpy(obs[self.obs_group])

    def step_async(self, actions: np.ndarray) -> None:
        self._actions = actions

    def step_wait(self) -> VecEnvStepReturn:
        obs, reward, terminated, truncated, info = self._step_env(self._actions)
        done = (terminated | truncated).cpu().numpy()
        infos: list[dict[str, Any]] = [{} for _ in range(self.num_envs)]
        done_ids = np.flatnonzero(done)
        if done_ids.size:
            # Only the done rows go to the host.
            rows = torch.as_tensor(done_ids, device=reward.device)
            final_obs = _to_numpy(_index(info["final_obs"][self.obs_group], rows))
            has_final_obs = info["_final_obs"][rows].cpu().numpy()
            time_out = time_outs(terminated, truncated)[rows].cpu().numpy()
            episode_return = info["episode"]["r"][rows].cpu().numpy()
            episode_length = info["episode"]["l"][rows].cpu().numpy()
            for k, i in enumerate(done_ids):
                infos[i] = {
                    "TimeLimit.truncated": bool(time_out[k]),
                    "episode": {"r": float(episode_return[k]), "l": int(episode_length[k])},
                }
                if has_final_obs[k]:
                    infos[i]["terminal_observation"] = _index(final_obs, k)
        return _to_numpy(obs[self.obs_group]), reward.cpu().numpy(), done, infos

    def close(self) -> None:
        self._close_env()

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> list[Any]:
        """``attr_name`` of the one batched env, repeated for each index (the envs are not separate objects).

        ``render_mode`` is ``None``: Eden renders through its viewer and cameras, not SB3's ``render()``.
        """
        value = None if attr_name == "render_mode" else getattr(self.env, attr_name)
        return [value] * len(self._get_indices(indices))

    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:
        raise NotImplementedError(
            f"set_attr({attr_name!r}): an Eden env is one batched simulation, not {self.num_envs} env objects, "
            "so it has no per-env attribute to set. Change the task's config before building it instead."
        )

    def env_method(self, method_name: str, *method_args, indices: VecEnvIndices = None, **method_kwargs) -> list[Any]:
        raise NotImplementedError(
            f"env_method({method_name!r}): an Eden env is one batched simulation, not {self.num_envs} env objects, "
            "so it has no per-env method to call. Call it on the batched env, `Sb3VecEnvWrapper.env`, instead."
        )

    def env_is_wrapped(self, wrapper_class: type[gym.Wrapper], indices: VecEnvIndices = None) -> list[bool]:
        return [False] * len(self._get_indices(indices))


def _to_numpy(value: torch.Tensor | dict[str, Any]) -> np.ndarray | dict[str, Any]:
    if isinstance(value, dict):
        return {key: _to_numpy(item) for key, item in value.items()}
    return value.cpu().numpy()


def _index(value: Any, index: Any) -> Any:
    """``value[index]`` for an array or tensor, or for each leaf of a dict of them."""
    if isinstance(value, dict):
        return {key: _index(item, index) for key, item in value.items()}
    return value[index]
