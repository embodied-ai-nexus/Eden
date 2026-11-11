"""Gymnasium vector-env wrapper for Eden: the entry point for RL loops Eden does not drive itself.

:class:`GymnasiumVecEnvWrapper` presents an :class:`RLEnvBase` as a batched
:class:`gymnasium.vector.VectorEnv` whose observations, rewards and dones stay torch tensors on the
env's device. CleanRL-style training loops use it directly. Its core — spaces, action clamping, final
observations, episode statistics — is :class:`~eden.envs.wrappers.vec_env_mixin.VecEnvMixin`, which the
stable-baselines3 and skrl wrappers mix in too.

Usage::

    import eden as en
    import genesis as gs
    import gymnasium as gym
    from eden.envs.wrappers.gymnasium_env import register_gymnasium_envs

    en.init(backend=gs.gpu)
    register_gymnasium_envs()  # every registered task, as "Eden/<task>"
    envs = gym.make_vec("Eden/reacher", num_envs=1024)
    obs, info = envs.reset(seed=0)
    obs, reward, terminated, truncated, info = envs.step(actions)  # obs["policy"]: (1024, D) tensor

or, from a config you already hold, ``GymnasiumVecEnvWrapper.from_config(cfg)``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium.vector import AutoresetMode, VectorEnv
from gymnasium.vector.utils import batch_space

from eden.envs.wrappers.vec_env_mixin import VecEnvMixin

if TYPE_CHECKING:
    from eden.envs.base import RLEnvBase
    from eden.utils.configs import EdenRLConfig

#: The gymnasium namespace every Eden task is registered under: ``gym.make_vec("Eden/<task>")``.
GYM_NAMESPACE = "Eden"


class GymnasiumVecEnvWrapper(VecEnvMixin, VectorEnv):
    """An :class:`RLEnvBase` as a batched :class:`gymnasium.vector.VectorEnv` of torch tensors.

    ``step`` takes a ``(num_envs, action_dim)`` tensor (a numpy array is moved to the env's device)
    and returns ``(obs, reward, terminated, truncated, info)`` as :class:`VecEnvMixin` describes them:
    gymnasium's ``SAME_STEP`` autoreset, with the observation a done env ended on in
    ``info["final_obs"]``. ``reset(seed=...)`` takes one int: Eden's envs share their RNGs.

    :meth:`close` destroys the env's scene only when the wrapper built the env (:meth:`from_config`,
    ``gym.make_vec``); an env passed to ``__init__`` stays the caller's.

    Parameters
    ----------
    env:
        The Eden RL environment, already built. It stays reachable as :attr:`env`; ``unwrapped`` is
        gymnasium's, and returns this wrapper.
    clip_actions, final_observations, random_episode_starts:
        See :class:`VecEnvMixin`. ``env`` must record final observations unless
        ``final_observations=False``.
    """

    metadata = {"autoreset_mode": AutoresetMode.SAME_STEP, "render_modes": []}

    def __init__(
        self,
        env: RLEnvBase,
        clip_actions: float | None = None,
        *,
        final_observations: bool = True,
        random_episode_starts: bool = False,
    ):
        self._init_vec_env(
            env,
            clip_actions=clip_actions,
            final_observations=final_observations,
            random_episode_starts=random_episode_starts,
            observation_groups=None,
        )
        self.num_envs = env.num_envs
        self.device = self._sim_device
        self.render_mode = None
        self.observation_space = batch_space(self.single_observation_space, self.num_envs)
        self.action_space = batch_space(self.single_action_space, self.num_envs)

    @classmethod
    def from_config(
        cls,
        cfg: EdenRLConfig,
        *,
        clip_actions: float | None = None,
        final_observations: bool = True,
        random_episode_starts: bool = False,
        seed: int | None = None,
        show_viewer: bool = False,
        eval_mode: bool = False,
    ) -> GymnasiumVecEnvWrapper:
        """Construct and **build** the env ``cfg`` (any :class:`EdenRLConfig`) describes, and wrap it.

        The env records final observations as ``final_observations`` says, on a copy of ``cfg``;
        ``seed`` seeds every RNG before the build (``None`` leaves them alone); ``show_viewer`` and
        ``eval_mode`` go to the env's ``from_config``. :meth:`close` destroys the env.
        """
        return cls._from_config(
            cfg,
            clip_actions=clip_actions,
            observation_groups=None,
            record_final_observations=final_observations,
            seed=seed,
            show_viewer=show_viewer,
            eval_mode=eval_mode,
            final_observations=final_observations,
            random_episode_starts=random_episode_starts,
        )

    def reset(
        self, *, seed: int | list[int] | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reset every env. ``seed`` (one int — Eden's RNGs are shared by all envs) seeds them first."""
        if isinstance(seed, list):
            raise TypeError(
                f"reset(seed={seed!r}): Eden's envs share one set of RNGs, so they take a single int seed, "
                "not one per env. Fix: pass `seed=<int>`."
            )
        if options is not None:
            raise TypeError(
                f"reset(options={options!r}): the Eden gymnasium wrapper takes no reset options. Configure the "
                "task through its config (or its TaskMod kwargs to `gym.make_vec`) instead."
            )
        obs, log = self._reset_env(seed)
        return obs, {"log": log}

    def step(
        self, actions: torch.Tensor | np.ndarray
    ) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        return self._step_env(actions)

    def render(self) -> None:
        """Eden renders through its viewer and :class:`~eden.entities.camera.Camera`, not ``render()``."""
        return None

    def close_extras(self, **kwargs: Any) -> None:
        """Destroy the env's scene if this wrapper built it (gymnasium's ``close()`` calls this once)."""
        self._close_env()

    def __del__(self) -> None:
        # gymnasium closes a vector env when it is garbage-collected. Not this one: a finalizer can run at
        # interpreter shutdown, after Genesis has torn down, so the scene is destroyed only by an explicit close().
        pass


# ----------------------------------------------------------------------------------------------
# Registration: gym.make_vec("Eden/<task>")
# ----------------------------------------------------------------------------------------------


def register_gymnasium_envs() -> list[str]:
    """Register every task in ``TASK_REGISTRY`` with gymnasium as ``Eden/<task>``; return the new ids.

    Builds nothing and imports no task module or Genesis: the registry lists its tasks from source.
    A task is built only by ``gym.make_vec("Eden/<task>", num_envs=N, **kwargs)``, which needs
    ``eden.init()`` first and takes:

    - ``seed``, ``clip_actions``, ``final_observations``, ``random_episode_starts``, ``show_viewer``,
      ``eval_mode``: see :meth:`GymnasiumVecEnvWrapper.from_config`;
    - any other keyword: the task's TaskMod parameters, as ``TASK_REGISTRY.build(task, **kwargs)``.

    ``num_envs`` replaces the task's ``env_options.num_envs`` — gymnasium passes its own default of 1
    when you leave it out. Ids already registered (an earlier call) are left alone, so calling this
    again only adds tasks registered since.
    """
    from eden.tasks import TASK_REGISTRY

    added = []
    for task in TASK_REGISTRY.list_tasks():
        env_id = f"{GYM_NAMESPACE}/{task}"
        if env_id in gym.registry:
            continue
        gym.register(
            id=env_id,
            entry_point=_make_single_env,
            vector_entry_point=make_vec_env,
            kwargs={"task": task},
            disable_env_checker=True,
        )
        added.append(env_id)
    return added


def make_vec_env(
    num_envs: int,
    *,
    task: str,
    seed: int | None = None,
    clip_actions: float | None = None,
    final_observations: bool = True,
    random_episode_starts: bool = False,
    show_viewer: bool = False,
    eval_mode: bool = False,
    **modifier_kwargs: Any,
) -> GymnasiumVecEnvWrapper:
    """The ``vector_entry_point`` of ``Eden/<task>``: build ``task`` with ``num_envs`` envs and wrap it."""
    import genesis as gs

    from eden.tasks import TASK_REGISTRY

    if not getattr(gs, "_initialized", False):
        raise RuntimeError(
            f"gym.make_vec('{GYM_NAMESPACE}/{task}') builds a Genesis scene, but Genesis is not initialized. "
            "Fix: call `eden.init(backend=gs.gpu)` (or `gs.cpu`) before `gym.make_vec`."
        )
    if num_envs < 1:
        raise ValueError(f"gym.make_vec('{GYM_NAMESPACE}/{task}', num_envs={num_envs!r}): num_envs must be >= 1.")
    cfg = TASK_REGISTRY.build(task, **modifier_kwargs)
    cfg.env_options.num_envs = num_envs
    return GymnasiumVecEnvWrapper.from_config(
        cfg,
        clip_actions=clip_actions,
        final_observations=final_observations,
        random_episode_starts=random_episode_starts,
        seed=seed,
        show_viewer=show_viewer,
        eval_mode=eval_mode,
    )


def _make_single_env(task: str, **kwargs: Any) -> gym.Env:
    """The ``entry_point`` of ``Eden/<task>``: refuses, since an Eden env is batched by construction."""
    raise TypeError(
        f"'{GYM_NAMESPACE}/{task}' was asked for a single, unbatched env — by `gym.make`, or by `gym.make_vec` with "
        "`vectorization_mode='sync'`/`'async'`, which stacks single envs — but every Eden env is a vector env. "
        f"Fix: `gym.make_vec('{GYM_NAMESPACE}/{task}', num_envs=N)` with no `vectorization_mode`. Do NOT wrap a "
        "`num_envs=1` env to fake a single one: its tensors keep the leading env dimension."
    )
