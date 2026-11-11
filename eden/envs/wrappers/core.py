"""What every Eden env wrapper shares, whichever RL library it serves.

Five wrappers present an :class:`RLEnvBase` to an RL library: ``RslRlVecEnvWrapper`` and
``RlGamesVecEnvWrapper``, which ``eden train`` drives, and the gymnasium, stable-baselines3 and skrl
wrappers, which share :class:`~eden.envs.wrappers.vec_env_mixin.VecEnvMixin` on top of this module.
:class:`EnvWrapperMixin` gives all five the same ``from_config`` build, env ownership and action clamp;
:func:`action_space`, :func:`observation_space` and :func:`time_outs` give them the same spaces and the
same split of an episode's end into terminal and timed out.

Nothing here copies a tensor: the runner wrappers hand the env's own buffers to learners that read them
at once. Copies, final observations and episode statistics, which only the outside libraries need, are
:class:`~eden.envs.wrappers.vec_env_mixin.VecEnvMixin`'s.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Self

import gymnasium as gym
import numpy as np
import torch

import eden as en

if TYPE_CHECKING:
    from eden.envs.base import RLEnvBase
    from eden.utils.configs import EdenRLConfig


class EnvWrapperMixin:
    """Building and owning the wrapped env, and clamping its actions: the part of a wrapper that holds state.

    A wrapper's ``from_config`` checks what it can against the config, then calls
    :meth:`_build_and_wrap`; its ``step`` passes the actions through :meth:`_clamp_actions`; its
    ``close`` calls :meth:`_close_env`. The mixin defines no ``__init__``: the wrapper sets :attr:`env`
    and :attr:`clip_actions` itself, checking the bound with :meth:`_check_clip_actions`.

    Attributes
    ----------
    env:
        The wrapped env.
    clip_actions:
        Actions are clamped to ``[-clip_actions, clip_actions]`` before the action manager reads them.
        ``None`` leaves them unclamped.
    """

    # Set by _build_and_wrap, the one path that builds the env: only then does _close_env destroy it. A class
    # default, so a wrapper whose __init__ raised still closes (as a no-op).
    _owns_env: bool = False

    env: RLEnvBase
    clip_actions: float | None

    @classmethod
    def _check_clip_actions(cls, clip_actions: float | None, field: str = "clip_actions") -> None:
        """Raise unless ``clip_actions`` is a finite positive bound or ``None``; a wrapper may refuse ``None`` too.

        ``field`` is where the caller set it, for the message. Runner options validate the field at
        config load (:data:`~eden.options.learning.clip_actions.ClipActions`), but not a value assigned
        after construction, so a wrapper checks it again.
        """
        # Imported here: `eden.options` loads genesis, which registering the gymnasium envs must not.
        from eden.options.learning.clip_actions import validate_clip_actions

        if clip_actions is not None:
            validate_clip_actions(clip_actions, field)

    @classmethod
    def _build_and_wrap(
        cls, cfg: EdenRLConfig, *, seed: int | None, show_viewer: bool, eval_mode: bool, **init_kwargs: Any
    ) -> Self:
        """Build the env ``cfg`` describes and wrap it as ``cls(env, **init_kwargs)``, owning the env.

        ``seed`` seeds every RNG (:func:`eden.utils.random.apply_run_seed`) before the env is
        constructed, so the build, its first reset and domain randomization are reproducible; ``None``
        leaves the process RNG as it is. The build can take minutes, so check everything ``cfg`` can
        tell before calling this. If the build or the wrapper fails, the env is destroyed before the
        error propagates.
        """
        from eden.envs.base import env_class_for_config
        from eden.utils.random import apply_run_seed

        if seed is not None:
            apply_run_seed(seed)
        env = env_class_for_config(cfg).from_config(cfg, show_viewer=show_viewer, eval_mode=eval_mode)
        try:
            env.build()
            en.logger.info(env.summary())
            wrapper = cls(env, **init_kwargs)
        except BaseException:
            # No wrapper owns the env yet, so nothing else would destroy its scene.
            try:
                env.destroy()
            except Exception as teardown_error:  # noqa: BLE001 - the error raised below is the one to read
                en.logger.warning(
                    f"{cls.__name__}.from_config failed (the error raised next), and destroying its env failed too: "
                    f"{teardown_error}"
                )
            raise
        wrapper._owns_env = True
        return wrapper

    def _clamp_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """``actions`` clamped to ``[-clip_actions, clip_actions]``; ``actions`` itself without a bound."""
        if self.clip_actions is None:
            return actions
        return torch.clamp(actions, -self.clip_actions, self.clip_actions)

    def _close_env(self) -> None:
        """Destroy the env's scene if this wrapper built it (:meth:`_build_and_wrap`); a second call does nothing.

        An env passed to the wrapper's ``__init__`` stays the caller's. Never call this from ``__del__``:
        a finalizer can run at interpreter shutdown, after Genesis has torn down.
        """
        if self._owns_env:
            self._owns_env = False
            self.env.destroy()


def action_space(num_actions: int, clip_actions: float | None, dtype: torch.dtype) -> gym.spaces.Box:
    """A ``Box`` of ``num_actions`` actions of ``dtype``, bounded by ``clip_actions`` (unbounded for ``None``)."""
    bound = np.inf if clip_actions is None else clip_actions
    return gym.spaces.Box(low=-bound, high=bound, shape=(num_actions,), dtype=_numpy_dtype(dtype))


def observation_space(obs: dict[str, Any]) -> gym.spaces.Dict:
    """The per-env space of an observation, read off a real one: a ``Dict`` with one entry per group.

    A group that concatenates its terms is a ``Box``, one that does not a ``Dict`` of its terms. Shapes
    and dtypes are the observation manager's: an integer or bool term gets an integer or bool ``Box``,
    not a float32 one.
    """
    return gym.spaces.Dict({group: _space_for(value) for group, value in obs.items()})


def time_outs(terminated: torch.Tensor, truncated: torch.Tensor) -> torch.Tensor:
    """The envs whose episode ended only by timing out: the ones a learner bootstraps from their value.

    ``terminated`` and ``truncated`` are the termination terms' verdicts split by ``time_out``, and an
    env can be both on one step (it failed on the step its episode timed out). Such an env is terminal,
    so it is not one of these.
    """
    return truncated & ~terminated


def _numpy_dtype(dtype: torch.dtype) -> np.dtype:
    return torch.empty((), dtype=dtype).numpy().dtype


def _space_for(value: torch.Tensor | dict[str, torch.Tensor]) -> gym.Space:
    """The per-env space of one observation group (or term): a ``Box`` of its row shape, a ``Dict`` for a dict."""
    if isinstance(value, dict):
        return gym.spaces.Dict({name: _space_for(term) for name, term in value.items()})
    dtype = value.dtype
    if dtype.is_floating_point:
        low, high = -np.inf, np.inf
    elif dtype == torch.bool:
        low, high = 0, 1
    else:
        info = torch.iinfo(dtype)
        low, high = info.min, info.max
    return gym.spaces.Box(low=low, high=high, shape=tuple(value.shape[1:]), dtype=_numpy_dtype(dtype))
