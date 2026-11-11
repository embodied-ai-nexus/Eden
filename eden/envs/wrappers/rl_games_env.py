"""rl_games environment wrapper for Eden.

Wraps :class:`RLEnvBase` as an :class:`IVecEnv` so that it can be consumed
directly by the rl_games ``Runner``.

Usage::

    from eden.envs.wrappers.rl_games_env import RlGamesVecEnvWrapper

    env = RlGamesVecEnvWrapper.from_config(config)

    from rl_games.torch_runner import Runner
    runner = Runner()
    runner.load(env.runner_dict)
    env.configure_runner(runner)
    runner.run({"train": True})

Multi-agent: with ``RlGamesPpoRunnerOptions(agents=["a", "b"])`` the env's observation groups,
reward groups and action terms named ``a`` / ``b`` become two agents of one shared rl_games policy
(``env_info["agents"] = 2``). Rows are env-major: row ``e * A + i`` is agent ``i`` of env ``e``.
"""

from __future__ import annotations

import math
import os
import re
from typing import TYPE_CHECKING, Any

import gymnasium as gym
import torch
from rl_games.common.ivecenv import IVecEnv

from eden.envs.wrappers.core import EnvWrapperMixin, action_space, time_outs
from eden.options.learning.rl_games.rl import check_agents_options
from eden.options.learning.rl_games.translate import translate_runner_dict

if TYPE_CHECKING:
    from eden.envs.base import RLEnvBase
    from eden.options.learning.rl_games.rl import RlGamesPpoRunnerOptions


def rl_games_checkpoints(nn_dir: str, name: str) -> list[tuple[int, str]]:
    """The run's periodic ``last_<name>_ep_<epoch>_rew_<reward>.pth`` checkpoints in ``nn_dir``, oldest first.

    Each is ``(epoch, filename)``; the best-reward ``<name>.pth`` is not one of them.
    """
    pattern = re.compile(rf"last_{re.escape(name)}_ep_(\d+)_rew_.*\.pth")
    found = ((pattern.fullmatch(filename), filename) for filename in os.listdir(nn_dir))
    return sorted((int(match.group(1)), filename) for match, filename in found if match)


def retain_checkpoints(agent: Any, keep_last_checkpoints: int | None) -> Any:
    """Have an rl_games agent keep only its run's newest ``keep_last_checkpoints`` periodic checkpoints.

    rl_games saves ``nn/last_<name>_ep_<epoch>_rew_<reward>.pth`` every ``save_frequency`` epochs and once more at
    ``max_epochs``, and never deletes one, so a long run at a short frequency fills a disk quota. After this, each such
    save deletes the older ones. The best-reward checkpoint ``<name>.pth`` (overwritten in place) and any other save
    prune nothing. rl_games uploads no checkpoint to a logger, so nothing can still be reading the deleted files.

    It wraps the instance's ``save`` rather than subclassing ``A2CAgent``, so it composes with whatever agent the
    runner's builder makes (rl_games' own, or one a caller registered). rl_games calls ``self.save``, which finds the
    wrapper first. Returns ``agent``.
    """
    save = agent.save

    def save_and_prune(fn: str) -> None:
        save(fn)  # writes ``fn + ".pth"``
        _prune_checkpoints(fn + ".pth", agent.config["name"], keep_last_checkpoints)

    agent.save = save_and_prune
    return agent


def _prune_checkpoints(saved: str, name: str, keep: int | None) -> None:
    """Delete all but the newest ``keep`` periodic checkpoints of run ``name`` beside ``saved``, ``saved`` included.

    At ``max_epochs`` rl_games saves twice at the same epoch: the periodic save, then the final one under a differently
    formatted reward. The later save supersedes the earlier, so ``keep`` counts distinct epochs. Only epochs before
    ``saved``'s are candidates, so ``saved`` itself always survives.
    """
    nn_dir, filename = os.path.split(saved)
    checkpoints = rl_games_checkpoints(nn_dir, name)
    epoch = next((epoch for epoch, other in checkpoints if other == filename), None)
    if keep is None or epoch is None:
        return
    older = [other for other_epoch, other in checkpoints if other_epoch < epoch]
    superseded = [other for other_epoch, other in checkpoints if other_epoch == epoch and other != filename]
    for other in older[: max(len(older) - (keep - 1), 0)] + superseded:
        os.remove(os.path.join(nn_dir, other))


class RlGamesVecEnvWrapper(EnvWrapperMixin, IVecEnv):
    """Adapter between Eden's :class:`RLEnvBase` and rl_games' :class:`IVecEnv`.

    Observations are returned as GPU tensors so that rl_games' auto-detection
    (``is_tensor_obses``) keeps the entire pipeline on-device.

    With ``options.agents`` set (``A`` agents over ``N`` envs) the wrapper presents ``N * A`` rows
    to rl_games, env-major (row ``e * A + i`` is agent ``i`` of env ``e`` — rl_games' episode stats
    take every ``A``-th done index, so an env's agents must be contiguous). Each row's observation is
    its agent's observation group, its action goes to the action term of the same name (the terms
    must be declared in ``agents`` order), and its reward is
    ``reward_manager.get_group_reward(agent)``; dones and ``time_outs`` repeat per agent. The spaces
    are per agent, and there is no central-value critic.

    ``step`` returns the env's own buffers (or their concatenation), not copies: rl_games
    copies them into its rollout storage before the next step. :meth:`close` destroys the
    env's scene only when :meth:`from_config` built it.

    Parameters
    ----------
    env:
        The Eden RL environment (already built).
    options:
        rl_games runner options that will be translated into the
        ``params`` dict consumed by ``Runner.load()``.
    """

    def __init__(
        self,
        env: RLEnvBase,
        options: RlGamesPpoRunnerOptions | None,
    ):
        # Optional in the signature only so a task with no `runner_options` reaches this
        # error instead of an `AttributeError` on `options.clip_actions`; a wrapper without
        # runner options cannot be constructed.
        if options is None:
            raise ValueError(
                "`RlGamesVecEnvWrapper` requires `runner_options` (it assumes a PPO/rl_games runner). "
                "For a task without one, build via `RLEnvBase.from_config(cfg)` "
                "(or `MimicRLEnvBase.from_config(cfg)` for motion tasks) instead."
            )
        self._check_clip_actions(options.clip_actions, field="runner_options.clip_actions")
        self.env = env
        self.runner_options = options
        self.clip_actions = options.clip_actions

        # Re-checked here, not only in the options validator: a field assigned after construction skips it.
        check_agents_options(options.agents, options.obs_groups)

        self.num_envs = self.unwrapped.num_envs
        self.device = self.unwrapped.device

        self._agents: list[str] = list(options.agents)
        self._num_agents = max(len(self._agents), 1)

        if self._agents:
            self._use_central_value = False
            self._actor_obs_dim, self.num_actions = self._validate_agents()
        else:
            self.num_actions = self.unwrapped.action_manager.total_action_dim

            # Resolve observation group mapping.
            obs_groups = options.obs_groups
            if not obs_groups:
                groups = list(self.unwrapped.observation_manager.group_term_names.keys())
                obs_groups = {"actor": groups[:1], "critic": groups[:1]}

            self._actor_groups = obs_groups.get("actor", [])
            self._critic_groups = obs_groups.get("critic", self._actor_groups)
            self._use_central_value = self._actor_groups != self._critic_groups

            # Pre-compute observation dimensions (concatenated).
            self._actor_obs_dim = self._total_obs_dim(self._actor_groups)
            self._critic_obs_dim = self._total_obs_dim(self._critic_groups) if self._use_central_value else 0

        self._configure_gym_spaces()

        self._runner_dict_cache: dict | None = None

        # Reset at the start — rl_games calls env_reset() before training but
        # we need spaces / dim info which requires at least one reset.
        self.unwrapped.reset()

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls, cfg, *, show_viewer: bool = False, eval_mode: bool = False, apply_seed: bool = True
    ) -> "RlGamesVecEnvWrapper":
        """Create a wrapper from an :class:`EdenRLConfig`.

        Unlike ``RLEnvBase.from_config``, this also **builds** the environment, and (unless
        ``apply_seed=False``) mutates **process-global** RNG state via
        :func:`eden.utils.random.apply_run_seed`. ``cfg`` itself is left untouched. The wrapper
        owns the env: :meth:`close` destroys it, and so does a wrapper that fails to construct.

        Parameters
        ----------
        cfg: EdenRLConfig
            The configuration object.
        show_viewer: bool
            Whether to show the viewer.
        eval_mode: bool
            Whether to run in eval mode.
        apply_seed: bool
            Whether to seed the run from ``cfg.runner_options.seed`` before constructing the
            env. Pass ``False`` for a caller that must not disturb process-global RNG state.
        """
        # `getattr`, not `cfg.runner_options`: a plain `EdenConfig` does not declare the
        # field at all, and that caller mistake deserves this message rather than an
        # `AttributeError` naming neither the task nor the fix.
        if getattr(cfg, "runner_options", None) is None:
            raise ValueError(
                f"Task '{type(cfg).__name__}' has no `runner_options`, so it cannot be built through "
                "`RlGamesVecEnvWrapper` (which assumes a PPO/rl_games runner). Build it via "
                "`RLEnvBase.from_config(cfg)` (or `MimicRLEnvBase.from_config(cfg)` for motion tasks) instead."
            )
        cls._check_clip_actions(cfg.runner_options.clip_actions, field="runner_options.clip_actions")
        # rl_games does seed itself, but only inside `Runner.load(env.runner_dict)` — which
        # runs *after* this method has already built the env, so unseeded here the build and
        # its first reset would fall outside the configured seed entirely. rl_games then
        # re-seeds from `runner_dict["seed"]` (the base), a no-op for every rl_games run Eden
        # supports: they are single-process, so the base *is* what was applied here. rl_games
        # does its own `global_rank` offset when `multi_gpu` is set, which Eden never sets for
        # this backend.
        return cls._build_and_wrap(
            cfg,
            seed=cfg.runner_options.seed if apply_seed else None,
            show_viewer=show_viewer,
            eval_mode=eval_mode,
            options=cfg.runner_options,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def unwrapped(self) -> RLEnvBase:
        return self.env.unwrapped

    @property
    def dt(self) -> float:
        return self.unwrapped.dt

    @property
    def runner_dict(self) -> dict[str, Any]:
        """rl_games config dict, ready for ``Runner.load()``.

        This dict is safe to ``deepcopy`` (no references to the live
        environment).  After loading, call :meth:`configure_runner` to
        inject the live ``env_info`` and ``vec_env`` references that the
        algorithm needs at construction time.
        """
        if self._runner_dict_cache is None:
            self._runner_dict_cache = translate_runner_dict(self.runner_options.dict(), num_envs=self.num_envs)
        return self._runner_dict_cache

    def configure_runner(self, runner: Any) -> None:
        """Inject ``env_info`` and ``vec_env`` into a loaded ``Runner``, and have the agent it builds honour
        ``keep_last_checkpoints`` (:func:`retain_checkpoints`).

        Must be called **after** ``runner.load(self.runner_dict)`` and
        **before** ``runner.run_train()`` (or manual agent creation).

        Parameters
        ----------
        runner:
            A :class:`rl_games.torch_runner.Runner` instance that has
            already been loaded with :attr:`runner_dict`.
        """
        runner.params["config"]["env_info"] = self.get_env_info()
        runner.params["config"]["vec_env"] = self
        # Wrap the builder already registered rather than replacing it, so a caller's own agent survives.
        # ObjectFactory has no getter, hence its private dict.
        from rl_games.algos_torch.a2c_continuous import A2CAgent

        keep = self.runner_options.keep_last_checkpoints
        build = runner.algo_factory._builders.get("a2c_continuous") or (lambda **kwargs: A2CAgent(**kwargs))
        runner.algo_factory.register_builder(
            "a2c_continuous", lambda **kwargs: retain_checkpoints(build(**kwargs), keep)
        )

    @property
    def config(self):
        """The wrapped env's :attr:`~eden.envs.base.EnvBase.config`, with *this* wrapper's runner options.

        The base env never receives ``runner_options``, so it can only report the ones its source
        config carried — which differ from the live ones whenever a wrapper was constructed
        directly (``RlGamesVecEnvWrapper(env, options=...)``) rather than through
        :meth:`from_config`.

        ``model_copy``, not attribute assignment: the base config's own class is already the one
        to return, and ``update=`` writes past both the lock guard and the class's extra-forbid
        policy — the latter mattering for the degenerate case of a wrapped env whose config class
        does not declare ``runner_options`` at all, where rebuilding the class through
        ``model_construct`` would drop the field instead of reporting it.
        """
        return self.unwrapped.config.model_copy(update={"runner_options": self.runner_options})

    # ------------------------------------------------------------------
    # IVecEnv interface
    # ------------------------------------------------------------------

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor | dict, torch.Tensor, torch.Tensor, dict]:
        actions = self._clamp_actions(actions)
        if self._agents:
            return self._step_agents(actions)

        obs_dict, rew, terminated, truncated, extras = self.unwrapped.step(actions)
        dones = (terminated | truncated).to(dtype=torch.uint8)
        # rl_games bootstraps every `time_outs` env's value.
        extras["time_outs"] = time_outs(terminated, truncated)
        return self._rl_obs(obs_dict), rew, dones, extras

    def reset(self) -> torch.Tensor | dict:
        obs_dict, _ = self.unwrapped.reset()
        return self._rl_obs(obs_dict)

    def _rl_obs(self, obs_dict: dict) -> torch.Tensor | dict:
        """The observation rl_games reads: the agents' rows, or the actor groups (with the critic's as ``states``)."""
        if self._agents:
            return self._agent_obs(obs_dict)
        obs = self._flatten_obs(obs_dict, self._actor_groups)
        if self._use_central_value:
            obs = {"obs": obs, "states": self._flatten_obs(obs_dict, self._critic_groups)}
        return obs

    def has_action_masks(self) -> bool:
        return False

    def get_number_of_agents(self) -> int:
        return self._num_agents

    def get_env_info(self) -> dict[str, Any]:
        info: dict[str, Any] = {
            "action_space": self.action_space,
            "observation_space": self.observation_space,
            "agents": self._num_agents,
            "value_size": 1,
        }
        if self._use_central_value:
            info["use_global_observations"] = True
            info["state_space"] = self.state_space
        return info

    def seed(self, seed: int) -> None:
        self.unwrapped.seed(seed)

    def set_train_info(self, env_frames: int, *args: Any, **kwargs: Any) -> None:
        pass

    def get_env_state(self) -> None:
        return None

    def set_env_state(self, env_state: Any) -> None:
        pass

    def close(self) -> None:
        """Destroy the env's scene if :meth:`from_config` built it; an env passed to ``__init__`` stays the caller's."""
        self._close_env()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _step_agents(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """:meth:`step` for ``options.agents``: ``(N * A, a)`` actions in, ``N * A`` rows out."""
        n, a = self.num_envs, self._num_agents
        # Row e*A+i is agent i of env e, so each env's A rows are contiguous: (N*A, d) -> (N, A, d) ->
        # (N, A*d) lays env e's actions out agent by agent, the order the action terms slice them in.
        actions = actions.reshape(n, a, -1).reshape(n, -1)

        obs_dict, _, terminated, truncated, extras = self.unwrapped.step(actions)
        reward_manager = self.unwrapped.reward_manager
        # Each group's share of the scalar reward the env returned — weighted and dt-scaled alike.
        rew = torch.stack([reward_manager.get_group_reward(g) for g in self._agents], dim=1).reshape(n * a)
        dones = (terminated | truncated).to(dtype=torch.uint8).repeat_interleave(a)

        extras["time_outs"] = time_outs(terminated, truncated).repeat_interleave(a)
        return self._agent_obs(obs_dict), rew, dones, extras

    def _agent_obs(self, obs_dict: dict) -> torch.Tensor:
        """Interleave the agents' observation groups into ``(N * A, D)``: row ``e * A + i`` is agent ``i`` of env
        ``e``.
        """
        per_agent = [self._flatten_obs(obs_dict, [g]) for g in self._agents]
        return torch.stack(per_agent, dim=1).reshape(self.num_envs * self._num_agents, -1)

    def _validate_agents(self) -> tuple[int, int]:
        """Check ``options.agents`` against the env; return the per-agent ``(obs_dim, action_dim)``."""
        agents = self._agents
        env = self.unwrapped
        obs_groups = list(env.observation_manager.group_term_names)
        reward_groups = list(env.reward_manager.group_term_names)
        action_terms = list(env.action_manager.active_terms)
        rosters = (
            ("an observation group", obs_groups, "an `ObservationGroupOptions` entry", "observation_options"),
            ("a reward group", reward_groups, "a `RewardGroupOptions` entry", "reward_options"),
            ("an action term", action_terms, "an action term", "action_options"),
        )
        for agent in agents:
            for kind, available, entry, options_field in rosters:
                if agent not in available:
                    raise ValueError(
                        f"RlGamesPpoRunnerOptions.agents names '{agent}', which is not {kind} of the env.\n"
                        f"The env's {kind.split(' ', 1)[1]}s: {available}; agents: {agents}.\n"
                        f"Fix: correct the name in `agents`, or declare {entry} named '{agent}' in `{options_field}` "
                        "(each agent owns the observation group, reward group and action term of its name)."
                    )
        if action_terms != agents:
            extra = [t for t in action_terms if t not in agents]
            why = (
                f"the action terms {extra} belong to no agent, so no policy row would drive them"
                if extra
                else "the terms are declared in a different order"
            )
            raise ValueError(
                f"RlGamesPpoRunnerOptions.agents={agents} does not match the action terms {action_terms}: {why}.\n"
                "Each env's action vector is the agents' actions concatenated in `agents` order, and the action "
                "manager slices it in its term-declaration order, so the two lists must be equal.\n"
                f"Fix: set `agents={action_terms}`"
                + ("" if extra else f", or declare the action terms in the order {agents}")
                + " (every agent also needs an observation group and a reward group of its name)."
            )
        extra_rewards = [g for g in reward_groups if g not in agents]
        if extra_rewards:
            raise ValueError(
                f"RlGamesPpoRunnerOptions.agents={agents} leaves the reward groups {extra_rewards} to no agent, so "
                "their rewards would be dropped from training. (Ungrouped reward terms form the group 'default'.)\n"
                f"Fix: move those terms into the groups of the agents they reward ({agents}), or name their group as "
                "an agent."
            )
        obs_dims = {g: self._total_obs_dim([g]) for g in agents}
        action_dims = dict(zip(action_terms, env.action_manager.action_term_dim))
        for what, dims in (("observation", obs_dims), ("action", action_dims)):
            if len(set(dims.values())) > 1:
                raise ValueError(
                    f"RlGamesPpoRunnerOptions.agents share one rl_games policy, but their {what} dims differ: {dims}.\n"
                    f"Fix: give every agent's {what} {'group' if what == 'observation' else 'term'} the same width. "
                    "Do NOT zero-pad the shorter one in the wrapper: the shared network would read a padding "
                    "column as a real feature for one agent and a real feature for the other."
                )
        return obs_dims[agents[0]], action_dims[agents[0]]

    def _total_obs_dim(self, groups: list[str]) -> int:
        """Total flattened observation dimension across *groups*."""
        total = 0
        for g in groups:
            dim = self.unwrapped.observation_manager.group_obs_dim[g]
            if isinstance(dim, tuple):
                total += math.prod(dim)
            else:
                # Non-concatenated group — sum individual term dims.
                total += sum(math.prod(d) for d in dim)
        return total

    def _flatten_obs(self, obs_dict: dict, groups: list[str]) -> torch.Tensor:
        """Concatenate observation groups into a single ``(num_envs, D)`` tensor."""
        n = self.num_envs
        tensors = []
        for g in groups:
            v = obs_dict[g]
            if isinstance(v, dict):
                for t in v.values():
                    tensors.append(t.reshape(n, -1))
            else:
                tensors.append(v.reshape(n, -1))
        if len(tensors) == 1:
            return tensors[0]
        return torch.cat(tensors, dim=-1)

    def _configure_gym_spaces(self) -> None:
        """Set up Gymnasium spaces used by rl_games for network construction.

        The observation spaces are float32 whatever the observation terms' dtypes (unlike
        :func:`~eden.envs.wrappers.core.observation_space`): rl_games allocates its rollout storage
        with the space's dtype and feeds it to a float network, so an integer or bool term is stored
        as a float.
        """
        self.observation_space = gym.spaces.Box(low=-math.inf, high=math.inf, shape=(self._actor_obs_dim,))
        if self._use_central_value:
            self.state_space = gym.spaces.Box(low=-math.inf, high=math.inf, shape=(self._critic_obs_dim,))
        self.action_space = action_space(
            self.num_actions, self.clip_actions, self.unwrapped.action_manager.action.dtype
        )
