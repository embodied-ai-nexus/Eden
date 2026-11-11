"""RSL RL environment wrapper for Eden.

Auto-detects the installed ``rsl_rl`` version and translates the runner
config dict accordingly.

Usage::

    from eden.envs.wrappers.rsl_rl_env import RslRlVecEnvWrapper, make_rsl_rl_runner

    env = RslRlVecEnvWrapper.from_config(config)
    runner = make_rsl_rl_runner(env)
"""

from __future__ import annotations

import contextlib
import copy
import inspect
import io
import json
import os
import re
import statistics
import sys
import threading
import time
from importlib import metadata
from typing import TYPE_CHECKING, Any, cast

import gymnasium as gym
import torch
from rsl_rl.env import VecEnv
from rsl_rl.runners import DistillationRunner, OnPolicyRunner
from tensordict import TensorDict

import eden as en
from eden.envs.base import MimicRLEnvBase
from eden.envs.wrappers.core import EnvWrapperMixin, action_space, observation_space, time_outs
from eden.options.learning.rsl_rl.translate import translate_runner_dict
from eden.utils import distributed as eden_dist
from eden.utils.assets import model_checkpoints
from eden.utils.run_manifest import read_run_manifest
from eden.utils.torch import read_checkpoint_index
from eden.utils.warning import warn_once

try:
    import wandb as _wandb
except ImportError:
    _wandb = None

if TYPE_CHECKING:
    from eden.envs.base import RLEnvBase
    from eden.options.learning.rsl_rl.rl import RslRlBaseRunnerOptions


class RslRlVecEnvWrapper(EnvWrapperMixin, VecEnv, gym.Env):
    """Wrapper for RSL RL environments.

    Works with rsl_rl v3 / v4 / v5 — the ``VecEnv`` interface is identical
    across all versions.  The ``runner_dict`` property automatically
    translates the Eden config to whatever format the installed version
    expects.

    ``step`` returns the env's own buffers, not copies: rsl_rl reads them before
    the next step rewrites them. :meth:`close` destroys the env's scene only
    when :meth:`from_config` built it.

    Parameters
    ----------
    env: RLEnvBase
        The environment to wrap, already built.
    options: RslRlBaseRunnerOptions
        The runner options; their ``clip_actions`` clamps the actions.
    """

    def __init__(
        self,
        env: RLEnvBase,
        options: RslRlBaseRunnerOptions | None,
    ):
        # Optional in the signature only so a task with no `runner_options` reaches this
        # error instead of an `AttributeError` on `options.clip_actions`; a wrapper without
        # runner options cannot be constructed.
        if options is None:
            raise ValueError(
                "`RslRlVecEnvWrapper` requires `runner_options` (it assumes a PPO/rsl_rl runner). "
                "For a task without one, build via `RLEnvBase.from_config(cfg)` "
                "(or `MimicRLEnvBase.from_config(cfg)` for motion tasks) instead."
            )
        self._check_clip_actions(options.clip_actions, field="runner_options.clip_actions")
        self._runner_dict_cache: dict | None = None

        self.env = env
        self.runner_options = options
        self.clip_actions = options.clip_actions

        self.num_envs = self.unwrapped.num_envs
        self.device = torch.device(self.unwrapped.device)
        self.max_episode_length = self.unwrapped.max_episode_length
        self.num_actions = self.unwrapped.action_manager.total_action_dim

        # Reset at the start since rsl_rl does not call reset; the spaces are read off its observation.
        obs, _ = self.unwrapped.reset()
        self._configure_gym_env_spaces(obs)

    @classmethod
    def from_config(
        cls, cfg, *, show_viewer: bool = False, eval_mode: bool = False, apply_seed: bool = True
    ) -> "RslRlVecEnvWrapper":
        """Create a wrapper from a configuration object.

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
            env. Pass ``False`` for a caller that must not disturb process-global RNG state
            (hardware deployment mirrors a live robot and has no run to reproduce).

        Warning
        -------
        Unlike ``RLEnvBase.from_config``, this method also **builds** the environment, and
        (unless ``apply_seed=False``) mutates **process-global** RNG state via
        :func:`eden.utils.random.apply_run_seed`. ``cfg`` itself is left untouched, but
        building a second env mid-run restarts the RNG stream the first one is drawing from.
        The wrapper owns the env: :meth:`close` destroys it, and so does a wrapper that fails
        to construct.
        """
        # `getattr`, not `cfg.runner_options`: a plain `EdenConfig` does not declare the
        # field at all, and that caller mistake deserves this message rather than an
        # `AttributeError` naming neither the task nor the fix.
        if getattr(cfg, "runner_options", None) is None:
            raise ValueError(
                f"Task '{type(cfg).__name__}' has no `runner_options`, so it cannot be built through "
                "`RslRlVecEnvWrapper` (which assumes a PPO/rsl_rl runner). Build it via "
                "`RLEnvBase.from_config(cfg)` (or `MimicRLEnvBase.from_config(cfg)` for motion tasks) instead."
            )
        cls._check_clip_actions(cfg.runner_options.clip_actions, field="runner_options.clip_actions")
        # Seed here, not in the caller: rsl-rl-lib never applies `runner_options.seed`
        # itself, so otherwise every programmatic entry point (examples, user scripts) but
        # `eden train` would run unseeded with the field silently inert. The seed is
        # applied before the env is constructed, so the build, its first reset and all
        # domain-randomization sampling draw from the seeded RNGs.
        return cls._build_and_wrap(
            cfg,
            seed=cfg.runner_options.seed if apply_seed else None,
            show_viewer=show_viewer,
            eval_mode=eval_mode,
            options=cfg.runner_options,
        )

    @property
    def runner_dict(self) -> dict:
        """Version-translated runner config dict, ready for ``OnPolicyRunner``: a fresh copy on every access.

        Auto-detects the installed rsl_rl version and translates the runner options accordingly, once. Each access
        returns its own deep copy, because a runner consumes the dict it is given: rsl-rl-lib 3.x to 5.4 pop
        ``class_name`` out of its ``algorithm`` / ``actor`` / ``critic`` entries (FlashSAC as well), so handing out the
        cached dict, or a ``{**runner_dict}`` that shares its nested entries, would leave every later runner a bare
        ``KeyError: 'class_name'``.
        """
        if self._runner_dict_cache is None:
            self._runner_dict_cache = _translated_runner_dict(self.runner_options.dict(), self.runner_options.algorithm)
        return copy.deepcopy(self._runner_dict_cache)

    @property
    def inference_runner_dict(self) -> dict:
        """:attr:`runner_dict` for replaying a checkpoint: a training-only algorithm swapped for the one it extends.

        An algorithm options class may define ``for_inference()``, returning the options of the algorithm that loads
        its checkpoints for replay — AMP-PPO's discriminator, expert clips and replay buffer only shape the training
        reward, so its checkpoints replay under stock PPO (any rsl-rl-lib Eden supports, nothing extra loaded).
        Without one this is :attr:`runner_dict`.
        """
        for_inference = getattr(getattr(self.runner_options, "algorithm", None), "for_inference", None)
        if for_inference is None:
            return self.runner_dict
        replay = for_inference()
        options = self.runner_options.dict()
        options["algorithm"] = replay.dict()
        return _translated_runner_dict(options, replay)

    @property
    def config(self):
        """The wrapped env's :attr:`~eden.envs.base.EnvBase.config`, with *this* wrapper's runner options.

        The base env never receives ``runner_options``, so it can only report the ones its source
        config carried — which differ from the live ones whenever a wrapper was constructed
        directly (``RslRlVecEnvWrapper(env, options=...)``) rather than through
        :meth:`from_config`.

        ``model_copy``, not attribute assignment: the base config's own class is already the one
        to return, and ``update=`` writes past both the lock guard and the class's extra-forbid
        policy — the latter mattering for the degenerate case of a wrapped env whose config class
        does not declare ``runner_options`` at all, where rebuilding the class through
        ``model_construct`` would drop the field instead of reporting it.
        """
        return self.unwrapped.config.model_copy(update={"runner_options": self.runner_options})

    # rsl_rl's OnPolicyRunner accesses env.cfg internally
    cfg = config

    @classmethod
    def class_name(cls) -> str:
        return cls.__name__

    @property
    def unwrapped(self) -> RLEnvBase:
        """Returns the base environment of the wrapper."""
        return self.env.unwrapped

    @property
    def dt(self) -> float:
        return self.unwrapped.dt

    @property
    def episode_length_buf(self) -> torch.Tensor:
        return self.unwrapped.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor) -> None:  # type: ignore
        self.unwrapped.episode_length_buf = value

    def seed(self, seed: int) -> int:
        """Seed the wrapped env's RNGs with exactly ``seed`` (no DDP rank offset).

        Not the run's seed — :meth:`from_config` already applied that. Nothing in rsl_rl
        calls this; it exists to satisfy the ``VecEnv`` surface.
        """
        return self.unwrapped.seed(seed)

    def get_observations(self) -> TensorDict:
        obs_dict = self.unwrapped.observation_manager.compute()
        return TensorDict(cast(dict[str, Any], obs_dict), batch_size=[self.num_envs])

    def reset(self) -> tuple[TensorDict, dict]:
        obs_dict, extras = self.unwrapped.reset()
        return TensorDict(cast(dict[str, Any], obs_dict), batch_size=[self.num_envs]), extras

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        obs_dict, rew, terminated, truncated, extras = self.unwrapped.step(self._clamp_actions(actions))
        term_or_trunc = terminated | truncated
        assert isinstance(rew, torch.Tensor)
        assert isinstance(term_or_trunc, torch.Tensor)
        dones = term_or_trunc.to(dtype=torch.long)
        # rsl_rl bootstraps V-targets at truncation boundaries from
        # extras["time_outs"], whatever `dones` says. Report it on every step:
        # withholding it makes PPO grade every time-out as a true terminal.
        extras["time_outs"] = time_outs(terminated, truncated)
        return (
            TensorDict(cast(dict[str, Any], obs_dict), batch_size=[self.num_envs]),
            rew,
            dones,
            extras,
        )

    def close(self) -> None:
        """Destroy the env's scene if :meth:`from_config` built it; an env passed to ``__init__`` stays the caller's."""
        self._close_env()

    def _configure_gym_env_spaces(self, obs: dict[str, Any]) -> None:
        self.single_observation_space = observation_space(obs)
        self.observation_space = gym.vector.utils.batch_space(self.single_observation_space, self.num_envs)
        self.single_action_space = action_space(
            self.num_actions, self.clip_actions, self.unwrapped.action_manager.action.dtype
        )
        self.action_space = gym.vector.utils.batch_space(self.single_action_space, self.num_envs)


# runners


def _translated_runner_dict(options: dict, algorithm: Any) -> dict:
    """``RslRlBaseRunnerOptions.dict()`` with its device resolved, translated for the installed rsl-rl-lib.

    ``algorithm`` is the options object ``options["algorithm"]`` was dumped from, whose ``adapt_runner_dict`` hook (if
    any) :func:`translate_runner_dict` runs on the translated dict.
    """
    from eden.utils.torch import resolve_device

    options["device"] = resolve_device(options["device"])
    return translate_runner_dict(options, algorithm=algorithm)


class DDPRunnerMixin:
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # rsl_rl's ``Logger.process_env_step`` gates on ``log_dir is not None``
        # (not on ``disable_logs``), so on non-zero ranks the rollout loop
        # appends ``extras["log"]`` to ``Logger.ep_extras`` every step.
        # ``Logger.log`` then short-circuits on ``disable_logs`` and never
        # reaches its ``ep_extras.clear()`` — the dicts hold GPU tensor refs
        # from reward / metric / command managers, so the list pins VRAM
        # that grows linearly with iters. Neutralize the
        # append on non-zero ranks without touching ``log_dir`` (which
        # ``OnPolicyRunner.save`` still uses for the periodic-checkpoint
        # path that runs on every rank).
        if self.is_distributed and self.gpu_global_rank != 0:
            self.logger.process_env_step = lambda *_a, **_k: None

    def _configure_multi_gpu(self) -> None:
        self.gpu_world_size = eden_dist.get_world_size()
        self.is_distributed = eden_dist.is_distributed()

        if not self.is_distributed:
            self.gpu_local_rank = 0
            self.gpu_global_rank = 0
            self.multi_gpu_cfg = None
            return

        self.gpu_local_rank = eden_dist.get_local_rank()
        self.gpu_global_rank = eden_dist.get_global_rank()
        # CUDA_VISIBLE_DEVICES isolation in eden.__init__ makes gs.device == cuda:0
        # for every rank; routing through resolve_device keeps DDP and non-DDP
        # device selection on the same code path.
        from eden.utils.torch import resolve_device

        self.device = resolve_device("auto")

        self.multi_gpu_cfg = {
            "global_rank": self.gpu_global_rank,
            "local_rank": self.gpu_local_rank,
            "world_size": self.gpu_world_size,
        }
        self.cfg["multi_gpu"] = self.multi_gpu_cfg


class MotionWeightCurriculumMixin:
    """Adds a per-PPO-iteration motion-weight curriculum hook to an rsl-rl runner.

    Wraps ``self.alg.update`` so that *after* each PPO update completes (i.e.
    at the natural end of every training iteration, regardless of partial
    rollouts, eval interruptions, or runner internals) we fire
    ``motion_manager.update_motion_weight_curriculum()`` exactly once. The
    hook is reversible — the original ``alg.update`` is restored in a
    ``finally`` block so the runner is unmodified after ``learn`` returns.

    The mixin no-ops when the env is not a ``MimicRLEnvBase`` or its motion
    manager has no adaptive motion terms, so applying it unconditionally
    adds zero overhead to non-mimic tasks.
    """

    def learn(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        env_unwrapped = self.env.unwrapped
        if not isinstance(env_unwrapped, MimicRLEnvBase):
            return super().learn(*args, **kwargs)
        mm = env_unwrapped.motion_manager
        if not mm.has_adaptive_motion_weights:
            return super().learn(*args, **kwargs)

        original_update = self.alg.update
        # Some rsl-rl-lib releases' Logger enumerates only the keys present in
        # ``ep_extras[0]``; mimic summaries are populated post-drain, so they're
        # absent from the first iter's ``ep_extras[0]`` and would be silently
        # dropped. Push them to ``wandb.run`` directly here.
        iter_count = [getattr(self, "current_learning_iteration", 0)]

        def update_with_curriculum(*update_args, **update_kwargs):
            result = original_update(*update_args, **update_kwargs)
            mm.update_motion_weight_curriculum()
            if _wandb is not None and _wandb.run is not None:
                summary = mm._collect_log_extras()
                # Per-motion histograms are emitted on the wandb-direct path
                # ONLY (never through ``extras["log"]`` / rsl_rl's per-step
                # Logger), so the 9k-motion fan-out can't reach the rsl-rl
                # print loop or inflate ``ep_extras``. See
                # ``MotionManager.collect_per_motion_distributions``.
                summary.update(mm.collect_per_motion_distributions(_wandb))
                if summary:
                    _wandb.log(summary, step=iter_count[0])
            iter_count[0] += 1
            return result

        self.alg.update = update_with_curriculum
        try:
            return super().learn(*args, **kwargs)
        finally:
            self.alg.update = original_update


class ProgressFileMixin:
    """Write a machine-readable ``progress.json`` in the runner's ``log_dir`` once per iteration.

    A coding agent (or any external supervisor) can poll a tiny JSON file far more cheaply than
    it can tail training logs, and need not invent its own. This wraps the per-iteration ``Logger.log``
    call so every rsl_rl runner Eden builds emits a canonical snapshot without editing the
    third-party training loop:

        {"status", "iteration", "total_iterations", "total_timesteps", "mean_reward",
         "mean_episode_length", "best_mean_reward", "elapsed_s", "updated_at"}

    ``status`` is ``"running"`` on the per-iteration writes and ``"completed"`` on a final write
    once ``learn()`` returns normally, and ``updated_at`` is wall-clock epoch seconds — so a poller
    can tell *done* (status ``completed``) from *stalled/crashed* (status still ``running`` with a
    stale ``updated_at``, since a hard stop never reaches the terminal write) and compute
    completion from ``iteration`` / ``total_iterations``.

    The write is atomic (temp file + ``os.replace``) so a poller never reads a half-written file,
    and wrapped in a broad ``except`` so a filesystem hiccup can never take training down. It is a
    no-op on an eval runner (no ``log_dir``) and on non-zero DDP ranks (``logger.disable_logs``),
    which is where rsl_rl itself withholds file output.

    Ordered *first* in the runner MRO so its ``learn`` wraps ``logger.log`` before delegating to
    the other cooperating mixins (``MotionWeightCurriculumMixin`` etc.) and the base runner loop.
    """

    def learn(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        logger = getattr(self, "logger", None)
        if logger is None or getattr(logger, "log_dir", None) is None:
            return super().learn(*args, **kwargs)

        original_log = logger.log
        best_mean_reward = [float("-inf")]
        # Target iteration count for this call, so a poller can compute % complete. rsl_rl runs
        # `num_learning_iterations` iterations from the current one (nonzero after a resume).
        num_iters = args[0] if args else kwargs.get("num_learning_iterations", 0)
        total_iterations = int(getattr(self, "current_learning_iteration", 0)) + int(num_iters or 0)

        def log_then_write(*log_args, **log_kwargs):
            result = original_log(*log_args, **log_kwargs)
            self._try_write_progress(best_mean_reward, total_iterations, "running")
            return result

        logger.log = log_then_write
        try:
            result = super().learn(*args, **kwargs)
        finally:
            logger.log = original_log
        # Reached only when learn() returned normally: stamp the terminal state so a poller can
        # distinguish a finished run from one that was killed mid-iteration (which never gets here).
        self._try_write_progress(best_mean_reward, total_iterations, "completed")
        return result

    def _try_write_progress(self, best_mean_reward: list[float], total_iterations: int, status: str) -> None:
        try:
            self._write_progress_file(best_mean_reward, total_iterations, status)
        except Exception as exc:  # a progress sidecar must never be load-bearing
            en.logger.warning(f"Could not write progress.json: {exc}")

    def _write_progress_file(self, best_mean_reward: list[float], total_iterations: int, status: str) -> None:
        logger = self.logger
        if logger.log_dir is None or logger.disable_logs:
            return
        rewards = list(logger.rewbuffer)
        lengths = list(logger.lenbuffer)
        mean_reward = statistics.mean(rewards) if rewards else None
        if mean_reward is not None and mean_reward > best_mean_reward[0]:
            best_mean_reward[0] = mean_reward
        progress = {
            "status": status,
            "iteration": int(self.current_learning_iteration),
            "total_iterations": total_iterations,
            "total_timesteps": int(logger.tot_timesteps),
            "mean_reward": mean_reward,
            "mean_episode_length": statistics.mean(lengths) if lengths else None,
            "best_mean_reward": None if best_mean_reward[0] == float("-inf") else best_mean_reward[0],
            "elapsed_s": float(logger.tot_time),
            "updated_at": time.time(),
        }
        path = os.path.join(logger.log_dir, "progress.json")
        tmp_path = f"{path}.tmp"
        with open(tmp_path, "w") as handle:
            json.dump(progress, handle, indent=2)
        os.replace(tmp_path, path)  # atomic swap: a concurrent poller sees old-or-new, never partial


#: rsl_rl's per-iteration header, ``Learning iteration 12/3000`` (bold-wrapped and centered).
_ITERATION_HEADER = re.compile(r"Learning iteration (\d+)/(\d+)")

#: rsl_rl right-aligns each ``key:`` to this width (``rsl_rl.utils.logger.Logger.log``'s ``pad``),
#: so a metric line's colon sits at column ``pad - 1`` or later -- a key longer than the pad just
#: overflows it. That is what tells its lines apart from anything else printed during the call
#: (a wandb notice is ``wandb: ...``, colon at column 5).
_RSL_RL_PAD = 40
_METRIC_LINE = re.compile(r"^(?P<lead> *)(?P<key>[^:]+?):(?: +(?P<value>.*?))? *$")


def compact_iteration_log(text: str) -> str:
    """Rewrite rsl_rl's per-iteration console block as ``key=value`` lines, for the agent theme.

    rsl_rl prints one right-aligned line per metric between ``#`` and ``-`` rules: 45 lines and
    about 580 tokens per iteration on the flat velocity task, most of it padding, rules and
    repeated ``Episode_Reward/`` prefixes. This keeps every number and drops the layout: one line
    of the scalar metrics, then one line per ``/``-namespaced group, e.g.::

        [it 2/3000] total_steps=4608 steps_per_second=3536 ... value_loss=5.3846 ... eta=00:00:00
          Episode_Reward: track_linear_velocity=0.0138 upright=0.0314 ...
          Episode_Termination: time_out=0.0000 fell_over=2.4762

    A scalar key is lower-cased and snake-cased with rsl_rl's ``Mean`` dropped (every one of them
    is a mean). A value containing whitespace is double-quoted, so each ``key=value`` stays one
    whitespace-delimited field: rsl_rl 5 renders the elapsed time and the ETA as a
    ``datetime.timedelta``, which is ``1 day, 1:01:01`` past a day, and a run name may hold
    spaces. A line that is not a metric -- anything else printed during the call -- passes
    through verbatim after the block, so nothing is lost; text without the iteration header is
    returned unchanged, which is what an rsl_rl release with a different console layout gets.

    Parameters
    ----------
    text : str
        Everything printed to stdout during one ``Logger.log`` call.

    Returns
    -------
    str
        The compact rendering, newline-terminated, or ``text`` itself if it has no header.
    """
    header = _ITERATION_HEADER.search(text)
    if header is None:
        return text
    scalars = [f"[it {header.group(1)}/{header.group(2)}]"]
    groups: dict[str, list[str]] = {}
    passthrough: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or set(stripped) <= {"#", "-"} or _ITERATION_HEADER.search(line):
            continue
        match = _METRIC_LINE.match(line)
        if match is None or len(match["lead"]) + len(match["key"]) < _RSL_RL_PAD - 1:
            passthrough.append(line)
            continue
        key, value = match["key"].strip(), match["value"] or ""
        if any(char.isspace() for char in value):
            value = '"' + value.replace('"', '\\"') + '"'
        if "/" in key:
            group, name = key.split("/", 1)
            groups.setdefault(group, []).append(f"{name}={value}")
        else:
            scalars.append(f"{key.lower().removeprefix('mean ').replace(' ', '_')}={value}")
    lines = [" ".join(scalars)]
    lines += [f"  {group}: {' '.join(items)}" for group, items in groups.items()]
    lines += passthrough
    return "\n".join(lines) + "\n"


class _ThreadStdoutCapture:
    """A ``sys.stdout`` stand-in that buffers one thread's writes and forwards every other thread's.

    ``sys.stdout`` is process-global, so a plain ``contextlib.redirect_stdout`` around rsl_rl's
    ``Logger.log`` would also swallow whatever another thread printed meanwhile (a viewer, a
    device bridge) and hand it to :func:`compact_iteration_log` as if it were part of the block.
    Only the thread that entered the capture is buffered; the others reach the real stream at
    once. Every other attribute (``encoding``, ``fileno``, the binary ``buffer``) is the real
    stream's.
    """

    def __init__(self, stream: Any) -> None:
        self._stream = stream
        self._owner = threading.get_ident()
        self._captured = io.StringIO()

    @property
    def captured(self) -> str:
        """Everything the owning thread wrote."""
        return self._captured.getvalue()

    def write(self, text: str) -> int:
        if threading.get_ident() == self._owner:
            return self._captured.write(text)
        return self._stream.write(text)

    def flush(self) -> None:
        if threading.get_ident() != self._owner:
            self._stream.flush()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)


class AgentIterationLogMixin:
    """Print rsl_rl's per-iteration console block through :func:`compact_iteration_log` under the agent theme.

    The block is the one part of a training run's output that grows with its length -- a
    thousand iterations of it is over half a million tokens for an agent that reads the log --
    and rsl_rl builds it inside ``Logger.log`` with a bare ``print``. So this captures the calling
    thread's stdout around that call (:class:`_ThreadStdoutCapture`) and prints the compact
    rendering instead, the same way :class:`ProgressFileMixin` hooks the call without editing the
    third-party loop. The human theme keeps rsl_rl's layout untouched. Eden's own log records are
    unaffected by the capture: their handler holds the stream it was built with, not
    ``sys.stdout``.
    """

    def learn(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        logger = getattr(self, "logger", None)
        if en.logger.theme != "agent" or logger is None or not callable(getattr(logger, "log", None)):
            return super().learn(*args, **kwargs)

        inner_log = logger.log

        def compact_log(*log_args, **log_kwargs):
            capture = _ThreadStdoutCapture(sys.stdout)
            try:
                with contextlib.redirect_stdout(capture):
                    return inner_log(*log_args, **log_kwargs)
            finally:
                # In `finally`, so what was printed before a raise still reaches the console.
                if text := capture.captured:
                    sys.stdout.write(compact_iteration_log(text))
                    sys.stdout.flush()

        logger.log = compact_log
        try:
            return super().learn(*args, **kwargs)
        finally:
            logger.log = inner_log


class CheckpointRetentionMixin:
    """Keep only the run's newest ``keep_last_checkpoints`` ``model_<iteration>.pt`` files (``None`` keeps them all).

    rsl_rl saves into the run directory every ``save_interval`` iterations and once more at the end, so a long run at
    a short interval fills a disk quota (27k iterations at 100 is 270 checkpoints). Each save deletes the older ones,
    so the newest is always on disk for ``--resume``. A save to a path that is not a ``model_<iteration>.pt`` (an
    export, say) prunes nothing.

    **Uploads.** With a wandb or neptune logger, rsl_rl hands each checkpoint to an uploader that reads the file in the
    background (wandb symlinks it into its run directory), so deleting it right after its save could lose the upload.
    The last save comes seconds after the one before it at the end of ``learn``. So while uploading, one checkpoint more
    is kept during training (each is deleted a full save interval after the next one began uploading), and the extra
    one goes once ``learn`` returns with the uploads done: when wandb's run has finished (rsl_rl's ``learn`` finishes
    it on rsl-rl-lib >= 4). Otherwise, as under rsl-rl-lib 3 or with neptune, that one extra checkpoint stays.
    """

    _last_saved_checkpoint: str | None = None
    _reported_later_checkpoints: bool = False

    def save(self, path: str, infos: dict | None = None) -> None:
        super().save(path, infos)  # type: ignore[misc]
        self._last_saved_checkpoint = path
        self._prune_checkpoints(path, in_flight=1 if self._uploads_checkpoints() else 0)

    def learn(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        result = super().learn(*args, **kwargs)  # type: ignore[misc]
        if self._last_saved_checkpoint is not None and not self._upload_pending():
            self._prune_checkpoints(self._last_saved_checkpoint, in_flight=0)
        return result

    def _uploader(self) -> str | None:
        """``"wandb"`` / ``"neptune"`` when the run's logger uploads checkpoints, else ``None``.

        rsl_rl takes the logger as a name (``"wandb"``) or, from rsl-rl-lib 5.5, as ``{"class_name": "WandbLogWriter",
        ...}``; both are read, so the upload grace is not silently lost to the newer form.
        """
        logger = getattr(self, "cfg", {}).get("logger")
        name = str(logger.get("class_name", "") if isinstance(logger, dict) else logger).lower()
        return next((service for service in ("wandb", "neptune") if service in name), None)

    def _uploads_checkpoints(self) -> bool:
        return self._uploader() is not None

    def _upload_pending(self) -> bool:
        """Whether a checkpoint may still be uploading: always with neptune, while wandb's run is live."""
        uploader = self._uploader()
        if uploader != "wandb":
            return uploader is not None
        wandb = sys.modules.get("wandb")
        return wandb is not None and wandb.run is not None

    def _prune_checkpoints(self, saved: str, in_flight: int) -> None:
        """Keep ``saved`` and the newest ``keep_last_checkpoints - 1`` (+ ``in_flight``) checkpoints before it.

        Only checkpoints of an *earlier* iteration than ``saved`` are candidates. A later one is left from another run
        in a reused log dir, or from the timeline a rollback resume abandoned: ranking it with this run's checkpoints
        would keep it and delete the checkpoint just written. It is not deleted either (it is not this run's to
        delete), but it is reported once, because ``find_latest_checkpoint`` picks it over this run's.
        """
        keep = getattr(self, "cfg", {}).get("keep_last_checkpoints")
        log_dir, name = os.path.split(saved)
        checkpoints = model_checkpoints(log_dir)
        iteration = next((it for it, filename in checkpoints if filename == name), None)
        if keep is None or iteration is None:
            return
        earlier = [filename for it, filename in checkpoints if it < iteration]
        for old in earlier[: max(len(earlier) - (keep - 1 + in_flight), 0)]:
            os.remove(os.path.join(log_dir, old))
        later = [filename for it, filename in checkpoints if it > iteration]
        if later and not self._reported_later_checkpoints:
            self._reported_later_checkpoints = True
            en.logger.warning(
                f"{log_dir} holds checkpoints of a later iteration than this run's {name} ({', '.join(later)}), left "
                "from another run in the same directory or from before a rollback. They are not deleted, and a resume "
                "that takes the newest checkpoint there (`eden train --resume`) will pick them over this run's."
            )


#: What Eden adds to every rsl_rl runner it builds, in MRO order: ``progress.json``, the agent log theme and checkpoint
#: retention. A runner for another rsl_rl-based learner (FlashSAC's ``OffPolicyRunner``) composes the same tuple, so a
#: feature added here reaches every learner. :class:`MotionWeightCurriculumMixin` is not in it: it steps the curriculum
#: once per PPO iteration (a rollout of ``num_steps_per_env`` steps), which is not what an off-policy iteration is.
EDEN_RUNNER_MIXINS = (ProgressFileMixin, AgentIterationLogMixin, CheckpointRetentionMixin)


class MimicOnPolicyRunner(*EDEN_RUNNER_MIXINS, MotionWeightCurriculumMixin, OnPolicyRunner):
    """``OnPolicyRunner`` plus the per-iter motion-weight curriculum hook for mimic tasks."""


class MimicDistillationRunner(*EDEN_RUNNER_MIXINS, MotionWeightCurriculumMixin, DistillationRunner):
    """``DistillationRunner`` plus the per-iter motion-weight curriculum hook for mimic tasks."""


class DDPOnPolicyRunner(*EDEN_RUNNER_MIXINS, MotionWeightCurriculumMixin, DDPRunnerMixin, OnPolicyRunner):
    """Drop-in replacement for :class:`OnPolicyRunner` for Eden's single-GPU-per-process setup.

    Expects the NCCL process group to be **already initialized** by
    ``eden.init()``.  The device is always ``cuda:0`` because
    ``CUDA_VISIBLE_DEVICES`` restricts each process to one physical GPU.

    Usage
    -----
    ``torchrun --standalone --nproc_per_node=NUM_GPUS my_script.py``
    """


class DDPDistillationRunner(*EDEN_RUNNER_MIXINS, MotionWeightCurriculumMixin, DDPRunnerMixin, DistillationRunner):
    """Drop-in replacement for :class:`DistillationRunner` for Eden's single-GPU-per-process setup.

    Expects the NCCL process group to be **already initialized** by
    ``eden.init()``.  The device is always ``cuda:0`` because
    ``CUDA_VISIBLE_DEVICES`` restricts each process to one physical GPU.

    Usage
    -----
    ``torchrun --standalone --nproc_per_node=NUM_GPUS my_script.py``
    """


# The runner-dict heads that can carry a Gaussian distribution, in the order
# :func:`_detect_std_type` consults them. ``teacher`` precedes ``student``: a legacy
# checkpoint's ``std`` only ever lands in the teacher (``Distillation.load`` feeds it the
# RL ``actor_state_dict``), so its parameterisation is the one that must match.
_STD_TYPE_HEADS = ("actor", "teacher", "student")


def _detect_std_type(runner_dict: dict) -> str | None:
    """Find the destination Gaussian std parameterisation in a runner dict.

    Returns 'scalar', 'log', or None if unknown. Handles rsl_rl v3 / v4 / v5
    layouts produced by :func:`translate_runner_dict`, for PPO and distillation:

    - v5 (current): ``runner_dict[head]["distribution_cfg"]["std_type"]``
    - v4:           ``runner_dict[head]["noise_std_type"]`` (the v4 translator
                    pops ``distribution_cfg`` and writes the flat noise params)
    - v3:           ``runner_dict["policy"]["noise_std_type"]``

    where ``head`` is ``actor`` (PPO), else ``teacher`` then ``student``
    (distillation). A teacher usually has no distribution — it is run
    deterministically — and then drops the converted ``std`` under
    ``strict=False``, so the student's parameterisation is a safe stand-in.
    """
    for head in _STD_TYPE_HEADS:
        model = runner_dict.get(head)
        if not isinstance(model, dict):
            continue
        dist = model.get("distribution_cfg")
        if isinstance(dist, dict) and isinstance(dist.get("std_type"), str):
            return dist["std_type"]
        if isinstance(model.get("noise_std_type"), str):
            return model["noise_std_type"]
    policy = runner_dict.get("policy")
    if isinstance(policy, dict) and isinstance(policy.get("noise_std_type"), str):
        return policy["noise_std_type"]
    return None


def _is_legacy_layout(index: dict | None) -> bool:
    """True when a checkpoint's top-level mapping is the pre-split combined ``model_state_dict``."""
    return index is not None and "actor_state_dict" not in index and "model_state_dict" in index


# What :func:`_maybe_convert_legacy_checkpoint` can map: the rsl-rl-lib 3.x ``ActorCritic``
# (MLP) ``model_state_dict``. Anything else in a legacy file — a recurrent policy's
# ``memory_*``, a 3.x ``StudentTeacher``'s ``student.*`` / ``teacher.*`` — has no destination.
_LEGACY_KEY_PREFIXES = ("actor", "actor_obs_normalizer", "critic", "critic_obs_normalizer")
_LEGACY_BARE_KEYS = ("std", "log_std")
_LEGACY_DISTILLATION_PREFIXES = ("student", "student_obs_normalizer", "teacher", "teacher_obs_normalizer")


def _refuse_unconvertible_legacy_keys(index: dict, path: str) -> None:
    """Raise when a legacy ``model_state_dict`` holds keys the converter would have to drop.

    Dropping them converts "loads" a partly random policy under ``strict=False`` — for a 3.x
    distillation file, a teacher from an empty ``actor_state_dict`` with ``teacher_loaded``
    set — so an unknown layout is refused before anything is converted or loaded.

    Parameters
    ----------
    index : dict
        The checkpoint's top-level mapping (:func:`read_checkpoint_index`), in the legacy layout.
    path : str
        The checkpoint, for the message.
    """
    counts: dict[str, int] = {}
    for key in index["model_state_dict"]:
        prefix = str(key).split(".", 1)[0]
        if str(key) not in _LEGACY_BARE_KEYS and prefix not in _LEGACY_KEY_PREFIXES:
            counts[prefix] = counts.get(prefix, 0) + 1
    if not counts:
        return
    found = ", ".join(f"{prefix}.* ({n} key{'s' if n != 1 else ''})" for prefix, n in sorted(counts.items()))
    if set(counts) <= set(_LEGACY_DISTILLATION_PREFIXES):
        raise RuntimeError(
            f"Legacy checkpoint {path} is an rsl-rl-lib 3.x distillation checkpoint (its 'model_state_dict' "
            f"holds {found}), which Eden does not convert: 3.x saved one optimizer over student, teacher and "
            f"std together, while 4.x/5.x Distillation.load restores a student-only optimizer from "
            f"'optimizer_state_dict', so a converted file could not resume. Nothing was converted or loaded. "
            f"It replays under rsl-rl-lib 3.x, which Eden's 'rsl-rl-lib>=3.0' pin admits; another 4.x/5.x "
            f"release will not read it either. Under 4.x/5.x, re-run the distillation with its teacher "
            f"bootstrapped from the RL checkpoint that trained it."
        )
    raise RuntimeError(
        f"Legacy checkpoint {path} has 'model_state_dict' keys the legacy converter does not recognise: "
        f"{found}. It maps only the rsl-rl-lib 3.x MLP actor-critic layout ("
        f"{', '.join(f'{prefix}.*' for prefix in _LEGACY_KEY_PREFIXES)}, and a bare "
        f"{' / '.join(map(repr, _LEGACY_BARE_KEYS))}). Converting anyway would drop those weights and load "
        f"a partly random policy under strict=False, so nothing was converted or loaded — and deleting the "
        f"keys to get past this check does exactly that. This is a layout the converter lacks, not an "
        f"rsl-rl-lib 4.x/5.x version to install: replay the file under the rsl-rl-lib 3.x that wrote it, "
        f"or map those keys in eden.envs.wrappers.rsl_rl_env._maybe_convert_legacy_checkpoint."
    )


def _refuse_legacy_into_a_non_mlp_model(runner, path: str) -> None:
    """Raise when the model a legacy checkpoint's actor would load into is not an ``MLPModel``.

    The converter rewrites ``actor.*`` to ``mlp.*``, the ``MLPModel`` layout. A runner whose actor (or, for a
    distillation, teacher) is another class — the RMA teacher's ``RmaActorModel`` holds ``net.*`` / ``estimator.*`` —
    would drop every converted actor key under ``strict=False`` while the critic and ``std`` still load: a randomly
    initialized policy that replays without an error. So it is refused before anything is converted or loaded.

    Parameters
    ----------
    runner : rsl_rl runner
        The split-layout runner the checkpoint is for.
    path : str
        The checkpoint, for the message.
    """
    for attr in ("actor", "teacher"):
        model = getattr(runner.alg, f"_raw_{attr}", None)
        if model is None:
            model = getattr(runner.alg, attr, None)
        if model is None:
            continue
        keys = list(model.state_dict())
        if any(key.startswith("mlp.") for key in keys):
            return
        prefixes = sorted({key.split(".", 1)[0] for key in keys})
        raise RuntimeError(
            f"Legacy checkpoint {path} is in the pre-split layout (one 'model_state_dict'), which Eden's converter "
            f"maps onto an MLPModel ('mlp.*' keys). This runner's {attr} is a {type(model).__name__} with no 'mlp.*' "
            f"keys (its state dict holds {', '.join(f'{p}.*' for p in prefixes) or 'nothing'}), so every converted "
            f"{attr} weight would be dropped under strict=False and the policy would run randomly initialized. "
            "Nothing was converted or loaded. The checkpoint was written by another model class: replay it under "
            "the rsl-rl-lib 3.x stack that trained it, or retrain it with this one."
        )


def _undetectable_std_type_error(runner_dict: dict, checkpoint: str) -> RuntimeError:
    """The error for a legacy checkpoint whose destination std parameterisation the runner dict never states."""
    present = [head for head in (*_STD_TYPE_HEADS, "policy") if isinstance(runner_dict.get(head), dict)]
    return RuntimeError(
        f"Checkpoint {checkpoint} is in the pre-split legacy layout (one 'model_state_dict'), and "
        f"converting it needs the std parameterisation of the model its actor weights load into "
        f"('actor' for PPO; 'teacher', else 'student', for distillation), which the runner dict "
        f"does not state: none of the heads {list(_STD_TYPE_HEADS)} has "
        f"distribution_cfg.std_type (rsl-rl 5.x) or noise_std_type (4.x), and no 'policy' has "
        f"noise_std_type (3.x). Heads present: {present or 'none'}; runner dict keys: "
        f"[{', '.join(sorted(map(str, runner_dict)))}]. This is not an rsl-rl-lib version mismatch, "
        f"so installing another version will not help. Either give that head a "
        f"distribution_cfg=RslRlGaussianDistributionOptions(std_type=...), or convert the file "
        f"yourself with eden.envs.wrappers.rsl_rl_env._maybe_convert_legacy_checkpoint({checkpoint!r}, "
        f"std_type='scalar' or 'log') and load the path it returns — std_type must match the "
        f"destination model, and a wrong one yields NaN actions rather than an error."
    )


def _maybe_convert_legacy_checkpoint(path: str, *, std_type: str) -> str:
    """Return ``path`` unchanged for new-format checkpoints, else convert.

    rsl_rl's current ``PPO.load`` expects top-level ``actor_state_dict`` /
    ``critic_state_dict`` keys, each shaped like the new ``MLPModel``
    state dict (``mlp.*`` / ``obs_normalizer.*`` / ``distribution.*``).
    rsl-rl-lib 3.x checkpoints hold a single combined ``model_state_dict`` with
    ``actor.*`` / ``actor_obs_normalizer.*`` / ``critic.*`` /
    ``critic_obs_normalizer.*`` / ``std`` keys. Loading those into the
    current rsl_rl raises ``KeyError: 'actor_state_dict'``. A legacy file
    with any other key (a recurrent policy's ``memory_*``, a 3.x
    distillation ``student.*`` / ``teacher.*``) raises instead of being
    converted — see :func:`_refuse_unconvertible_legacy_keys`.

    When we detect the legacy layout we write a sibling
    ``<name>.<std_type>.converted.pt`` next to the original (or to a temp file if
    that directory isn't writable) and return that path. The original
    file is never modified. The conversion is **idempotent**: if the
    sibling exists *and* post-dates the source, it's reused without
    re-loading the original's weights (only its key index is read, so an
    unconvertible file is refused even when a sibling exists) — so repeat
    ``eden inference`` calls on the same legacy file pay one full
    ``torch.load`` only on the first invocation. Touching the source ``ckpt.pt`` (e.g. re-training
    overwrites it) automatically invalidates the cached sibling and
    forces re-conversion.

    Parameters
    ----------
    path:
        Filesystem path to the checkpoint to (possibly) convert.
    std_type:
        Destination Gaussian distribution parameterisation, one of
        ``'scalar'`` (writes ``distribution.std_param``) or ``'log'``
        (writes ``distribution.log_std_param``). MUST match the
        ``RslRlGaussianDistributionOptions.std_type`` of the task whose
        policy will be loaded — passing the wrong value silently
        miscompiles the checkpoint (a ``std_param`` of size N gets
        populated with raw std values when the forward path expects log
        std, producing NaN actions on first sample). The caller,
        :func:`load_runner_checkpoint`, pulls this from the runner dict via
        :func:`_detect_std_type`, so user code does not normally need to
        think about it.
    """
    import os
    import tempfile

    if std_type not in ("scalar", "log"):
        raise ValueError(
            f"_maybe_convert_legacy_checkpoint: std_type must be 'scalar' or "
            f"'log', got {std_type!r}. Cannot safely convert the legacy "
            f"``std`` parameter without knowing the destination."
        )

    # Deciding *whether* to convert needs the key names only, so it reads the index rather
    # than the weights — this is the check every modern checkpoint pays on its way through.
    # The refusal runs before the cache lookup, so an unconvertible file is refused even when a
    # converted sibling of it already exists.
    index = read_checkpoint_index(path)
    if _is_legacy_layout(index):
        _refuse_unconvertible_legacy_keys(index, path)

    # Fast path: if the converted sibling already exists and post-dates the
    # source, skip the entire load/convert/save cycle. Saves ~50-200 ms per
    # ``eden inference`` invocation on the second+ run against the same
    # legacy checkpoint (typical iterate-on-policy-eval workflow).
    #
    # The cache filename includes ``std_type`` so the same legacy checkpoint
    # converted for a ``'scalar'`` task vs a ``'log'`` task doesn't collide:
    # the two conversions route ``std`` to *different* parameters (with a
    # log() transform in the latter), so a single cache entry would silently
    # serve the wrong one when the destination switches.
    base, ext = os.path.splitext(path)
    cached_out_path = f"{base}.{std_type}.converted{ext or '.pt'}"
    if (
        os.path.exists(cached_out_path)
        and os.path.exists(path)
        and os.path.getmtime(cached_out_path) >= os.path.getmtime(path)
    ):
        en.logger.info(f"Reusing cached converted checkpoint at {cached_out_path}.")
        return cached_out_path

    if not _is_legacy_layout(index):
        return path

    # Converting does need the values. Re-read them eagerly rather than saving out of the
    # mapping, so the output file never depends on a live mapping of the input.
    state = torch.load(path, map_location="cpu", weights_only=False)
    msd = state["model_state_dict"]
    actor_sd: dict[str, Any] = {}
    critic_sd: dict[str, Any] = {}
    for k, v in msd.items():
        if k == "std":
            # Legacy rsl_rl saved RAW std (not log-std). Map to whichever
            # parameter the current rsl_rl GaussianDistribution exposes:
            # 'scalar' learns std directly, 'log' learns log(std).
            if std_type == "scalar":
                actor_sd["distribution.std_param"] = v
            else:  # 'log'
                # Clamp keeps log(0) out of the picture for any ill-formed
                # legacy checkpoints.
                actor_sd["distribution.log_std_param"] = torch.log(v.clamp(min=1e-6))
        elif k == "log_std":
            # Defensive: rsl_rl variants that stored log(std) directly.
            if std_type == "log":
                actor_sd["distribution.log_std_param"] = v
            else:  # 'scalar'
                actor_sd["distribution.std_param"] = torch.exp(v)
        elif k.startswith("actor_obs_normalizer."):
            actor_sd["obs_normalizer." + k[len("actor_obs_normalizer.") :]] = v
        elif k.startswith("actor."):
            actor_sd["mlp." + k[len("actor.") :]] = v
        elif k.startswith("critic_obs_normalizer."):
            critic_sd["obs_normalizer." + k[len("critic_obs_normalizer.") :]] = v
        elif k.startswith("critic."):
            critic_sd["mlp." + k[len("critic.") :]] = v

    converted = {
        "actor_state_dict": actor_sd,
        "critic_state_dict": critic_sd,
        "optimizer_state_dict": state.get("optimizer_state_dict"),
        "iter": state.get("iter", 0),
        "infos": state.get("infos"),
    }

    out_path = cached_out_path
    try:
        torch.save(converted, out_path)
    except OSError:
        out_path = tempfile.NamedTemporaryFile(suffix=".converted.pt", delete=False).name
        torch.save(converted, out_path)

    en.logger.info(
        f"Converted legacy rsl_rl checkpoint {path} -> {out_path} "
        f"(model_state_dict -> actor_state_dict/critic_state_dict, std_type={std_type!r})"
    )
    return out_path


def _audited_heads(runner, ckpt: dict) -> list[tuple[str, str]]:
    """The ``(alg attribute, checkpoint key)`` pairs ``runner.load`` will actually fill from ``ckpt``.

    PPO loads ``actor`` and ``critic`` by name. Distillation mirrors rsl_rl's
    ``Distillation.load`` (identical across 4.x/5.x): a checkpoint from RL training
    (any ``actor_state_dict`` key) loads only the teacher, from the RL actor, and
    leaves the student at its init by design; anything else loads the student by name
    and the teacher from ``teacher_state_dict``, falling back to ``actor_state_dict``.
    """
    if getattr(runner.alg, "student", None) is None:
        return [("actor", "actor_state_dict"), ("critic", "critic_state_dict")]
    teacher = ("teacher", "teacher_state_dict" if ckpt.get("teacher_state_dict") else "actor_state_dict")
    if any("actor_state_dict" in key for key in ckpt):
        en.logger.info(
            "[rsl_rl] RL checkpoint into a distillation runner: the teacher loads "
            f"'{teacher[1]}'; the student keeps its initialisation."
        )
        return [teacher]
    return [("student", "student_state_dict"), teacher]


def _audit_rsl_rl_checkpoint_keys(runner, checkpoint_path: str, device) -> list[str] | None:
    """Log a summary of key mismatches between checkpoint and runner models.

    rsl_rl's ``runner.load`` uses ``load_state_dict(..., strict=False)``,
    which silently skips keys present in the model but absent in the
    checkpoint (and ignores keys present only in the checkpoint). When
    loading converted legacy checkpoints — or any checkpoint produced by a
    slightly-different model spec — this can leave parts of the policy
    randomly initialized and the user has no visible signal until the
    robot starts behaving oddly. We audit before the load and surface
    ``missing`` / ``unexpected`` counts via ``en.logger`` so the silent
    no-load failure mode becomes observable.

    No effect on load behaviour — purely diagnostic. The runner-side load
    happens immediately after; if the audit fails to find a head it logs
    a warning and skips that side.

    Returns
    -------
    list[str] | None
        The checkpoint's top-level key *names*, so a failing load downstream can describe
        them without reading the file again, or ``None`` if it isn't a readable
        dict. Names, not the mapping: the caller holds this across ``runner.load``, and
        keeping the weights alive there would double the load's peak memory to serve an
        error path that only ever prints key names.
    """
    try:
        # Key-name diff only — never materialize tensors, here or on the target device.
        ckpt = read_checkpoint_index(checkpoint_path)
    except Exception as exc:  # noqa: BLE001 - the audit is diagnostic; the real load reports its own failure
        en.logger.warning(f"[rsl_rl] Checkpoint audit could not load {checkpoint_path}: {exc}")
        return None

    if ckpt is None:
        return None

    for side, ckpt_key in _audited_heads(runner, ckpt):
        if ckpt_key not in ckpt:
            en.logger.warning(f"[rsl_rl] Checkpoint has no '{ckpt_key}' — {side} will use random init.")
            continue
        # Prefer the raw module: under torch_compile_mode ``alg.actor`` is the
        # compiled OptimizedModule whose keys carry an ``_orig_mod.`` prefix,
        # which would make every key report as missing+unexpected.
        model = getattr(runner.alg, f"_raw_{side}", None) or getattr(runner.alg, side, None)
        if model is None:
            continue
        model_keys = set(model.state_dict().keys())
        ckpt_keys = set(ckpt[ckpt_key].keys())
        if ckpt_key != f"{side}_state_dict" and not any(k.startswith("distribution.") for k in model_keys):
            # A teacher bootstrapped from an RL actor: the actor's action noise has
            # nowhere to go on a deterministic teacher, and dropping it is the design.
            ckpt_keys = {k for k in ckpt_keys if not k.startswith("distribution.")}
        missing = sorted(model_keys - ckpt_keys)
        unexpected = sorted(ckpt_keys - model_keys)
        if missing or unexpected:
            en.logger.warning(
                f"[rsl_rl] {side} checkpoint key audit — "
                f"missing in ckpt: {len(missing)}, unexpected in ckpt: {len(unexpected)}. "
                f"With strict=False these are silently ignored — "
                f"missing buffers/weights will stay at their random init. "
                f"Sample missing: {missing[:3]}; sample unexpected: {unexpected[:3]}."
            )
        else:
            en.logger.info(f"[rsl_rl] {side} checkpoint keys match model state_dict exactly.")

    return sorted(map(str, ckpt))


def _runner_uses_split_policy(runner) -> bool:
    """True when the runner's alg uses the split-module layout (rsl-rl-lib 4.x / 5.x).

    Version map (the ``rsl-rl-lib>=3.0`` pin spans both): **3.x** algs hold a
    single ``alg.policy`` module and save/load a native ``model_state_dict``
    (``load()`` has no ``strict``); the **later 4.x / 5.x** releases split the
    modules — PPO into ``alg.actor``/``alg.critic``, Distillation into
    ``alg.student``/``alg.teacher`` — save ``actor_state_dict``/... keys, and
    their ``load()`` accepts ``strict``. The legacy-checkpoint conversion and
    key audit only make sense on the split layout; "legacy"-converting a
    native 3.x checkpoint would break it. The probe is structural (instance
    attributes, not package metadata) because the layout boundary is invisible
    to a major-version check and varies per algorithm class.
    """
    return getattr(runner.alg, "actor", None) is not None or getattr(runner.alg, "student", None) is not None


# The one statement of where rsl-rl-lib's checkpoint layout splits. Spliced into both the
# drift warning and the format error below: two hand-written copies of this sentence would
# drift apart the first time either is edited.
RSL_RL_LAYOUT_SPLIT = (
    "rsl-rl-lib 3.x saves a single 'model_state_dict'; 4.x/5.x split it into 'actor_state_dict' + 'critic_state_dict'"
)


def _installed_rsl_rl_version() -> str | None:
    """The installed ``rsl-rl-lib`` version, or ``None`` if the metadata is unavailable.

    Deliberately swallows everything: this only feeds diagnostics, and a diagnostic must
    never be the reason a checkpoint that would otherwise load fails to.
    """
    try:
        return metadata.version("rsl-rl-lib")
    except Exception:  # noqa: BLE001 - provenance is advisory; never break a working load
        return None


def _training_rsl_rl_version(checkpoint: str) -> str | None:
    """The ``rsl-rl-lib`` version that wrote ``checkpoint``, from its run's ``run.json``.

    Parameters
    ----------
    checkpoint : str
        Path to the checkpoint file. Its training run wrote ``run.json`` beside it.

    Returns
    -------
    str | None
        The recorded version, or ``None`` when unknown — a run predating the field, a
        checkpoint copied out of its run directory, or a run trained with another backend.
        Never fabricated.
    """
    manifest = read_run_manifest(os.path.dirname(checkpoint))
    rl_library = (manifest or {}).get("rl_library")
    if not isinstance(rl_library, dict) or rl_library.get("distribution") != "rsl-rl-lib":
        return None
    version = rl_library.get("version")
    return version if isinstance(version, str) else None


def _warn_on_rsl_rl_version_drift(checkpoint: str) -> None:
    """Warn when ``checkpoint`` was trained under a different ``rsl-rl-lib`` than the one installed.

    The counterpart of the eden/genesis provenance warnings in ``EdenConfig.load_from_data``:
    those cover the *task*, this covers the library that owns the checkpoint's on-disk layout.
    Advisory — the load proceeds, since most drift is harmless and a format-level mismatch
    raises on its own (:func:`_rsl_rl_checkpoint_format_error`).

    Parameters
    ----------
    checkpoint : str
        Path to the checkpoint about to be loaded.
    """
    # A path that isn't there has no provenance to report. Its directory's ``run.json``
    # describes some *other* checkpoint of that run, so "was trained with X" would be a
    # claim about a file that does not exist, and "Loading anyway" doubly wrong — nothing
    # loads. The FileNotFoundError the read below raises is the useful error; the most
    # likely cause is a mistyped path, and a version warning in front of it points away
    # from that. Stay quiet and let it through.
    if not os.path.exists(checkpoint):
        return
    trained_with = _training_rsl_rl_version(checkpoint)
    installed = _installed_rsl_rl_version()
    if trained_with is None or installed is None or trained_with == installed:
        return
    warn_once(
        f"rsl-rl-lib version mismatch: {checkpoint} was trained with {trained_with}, "
        f"{installed} is installed. Loading anyway — but {RSL_RL_LAYOUT_SPLIT}, so a drift "
        f"across *that* line will not replay."
    )


def _rsl_rl_checkpoint_format_error(
    exc: KeyError, *, source: str, loaded: str, keys: list[str] | None
) -> RuntimeError | None:
    """Turn a bare ``KeyError`` out of ``runner.load`` into a version-naming error, or ``None``.

    ``rsl-rl-lib`` reads its checkpoint dict by key and lets a missing one escape verbatim
    (``KeyError: 'model_state_dict'``), naming neither the file, nor the version that wrote
    it, nor the version reading it — the whole diagnosis of a cross-machine replay. This
    rebuilds the message from the checkpoint's actual top-level keys plus both versions.

    Parameters
    ----------
    exc : KeyError
        The exception ``runner.load`` raised.
    source : str
        The checkpoint the *caller* named. This is what the message leads with: naming a
        ``.converted.pt`` sibling the user never typed reads as a different bug.
    loaded : str
        The file actually handed to ``runner.load`` — ``source`` unless the legacy
        converter rewrote it. Called out separately when the two differ, since then the
        keys below are the sibling's, not ``source``'s.
    keys : list[str] | None
        ``loaded``'s top-level key names when the caller already has them (the key audit
        reads the same file moments earlier), else ``None`` to read them here.

    Returns
    -------
    RuntimeError | None
        The replacement error, or ``None`` to re-raise ``exc`` unchanged — which is the
        answer for anything but a top-level key the checkpoint genuinely lacks. A
        ``KeyError`` from deeper inside ``load`` is someone else's bug and must not be
        dressed up as a version mismatch.
    """
    missing = exc.args[0] if len(exc.args) == 1 and isinstance(exc.args[0], str) else None
    if missing is None:
        return None
    if keys is None:
        try:
            index = read_checkpoint_index(loaded)
        except Exception:  # noqa: BLE001 - unreadable checkpoint: let the original KeyError stand
            return None
        if index is None:
            return None
        keys = sorted(map(str, index))
    if missing in keys:
        return None

    # `_maybe_convert_legacy_checkpoint` falls back to a temp file when the checkpoint's own
    # directory is read-only, so the rewritten file is not always literally a sibling.
    via = "" if loaded == source else f" (loaded via the converted copy at {loaded})"
    trained_with = _training_rsl_rl_version(source)
    provenance = (
        f"was written by rsl-rl-lib {trained_with}"
        if trained_with
        else "records no rsl-rl-lib version in its run.json (an older run, or a checkpoint "
        "copied out of its run directory)"
    )
    return RuntimeError(
        f"Checkpoint {source}{via} {provenance} and has top-level keys "
        f"[{', '.join(keys)}], but the installed rsl-rl-lib "
        f"(version {_installed_rsl_rl_version() or 'unknown'}) expects {missing!r}. "
        f"{RSL_RL_LAYOUT_SPLIT}, and Eden's 'rsl-rl-lib>=3.0' pin admits both — so install "
        f"a version matching the one that trained this checkpoint to replay it."
    )


def load_runner_checkpoint(runner, checkpoint: str) -> None:
    """``runner.load`` across rsl-rl-lib versions (the ``>=3.0`` pin spans both layouts).

    On split-module rsl_rl (4.x/5.x — see :func:`_runner_uses_split_policy`)
    the checkpoint is first routed through the legacy converter (``std``/key
    remapping) and the pre-load key audit; single-module 3.x loads its native
    ``model_state_dict`` directly. ``strict=False`` is passed only where
    ``load()`` supports it (the split era); 3.x's ``load`` has no ``strict``
    and surfaces mismatches itself.

    Parameters
    ----------
    runner : rsl_rl runner (OnPolicyRunner/DistillationRunner or subclass)
        Target runner; its ``cfg`` (the translated runner dict it was built
        from) drives std-type detection and its ``device`` the map location.
    checkpoint : str
        Path to the checkpoint file. Never modified in place — legacy
        conversion writes a ``.<std_type>.converted.pt`` sibling, and ``checkpoint``
        keeps naming the caller's file so diagnostics can say which is which.

    Raises
    ------
    RuntimeError
        When the checkpoint's top-level layout is not the one the installed
        rsl-rl-lib reads — re-raised from rsl_rl's bare ``KeyError`` with both
        versions and the keys actually present. Also, before anything
        loads, when a legacy file holds keys the converter cannot map, or
        into a runner whose actor is not an ``MLPModel``, or when the runner
        dict states no std_type to convert it with.
    """
    device = runner.device
    _warn_on_rsl_rl_version_drift(checkpoint)
    # What rsl_rl is actually handed, which the legacy converter may redirect to a rewritten
    # copy, and the key names the audit read off it — both flow into the error message below.
    load_path, loaded_keys = checkpoint, None
    if _runner_uses_split_policy(runner):
        std_type = _detect_std_type(runner.cfg)
        index = read_checkpoint_index(checkpoint)
        if _is_legacy_layout(index):
            # An unconvertible layout is the more basic cause; report it first.
            _refuse_unconvertible_legacy_keys(index, checkpoint)
            _refuse_legacy_into_a_non_mlp_model(runner, checkpoint)
        if std_type is None:
            # Can't safely guess — a wrong guess silently mis-maps ``std``. A modern
            # checkpoint needs no conversion; a legacy one cannot load unconverted.
            if _is_legacy_layout(index):
                raise _undetectable_std_type_error(runner.cfg, checkpoint)
        else:
            load_path = _maybe_convert_legacy_checkpoint(checkpoint, std_type=std_type)
        # Pre-load diagnostic: surface silent key drift that ``strict=False``
        # would otherwise swallow. Purely informational; load proceeds either
        # way so a mismatched key still results in a partially-loaded policy
        # (just one the user knows about).
        loaded_keys = _audit_rsl_rl_checkpoint_keys(runner, load_path, device)
    try:
        if "strict" in inspect.signature(runner.load).parameters:
            runner.load(load_path, map_location=device, strict=False)
        else:
            runner.load(load_path, map_location=device)
    except KeyError as exc:
        error = _rsl_rl_checkpoint_format_error(exc, source=checkpoint, loaded=load_path, keys=loaded_keys)
        if error is None:
            raise
        raise error from exc


#: What ``eden train`` / ``eden inference`` call on a runner; a runner named by import path must provide all of them.
_RUNNER_INTERFACE = ("learn", "load", "get_inference_policy")

#: The bare runner names Eden maps to its own runners; anything else is named by import path.
STOCK_RUNNER_NAMES = ("OnPolicyRunner", "DistillationRunner")


def import_class_path(class_name: str, field: str) -> tuple[Any, Any]:
    """Import the module of a ``"module.path:Name"`` class name; return ``(module, its Name or None)``.

    Parameters
    ----------
    class_name : str
        The import path.
    field : str
        The option it came from (``runner_options.class_name``), for the message.

    Raises
    ------
    ValueError
        When the named module (or a parent package) does not exist. An ``ImportError`` raised *inside* a module that
        exists — a missing optional backend, whose message says how to install it — propagates unchanged.
    """
    import importlib

    module_path, attr = class_name.rsplit(":", 1)
    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as exc:
        # Only a miss on the named module (or a parent package) is this caller's typo; a module that exists but
        # imports something absent raised its own message, which names the missing dependency.
        if exc.name is None or not (module_path == exc.name or module_path.startswith(exc.name + ".")):
            raise
        raise ValueError(f"{field}={class_name!r}: no module {module_path!r}.") from exc
    return module, getattr(module, attr, None)


def rsl_rl_runner_class(class_name: str) -> type:
    """The runner class :func:`make_rsl_rl_runner` builds for a runner options' ``class_name``.

    A stock name (``"OnPolicyRunner"``, ``"DistillationRunner"``) maps to Eden's runner for it — the DDP variant when
    the process is one rank of a distributed launch. An import path (``"module.path:ClassName"``) names a runner that
    a learning package ships (:mod:`eden.learning.wrappers`) and is built as-is, as ``cls(env, runner_dict, log_dir=,
    device=)``: it owns its loop, so it composes the Eden runner mixins it wants itself. It need not subclass rsl_rl's
    ``OnPolicyRunner`` (a fork's runner does not), only provide ``learn`` / ``load`` / ``get_inference_policy``; under
    a distributed launch it must compose :class:`DDPRunnerMixin`, or it is refused.

    ``eden train`` calls this before building the env, so a runner whose backend is not installed fails in seconds
    rather than after the scene build.

    Raises
    ------
    ValueError
        When the name is neither a stock name nor an import path, when an import path names a module or attribute that
        does not exist or a class without the runner interface, or when the process is one rank of a distributed
        launch and the runner does not compose :class:`DDPRunnerMixin`. An ``ImportError`` raised *inside* the named
        module — a missing optional backend, whose message says how to install it — propagates unchanged.
    """
    if ":" not in class_name:
        if class_name == "DistillationRunner":
            return DDPDistillationRunner if eden_dist.is_distributed() else MimicDistillationRunner
        if class_name == "OnPolicyRunner":
            return DDPOnPolicyRunner if eden_dist.is_distributed() else MimicOnPolicyRunner
        dotted = (
            f" Write an import path with a colon before the class: {':'.join(class_name.rsplit('.', 1))!r}."
            if "." in class_name
            else ""
        )
        raise ValueError(
            f"runner_options.class_name={class_name!r} is neither a stock runner name "
            f"({', '.join(map(repr, STOCK_RUNNER_NAMES))}) nor an import path 'module.path:ClassName'.{dotted}"
        )

    module, runner_cls = import_class_path(class_name, "runner_options.class_name")
    missing = [name for name in _RUNNER_INTERFACE if not callable(getattr(runner_cls, name, None))]
    if not isinstance(runner_cls, type) or missing:
        defined = sorted(
            name
            for name, obj in vars(module).items()
            if isinstance(obj, type)
            and obj.__module__ == module.__name__
            and all(callable(getattr(obj, method, None)) for method in _RUNNER_INTERFACE)
        )
        found = (
            "no such attribute"
            if runner_cls is None
            else f"{runner_cls!r} lacks the runner interface ({', '.join(missing) or 'not a class'})"
        )
        raise ValueError(
            f"runner_options.class_name={class_name!r}: {found}. Runner classes defined in {module.__name__!r}: "
            f"{defined or 'none'}."
        )
    if eden_dist.is_distributed() and not issubclass(runner_cls, DDPRunnerMixin):
        raise ValueError(
            f"runner_options.class_name={class_name!r} cannot train under a distributed launch: "
            f"{runner_cls.__name__} does not compose eden.envs.wrappers.rsl_rl_env.DDPRunnerMixin, so nothing would "
            "broadcast its parameters or average its gradients and every rank would train its own copy. Launch it "
            "as one process (without torchrun)."
        )
    return runner_cls


def make_rsl_rl_runner(
    env: RslRlVecEnvWrapper,
    checkpoint: str | None = None,
    log_dir: str | None = None,
    *,
    for_inference: bool = False,
):
    """Build an rsl_rl runner for the given environment, optionally loading a checkpoint.

    Parameters
    ----------
    env: RslRlVecEnvWrapper
        The environment to create a runner for.
    checkpoint: str | None
        The checkpoint to load.
    log_dir: str | None
        The directory to save the logs to.
    for_inference: bool
        Build the runner only to replay ``checkpoint``: from :attr:`RslRlVecEnvWrapper.inference_runner_dict`, so a
        training-only algorithm extension (AMP-PPO) is not built. Never pass it for a runner that will ``learn``.

    Returns
    -------
    runner
        The runner :func:`rsl_rl_runner_class` names for ``runner_options.class_name``.
    """
    log_dir = log_dir if log_dir is not None else en.log_dir

    # A fresh copy per runner (see `RslRlVecEnvWrapper.runner_dict`), which the runner consumes
    runner_dict = env.inference_runner_dict if for_inference else env.runner_dict
    runner_cls = rsl_rl_runner_class(runner_dict["class_name"])
    runner = runner_cls(env, runner_dict, log_dir=log_dir, device=runner_dict["device"])

    if checkpoint:
        load_runner_checkpoint(runner, checkpoint)

    return runner
