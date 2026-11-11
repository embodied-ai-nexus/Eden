"""The Eden side of FlashSAC: the VecEnv adapter and the runner factory ``eden train`` and ``eden inference`` use for
:class:`~eden.learning.wrappers.rsl_rl.flashsac.options.RslRlFlashSacRunnerOptions`.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

import functools

import torch
from tensordict import TensorDict

import eden as en
from eden.envs.wrappers.rsl_rl_env import EDEN_RUNNER_MIXINS, DDPRunnerMixin, RslRlVecEnvWrapper
from eden.learning.wrappers.rsl_rl.flashsac.options import RslRlFlashSacRunnerOptions
from eden.utils import distributed as eden_dist
from eden.utils.distribution import install_hint

from eden.utils.configs import EdenMimicConfig

if TYPE_CHECKING:
    from eden.envs.base import RLEnvBase
    from eden.utils.configs import EdenConfig


def _import_runner():
    try:
        from rsl_rl_flashsac.runners import OffPolicyRunner
    except ImportError as e:
        raise ImportError(
            "Training with RslRlFlashSacRunnerOptions needs rsl_rl_flashsac, which is not installed. It is not on "
            f"PyPI, so the `flashsac` extra alone does not bring it: install it with {install_hint('flashsac')} "
            "(it needs rsl-rl-lib >= 5.4)."
        ) from e
    return OffPolicyRunner


@functools.cache
def _runner_class(distributed: bool) -> type:
    """``rsl_rl_flashsac``'s ``OffPolicyRunner`` with what Eden adds to every rsl_rl runner (``EDEN_RUNNER_MIXINS``:
    ``progress.json``, the agent log theme, ``keep_last_checkpoints``) and, under
    ``torchrun``, :class:`DDPRunnerMixin`: Eden gives each rank one GPU (``cuda:0`` to every rank) and starts NCCL
    itself, where rsl_rl's own multi-GPU setup expects ``cuda:<local_rank>`` and starts it again.

    Built on first use, since ``rsl_rl_flashsac`` is an optional extra.
    """
    mixins = (*EDEN_RUNNER_MIXINS, DDPRunnerMixin) if distributed else EDEN_RUNNER_MIXINS
    name = "DDPFlashSacRunner" if distributed else "FlashSacRunner"
    return type(name, (*mixins, _import_runner()), {"__module__": __name__, "__qualname__": name})


def _refuse_unsupported(config: EdenConfig, *, training: bool) -> None:
    """Raise if ``config`` asks FlashSAC for what it cannot do: bootstrap time-outs without the pre-reset observation,
    or (``training``) train a motion-weight curriculum.

    Both are config facts: :meth:`FlashSacVecEnvWrapper.from_config` checks them before it builds the env, and the
    wrapper's ``__init__`` again for an env built elsewhere.
    """
    if not config.env_options.record_final_observations:
        raise ValueError(
            "FlashSAC bootstraps truncated episodes from their last observation, but the env does not record it: "
            "set `env_options.record_final_observations=True` in the task config."
        )
    adaptive = config.motion_options.adaptive_motion_terms() if isinstance(config, EdenMimicConfig) else []
    if training and adaptive:
        fields = ", ".join(f"`motion_options.{name}.adaptive_motion_weights=False`" for name in adaptive)
        raise ValueError(
            f"Motion term(s) {adaptive} mine hard motions (adaptive_motion_weights=True), but FlashSAC's runner never "
            "steps that curriculum, so their weights would stay at their initial values for the whole run. "
            f"Set {fields}, or train with PPO (RslRlOnPolicyRunnerOptions). Adding MotionWeightCurriculumMixin to the "
            "FlashSAC runner is not the fix: it steps once per runner iteration, and a FlashSAC iteration is far "
            "shorter than a PPO one (one env step for flashdexretarget, against PPO's 32-step rollout), so "
            "motion_weight_update_every_n_iters would mean something else."
        )


class FlashSacVecEnvWrapper(RslRlVecEnvWrapper):
    """:class:`RslRlVecEnvWrapper` plus what FlashSAC needs of the env.

    * The action bounds FlashSAC reads (``action_bias`` / ``action_scale``), applied here: the policy emits tanh
      actions in ``[-1, 1]`` and the env receives ``action_bias + action_scale * a``, i.e. ``[-action_bound_scale,
      action_bound_scale]``.
    * On time-outs, the observation before the reset (``extras["time_outs_obs"]``), so a truncated episode
      bootstraps from its true last state. The env must record it (``EnvOptions.record_final_observations``), or
      every truncation would bootstrap from the next episode's first observation.
    * The run sized for this env count over every rank (:meth:`RslRlFlashSacRunnerOptions.for_num_envs`): read
      ``runner_options.max_iterations`` off the wrapper, not the config.

    For training, a mimic env whose motion terms mine hard motions (``adaptive_motion_weights``) is refused: FlashSAC's
    runner does not step that curriculum.
    """

    def __init__(self, env: RLEnvBase, options: RslRlFlashSacRunnerOptions) -> None:
        if not isinstance(options, RslRlFlashSacRunnerOptions):
            raise TypeError(
                f"FlashSacVecEnvWrapper takes RslRlFlashSacRunnerOptions, got {type(options).__name__}; build a PPO "
                "runner's env with RslRlVecEnvWrapper."
            )
        _refuse_unsupported(env.config, training=env.training)
        super().__init__(env, options=options.for_num_envs(env.num_envs, world_size=eden_dist.get_world_size()))
        self.action_bias = torch.zeros(self.num_actions, device=self.device)
        self.action_scale = torch.full((self.num_actions,), float(options.action_bound_scale), device=self.device)

    @classmethod
    def from_config(
        cls, cfg, *, show_viewer: bool = False, eval_mode: bool = False, apply_seed: bool = True
    ) -> FlashSacVecEnvWrapper:
        """:meth:`RslRlVecEnvWrapper.from_config`, refusing what FlashSAC cannot do before the env is built.

        The task's own :meth:`~eden.utils.configs.EdenConfig.check_bindings` runs first: it can name the variant flag
        that sets these fields (flashdexretarget: ``--learner flashsac``).
        """
        cfg.check_bindings()
        _refuse_unsupported(cfg, training=not eval_mode)
        return super().from_config(cfg, show_viewer=show_viewer, eval_mode=eval_mode, apply_seed=apply_seed)

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        obs, rew, dones, extras = super().step(self.action_bias + self.action_scale * actions)
        final = extras.get("final_observations")
        if final is not None:
            extras["time_outs_obs"] = TensorDict(dict(final), batch_size=[self.num_envs])
        return obs, rew, dones, extras


def make_flashsac_runner(
    env: FlashSacVecEnvWrapper,
    checkpoint: str | None = None,
    log_dir: str | None = None,
    *,
    inference: bool = False,
):
    """Build ``rsl_rl_flashsac``'s ``OffPolicyRunner``, with Eden's runner features (:func:`_runner_class`), on
    ``env``, optionally loading ``checkpoint``.

    Parameters
    ----------
    env : FlashSacVecEnvWrapper
        The env.
    checkpoint : str | None
        A FlashSAC checkpoint (``model_<iter>.pt``) to load, strictly.
    log_dir : str | None
        The run directory; ``en.log_dir`` by default.
    inference : bool
        Only the policy will run (``eden inference``): the replay buffer is allocated at one row per env instead
        of ``replay_buffer_size`` (50M rows, ~154 GB, for flashdexretarget), and nothing is compiled.
    """
    runner_cls = _runner_class(eden_dist.is_distributed())
    cfg = env.runner_dict  # a fresh copy, which constructing the runner consumes (FlashSAC pops the class names)
    if inference:
        cfg["algorithm"].update(replay_buffer_size=env.num_envs, buffer_min_length=env.num_envs, use_compile=False)
    device = cfg["device"]
    runner = runner_cls(env, cfg, log_dir=log_dir or en.log_dir, device=device)
    if checkpoint:
        # FlashSAC's own load, strict: the rsl_rl PPO path's legacy conversion and key audit are for its layouts.
        runner.load(checkpoint, map_location=device, strict=True)
    return runner
