"""Runner options for FlashSAC (github.com/Holiday-Robot/rsl_rl_flashsac): off-policy SAC with SimBa-style residual
networks, a distributional critic and a large GPU replay buffer.

Declarative only — importing this module pulls in neither ``rsl_rl_flashsac`` nor rsl_rl, so a task config that uses
it imports anywhere. Training needs the ``flashsac`` extra
(``uv sync --inexact --extra flashsac --group flashsac-git``; rsl-rl-lib >= 5.4).
The class names are the ones ``rsl_rl_flashsac`` resolves (its own modules first, then rsl_rl's).
"""

from __future__ import annotations

from typing import Literal

from pydantic import model_validator

from eden.options.learning.rsl_rl.rl import RslRlBaseRunnerOptions
from eden.options.options import ConfigurableOptions


class RslRlFlashSacActorOptions(ConfigurableOptions):
    """The FlashSAC actor: a tanh-squashed Gaussian on residual blocks.

    Parameters
    ----------
    class_name : str
        The model class ``rsl_rl_flashsac`` resolves.
    num_blocks : int
        Residual blocks.
    hidden_dim : int
        Block width.
    log_std_min, log_std_max : float
        Bounds of the log standard deviation.
    """

    class_name: str = "FlashSACActor"
    num_blocks: int = 2
    hidden_dim: int = 128
    log_std_min: float = -10.0
    log_std_max: float = 2.0


class RslRlFlashSacCriticOptions(ConfigurableOptions):
    """The FlashSAC critic: a distributional (categorical) double Q on residual blocks.

    Parameters
    ----------
    class_name : str
        The model class ``rsl_rl_flashsac`` resolves.
    num_blocks : int
        Residual blocks.
    hidden_dim : int
        Block width.
    num_bins : int
        Return bins.
    min_v, max_v : float
        Support of the (normalised) return.
    """

    class_name: str = "FlashSACCritic"
    num_blocks: int = 2
    hidden_dim: int = 256
    num_bins: int = 101
    min_v: float = -5.0
    max_v: float = 5.0


class RslRlFlashSacAlgorithmOptions(ConfigurableOptions):
    """FlashSAC's hyperparameters (``rsl_rl_flashsac.algorithms.FlashSAC``).

    Parameters
    ----------
    class_name : str
        The algorithm class ``rsl_rl_flashsac`` resolves.
    replay_buffer_size : int
        Transitions kept, over all envs.
    buffer_min_length : int
        Transitions before the first update (random actions until then).
    buffer_optimize_memory_usage : bool
        Store each observation once (next observations by index).
    buffer_device : str | None
        Where the buffer lives; ``None`` the training device.
    buffer_obs_dtype : str | None
        Observation storage dtype (e.g. ``"float16"``); ``None`` keeps float32.
    num_learning_epochs, num_mini_batches : int
        Gradient updates per runner iteration = their product.
    mini_batch_size : int
        Transitions per update.
    learning_rate_init, learning_rate_peak, learning_rate_end : float
        Warm-up start, peak and final learning rate.
    learning_rate_warmup_steps : int
        Updates of linear warm-up.
    learning_rate_decay_steps : int | None
        Updates of decay; ``None`` spans the run (``max_iterations`` x updates per iteration).
    actor_bc_alpha : float
        Behaviour-cloning weight on the actor loss.
    actor_noise_zeta_mu, actor_noise_zeta_max : float
        Exploration-noise schedule.
    actor_update_period : int
        Critic updates per actor update.
    critic_target_update_tau : float
        Polyak rate of the target critic.
    temp_initial_value, temp_target_sigma : float
        Entropy temperature start and target action standard deviation.
    temp_target_entropy : float | None
        Explicit target entropy; ``None`` derives it from ``temp_target_sigma``.
    gamma : float
        Discount.
    n_steps : int
        n-step return length.
    normalize_reward : bool
        Normalise rewards by a running return scale.
    normalized_G_max : float
        Return bound after normalisation (should match the critic's support).
    use_compile, use_amp : bool
        ``torch.compile`` the networks; train in mixed precision.
    compile_mode : str
        ``torch.compile`` mode.
    """

    class_name: str = "FlashSAC"
    replay_buffer_size: int = 10_000_000
    buffer_min_length: int = 100_000
    buffer_optimize_memory_usage: bool = True
    buffer_device: str | None = None
    buffer_obs_dtype: str | None = None
    num_learning_epochs: int = 1
    num_mini_batches: int = 2
    mini_batch_size: int = 2048
    learning_rate_init: float = 3.0e-4
    learning_rate_peak: float = 3.0e-4
    learning_rate_end: float = 1.5e-4
    learning_rate_warmup_steps: int = 0
    learning_rate_decay_steps: int | None = None
    actor_bc_alpha: float = 0.0
    actor_noise_zeta_mu: float = 2.0
    actor_noise_zeta_max: int = 16
    actor_update_period: int = 2
    critic_target_update_tau: float = 0.01
    temp_initial_value: float = 0.01
    temp_target_sigma: float = 0.15
    temp_target_entropy: float | None = None
    gamma: float = 0.99
    n_steps: int = 3
    normalize_reward: bool = True
    normalized_G_max: float = 5.0
    use_compile: bool = True
    compile_mode: str = "auto"
    use_amp: bool = True


class RslRlFlashSacRunnerOptions(RslRlBaseRunnerOptions):
    """``rsl_rl_flashsac``'s ``OffPolicyRunner``: each iteration steps every env ``num_steps_per_env`` times, then
    runs ``num_learning_epochs x num_mini_batches`` updates; ``eden train`` selects it from this class.

    Parameters
    ----------
    class_name : str
        The runner class.
    max_iterations : int | None
        The run's length in iterations; ``None`` when :attr:`num_env_steps` sets it. Exactly one of the two is set.
    num_env_steps : int | None
        The run's length in env steps, over all envs of every rank: ``max_iterations`` is derived from it once the env
        count and the ``torchrun`` world size are known (:meth:`for_num_envs`, which :class:`FlashSacVecEnvWrapper`
        applies), so the run follows ``--num_envs`` and the GPU count. Exactly one of the two is set.
    log_interval : int
        Iterations per log line (one iteration is one env step by default).
    start_training : int
        Iterations before the first update.
    action_bound_scale : float
        The policy's tanh output is scaled to ``[-action_bound_scale, action_bound_scale]``.
    actor : RslRlFlashSacActorOptions
        The actor.
    critic : RslRlFlashSacCriticOptions
        The critic.
    algorithm : RslRlFlashSacAlgorithmOptions
        The algorithm.
    """

    class_name: Literal["OffPolicyRunner"] = "OffPolicyRunner"
    num_steps_per_env: int = 1
    max_iterations: int | None = 50_000
    num_env_steps: int | None = None
    save_interval: int = 10_000
    log_interval: int = 20
    start_training: int = 0
    action_bound_scale: float = 1.0
    actor: RslRlFlashSacActorOptions = RslRlFlashSacActorOptions()
    critic: RslRlFlashSacCriticOptions = RslRlFlashSacCriticOptions()
    algorithm: RslRlFlashSacAlgorithmOptions = RslRlFlashSacAlgorithmOptions()

    @model_validator(mode="after")
    def _one_run_length(self) -> RslRlFlashSacRunnerOptions:
        if (self.max_iterations is None) == (self.num_env_steps is None):
            raise ValueError(
                f"RslRlFlashSacRunnerOptions takes the run's length as exactly one of max_iterations "
                f"({self.max_iterations}) and num_env_steps ({self.num_env_steps}), so neither silently wins. To set "
                "iterations on a config that sizes the run in env steps, clear the other in the same override: "
                "`--override 'runner_options: {max_iterations: 1000, num_env_steps: null}'`."
            )
        return self

    def for_num_envs(self, num_envs: int, *, world_size: int) -> RslRlFlashSacRunnerOptions:
        """These options with ``max_iterations`` sized for ``num_envs`` envs on each of ``world_size`` ranks when
        :attr:`num_env_steps` sets the run.

        Every rank runs ``max_iterations`` iterations of its own ``num_envs`` envs, so the run's env steps are split
        across the ranks rather than repeated on each. With ``algorithm.learning_rate_decay_steps=None`` the
        learning-rate decay spans the derived run too.
        """
        if self.num_env_steps is None:
            return self
        per_iteration = num_envs * self.num_steps_per_env * world_size
        return self.model_copy(
            update={"max_iterations": max(1, self.num_env_steps // per_iteration), "num_env_steps": None}
        )
