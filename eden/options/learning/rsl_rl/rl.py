"""rsl_rl model, distribution, and PPO configuration options."""

from __future__ import annotations

from typing import Any, Literal

from eden.options.learning.clip_actions import ClipActions
from eden.options.learning.rsl_rl.rnd import RslRlRndOptions
from eden.options.learning.rsl_rl.symmetry import RslRlSymmetryOptions
from eden.options.options import ConfigurableOptions
from eden.utils.random import Seed
from eden.utils.torch import DeviceStr

####################################
# Distribution configurations      #
####################################


class RslRlDistributionOptions(ConfigurableOptions):
    """Base configuration for an output distribution.

    Parameters
    ----------
    class_name: str
        The distribution class name.
    """

    class_name: str = "GaussianDistribution"


class RslRlGaussianDistributionOptions(RslRlDistributionOptions):
    """Configuration for a state-independent Gaussian distribution.

    Parameters
    ----------
    class_name: str
        The distribution class name.
    init_std: float
        Initial standard deviation.
    std_type: Literal["scalar", "log"]
        Parameterisation of the standard deviation.
    std_range: array-like[float, float]
        Hard ``(min, max)`` clamp on the standard deviation. The floor is what keeps a
        policy exploring once its advantage signal flattens out -- without one, a task
        whose reward briefly stops depending on the action can drive the std to zero and
        never recover. Defaults to ``(1e-6, 1e6)``, rsl-rl's own effectively-unclamped
        range -- a concrete default rather than ``None`` because rsl-rl splats this whole
        dict into the distribution constructor (``dist_class(output_dim, **dist_cfg)``),
        where a ``None`` placeholder raises ``'NoneType' object is not iterable`` instead
        of meaning "unset".

        **rsl-rl >= 5 only.** The v3.2 and v4 translators
        (:mod:`eden.options.learning.rsl_rl.translate`) forward only ``init_std`` and
        ``std_type``, so on an older rsl-rl this field is dropped and the clamp simply
        does not apply -- set it as insurance, not as something a run depends on.
    """

    class_name: str = "GaussianDistribution"
    init_std: float = 1.0
    std_type: Literal["scalar", "log"] = "log"
    std_range: tuple[float, float] = (1e-6, 1e6)


class RslRlHeteroscedasticGaussianDistributionOptions(RslRlDistributionOptions):
    """Configuration for a state-dependent (heteroscedastic) Gaussian distribution.

    The MLP outputs both mean and standard deviation.  The ``init_std``
    value is used to initialise the bias of the std head.

    Parameters
    ----------
    class_name: str
        The distribution class name.
    init_std: float
        Initial standard deviation (used to initialise MLP std-head bias).
    std_type: Literal["scalar", "log"]
        Parameterisation of the standard deviation.
    """

    class_name: str = "HeteroscedasticGaussianDistribution"
    init_std: float = 1.0
    std_type: Literal["scalar", "log"] = "scalar"


##############################
# Model configurations       #
##############################


class RslRlModelOptions(ConfigurableOptions):
    """
    Configuration for an rsl_rl model.

    Parameters
    ----------
    class_name: str
        The rsl_rl class name. This parameter should not be overridden.
    hidden_dims: list[int]
        Hidden layer dimensions for the MLP trunk.
    activation: str
        Activation function.
    obs_normalization: bool
        Whether to normalize observations before the network.
    """

    class_name: str = "MLPModel"
    hidden_dims: list[int] = [512, 256, 128]
    activation: str = "elu"
    obs_normalization: bool = True


class RslRlModelWithDistributionOptions(RslRlModelOptions):
    """Mixin for models with a distribution.

    Parameters
    ----------
    distribution_cfg: RslRlDistributionOptions | None
        Output distribution.  ``None`` means a deterministic head.
    """

    distribution_cfg: RslRlDistributionOptions | None = RslRlGaussianDistributionOptions()


class RslRlRNNModelOptions(RslRlModelOptions):
    """Configuration for a recurrent (RNN) model."""

    class_name: str = "RNNModel"
    rnn_type: str = "gru"
    rnn_hidden_dim: int = 256
    rnn_num_layers: int = 1


class RslRlCNNModelOptions(RslRlModelOptions):
    """Configuration for a CNN-backed model."""

    class_name: str = "CNNModel"
    cnn_cfg: dict[str, dict] | dict[str, Any] | None = None


class RslRlActorRNNOptions(RslRlRNNModelOptions, RslRlModelWithDistributionOptions):
    """Recurrent policy actor."""


class RslRlActorCNNOptions(RslRlCNNModelOptions, RslRlModelWithDistributionOptions):
    """CNN policy actor."""


# Role names: an actor carries a distribution, a critic does not.
RslRlActorOptions = RslRlModelWithDistributionOptions
RslRlCriticOptions = RslRlModelOptions
RslRlCriticRNNOptions = RslRlRNNModelOptions
RslRlCriticCNNOptions = RslRlCNNModelOptions

############################
# Algorithm configurations #
############################


class RslRlPpoAlgorithmOptions(ConfigurableOptions):
    """Configuration for the PPO algorithm.

    Parameters
    ----------
    class_name: str
        The algorithm class name.
    num_learning_epochs: int
        The number of learning epochs per update.
    num_mini_batches: int
        The number of mini-batches per update.
    learning_rate: float
        The learning rate for the policy.
    schedule: str
        The learning rate schedule.
    gamma: float
        The discount factor.
    lam: float
        The lambda parameter for Generalized Advantage Estimation (GAE).
    entropy_coef: float
        The coefficient for the entropy loss.
    desired_kl: float
        The desired KL divergence.
    max_grad_norm: float
        The maximum gradient norm.
    value_loss_coef: float
        The coefficient for the value loss.
    use_clipped_value_loss: bool
        Whether to use clipped value loss.
    clip_param: float
        The clipping parameter for the policy.
    normalize_advantage_per_mini_batch: bool
        Whether to normalize the advantage per mini-batch.
    share_cnn_encoders: bool
        Whether to share CNN encoders between actor and critic.
    rnd_cfg: RslRlRndOptions | None
        The RND configuration.
    symmetry_cfg: RslRlSymmetryOptions | None
        The symmetry configuration.
    """

    class_name: str = "PPO"
    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    learning_rate: float = 1.0e-3
    schedule: str = "adaptive"
    gamma: float = 0.99
    lam: float = 0.95
    entropy_coef: float = 0.005
    desired_kl: float = 0.01
    max_grad_norm: float = 1.0
    value_loss_coef: float = 1.0
    use_clipped_value_loss: bool = True
    clip_param: float = 0.2
    normalize_advantage_per_mini_batch: bool = False
    share_cnn_encoders: bool = False
    rnd_cfg: RslRlRndOptions | None = None
    symmetry_cfg: RslRlSymmetryOptions | None = None


#########################
# Runner configurations #
#########################


class RslRlBaseRunnerOptions(ConfigurableOptions):
    """Base configuration of the runner.

    Parameters
    ----------
    seed: Seed
        The seed for the experiment. Non-negative; see ``eden.utils.random.validate_seed``.
    device: str
        The device for the rl-agent.
    num_steps_per_env: int
        The number of steps per environment per update.
    max_iterations: int
        The maximum number of iterations.
    obs_groups: dict[str, list[str]]
        A mapping from observation sets to observation groups.
        Keys are ``"actor"``, ``"critic"`` (and optionally ``"rnd_state"``).
        Values are lists of observation group names provided by the environment.
    clip_actions: float | None
        Actions are clamped to ``[-clip_actions, clip_actions]``; a finite positive bound, or ``None`` to leave
        them unclamped.
    save_interval: int
        The number of iterations between saves.
    keep_last_checkpoints: int | None
        How many of the run's newest ``model_<iteration>.pt`` checkpoints to keep on disk; each save deletes the older
        ones. Defaults to 1, the newest, which is all ``eden train --resume`` needs; ``None`` keeps every checkpoint.
        Checkpoints already uploaded to the logger (wandb) stay there.
    check_for_nan: bool
        Whether to check for NaN values in environment outputs.
    experiment_name: str
        The experiment name.
    run_name: str
        The run name.
    logger: Literal["tensorboard", "neptune", "wandb"]
        The logger to use.
    neptune_project: str
        The neptune project name.
    wandb_project: str
        The wandb project name.
    resume: bool
        Whether to resume a previous training.
    load_run: str
        The run directory to load.
    load_checkpoint: str
        The checkpoint file to load.
    """

    seed: Seed = 7100
    # "auto" resolves to ``str(gs.device)`` at runner construction time, so the
    # default tracks whatever backend Genesis was initialised with (CUDA, MPS,
    # or CPU). Explicit values (e.g. "cuda:0", "cpu") are passed through unchanged.
    # ``DeviceStr`` rejects typos like "cdua:0" at config load.
    device: DeviceStr = "auto"
    num_steps_per_env: int = 24
    max_iterations: int = 30000
    obs_groups: dict[str, list[str]] = {}
    clip_actions: ClipActions | None = None
    save_interval: int = 500
    keep_last_checkpoints: int | None = 1
    check_for_nan: bool = True
    experiment_name: str = ""
    run_name: str = ""
    logger: Literal["tensorboard", "neptune", "wandb"] = "wandb"
    neptune_project: str = "eden"
    wandb_project: str = "eden"
    resume: bool = False
    load_run: str = ".*"
    load_checkpoint: str = "model_.*.pt"

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        if self.keep_last_checkpoints is not None and self.keep_last_checkpoints < 1:
            raise ValueError(
                f"{type(self).__name__}(keep_last_checkpoints={self.keep_last_checkpoints}) would delete the checkpoint "
                "it just saved; pass a count >= 1 (1 keeps only the newest), or None to keep every checkpoint."
            )


class RslRlOnPolicyRunnerOptions(RslRlBaseRunnerOptions):
    """Configuration of the runner for on-policy algorithms.

    The runner carries separate ``actor`` and ``critic`` model configs.

    Parameters
    ----------
    class_name: str
        The runner class name.
    actor: RslRlModelWithDistributionOptions
        The actor model configuration with distribution to sample from.
    critic: RslRlModelOptions
        The critic model configuration.
    algorithm: RslRlPpoAlgorithmOptions
        The algorithm configuration.
    """

    class_name: str = "OnPolicyRunner"
    actor: RslRlModelWithDistributionOptions
    critic: RslRlModelOptions
    algorithm: RslRlPpoAlgorithmOptions
