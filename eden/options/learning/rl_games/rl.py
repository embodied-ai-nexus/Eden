"""rl_games network and PPO runner configuration options."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import model_validator

from eden.options.learning.clip_actions import ClipActions
from eden.options.options import ConfigurableOptions
from eden.utils.random import Seed
from eden.utils.torch import DeviceStr


class RlGamesNetworkOptions(ConfigurableOptions):
    """Network architecture configuration for rl_games.

    Parameters
    ----------
    name: str
        The network builder name.
    separate: bool
        Whether actor and critic have separate network trunks.
    units: list[int]
        Hidden layer sizes for the MLP.
    activation: str
        Activation function name.
    mu_activation: str
        Activation for the mean output head.  ``"None"`` maps to identity.
    sigma_activation: str
        Activation for the sigma output head.  ``"None"`` maps to identity.
    fixed_sigma: bool
        Whether sigma is a fixed (learned) parameter or state-dependent.
    init_sigma_val: float
        Initial value for sigma (const initializer).
    mu_init_name: str
        Initializer name for mu weights.
    """

    name: str = "actor_critic"
    separate: bool = True
    units: list[int] = [512, 256, 128]
    activation: str = "elu"
    mu_activation: str = "None"
    sigma_activation: str = "None"
    fixed_sigma: bool = True
    init_sigma_val: float = 0.0
    mu_init_name: str = "default"


class RlGamesPpoRunnerOptions(ConfigurableOptions):
    """PPO runner configuration for rl_games.

    Mirrors the ``params`` dict expected by :class:`rl_games.torch_runner.Runner`.
    :func:`~eden.options.learning.rl_games.translate.translate_runner_dict` translates these options
    into the nested dict that ``Runner.load()`` consumes.

    Parameters
    ----------
    seed: Seed
        Random seed. Non-negative; see ``eden.utils.random.validate_seed``.
    experiment_name: str
        Name for the experiment (used in log/checkpoint dirs).
    device: str
        Torch device string.
    max_epochs: int
        Maximum number of training epochs.
    save_frequency: int
        Save checkpoint every N epochs.  0 disables periodic saving.
    keep_last_checkpoints: int | None
        How many of the run's newest periodic ``last_<name>_ep_<epoch>_rew_<reward>.pth`` checkpoints to keep on
        disk; each save deletes the older ones. Defaults to 1, the newest; ``None`` keeps every checkpoint. The
        best-reward checkpoint ``<name>.pth`` is never deleted.
    horizon_length: int
        Number of environment steps per rollout.
    mini_epochs: int
        Number of PPO optimisation passes per rollout.
    minibatch_size: int | None
        Minibatch size.  If ``None``, defaults to
        ``horizon_length * num_envs`` (full-batch), times ``len(agents)`` when ``agents`` is set.
    learning_rate: float
        Initial learning rate.
    lr_schedule: str
        Learning-rate schedule (``"adaptive"``, ``"linear"``, or ``"None"``).
    kl_threshold: float
        Target KL divergence for adaptive LR.
    gamma: float
        Discount factor.
    tau: float
        GAE lambda.
    e_clip: float
        PPO clipping parameter.
    clip_value: bool
        Whether to clip value-function loss.
    entropy_coef: float
        Entropy bonus coefficient.
    critic_coef: float
        Value-loss coefficient.
    grad_norm: float
        Maximum gradient norm for clipping.
    truncate_grads: bool
        Whether to clip gradients.
    normalize_input: bool
        Whether to normalize observations with running statistics.
    normalize_value: bool
        Whether to normalize value targets.
    normalize_advantage: bool
        Whether to normalize advantages.
    value_bootstrap: bool
        Whether to bootstrap value on time-outs.
    reward_scale: float
        Multiplicative reward scale applied by the reward shaper.
    clip_actions: float | None
        Symmetric action clipping bound: a finite positive bound, or ``None`` to disable clipping.
    bounds_loss_coef: float
        Coefficient for the action-bounds regularisation loss.
    obs_groups: dict[str, list[str]]
        Mapping from ``"actor"`` / ``"critic"`` to observation group names.
    agents: list[str]
        Names the agents of a multi-agent env, in order. Each name must be an observation
        group, a reward group and an action term of the env, and the action terms must be
        declared in this same order. rl_games then trains one shared policy over
        ``num_envs * len(agents)`` rows (``env_info["agents"] = len(agents)``), each fed its
        agent's observation group, action term and group reward. Empty (the default) is the
        single-agent wrapper. Mutually exclusive with ``obs_groups``: a multi-agent run has no
        central-value critic, and each agent's observation is its own group.
    network: RlGamesNetworkOptions
        Network architecture.
    torch_compile: bool | str | dict | None
        ``torch.compile`` configuration passed to the runner.
    """

    seed: Seed = 7100
    experiment_name: str = ""
    # "auto" resolves to ``str(gs.device)`` at runner construction time. See
    # ``eden.utils.torch.resolve_device``. ``DeviceStr`` rejects typos at
    # config load.
    device: DeviceStr = "auto"

    max_epochs: int = 30000
    save_frequency: int = 500
    keep_last_checkpoints: int | None = 1

    # Rollout / optimisation
    horizon_length: int = 24
    mini_epochs: int = 5
    minibatch_size: int | None = None
    learning_rate: float = 1.0e-3
    lr_schedule: Literal["adaptive", "linear", "None"] = "adaptive"
    kl_threshold: float = 0.01

    # PPO
    gamma: float = 0.99
    tau: float = 0.95
    e_clip: float = 0.2
    clip_value: bool = True
    entropy_coef: float = 0.01
    critic_coef: float = 1.0
    grad_norm: float = 1.0
    truncate_grads: bool = True

    # Normalisation
    normalize_input: bool = False
    normalize_value: bool = False
    normalize_advantage: bool = True
    value_bootstrap: bool = True

    # Reward / actions
    reward_scale: float = 1.0
    clip_actions: ClipActions | None = None
    bounds_loss_coef: float = 0.0

    # Observation groups (same semantics as rsl_rl)
    obs_groups: dict[str, list[str]] = {}

    # Multi-agent: one observation group + reward group + action term per agent, by name.
    agents: list[str] = []

    # Network
    network: RlGamesNetworkOptions = RlGamesNetworkOptions()

    torch_compile: bool | str | dict | None = False

    @model_validator(mode="after")
    def _validate_agents(self):
        """Refuse ``agents`` alongside ``obs_groups``, and a repeated agent name."""
        check_agents_options(self.agents, self.obs_groups)
        return self

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        if self.keep_last_checkpoints is not None and self.keep_last_checkpoints < 1:
            raise ValueError(
                f"{type(self).__name__}(keep_last_checkpoints={self.keep_last_checkpoints}) would delete the checkpoint "
                "it just saved; pass a count >= 1 (1 keeps only the newest), or None to keep every checkpoint."
            )


def check_agents_options(agents: list[str], obs_groups: dict[str, list[str]]) -> None:
    """Raise ``ValueError`` if ``agents`` repeats a name or is set together with ``obs_groups``.

    Shared by the options validator and :class:`~eden.envs.wrappers.rl_games_env.RlGamesVecEnvWrapper`,
    which re-checks because a field assigned after construction skips the validator.
    """
    if not agents:
        return
    if obs_groups:
        raise ValueError(
            f"RlGamesPpoRunnerOptions sets both `agents={agents!r}` and `obs_groups={obs_groups!r}`; "
            "they are mutually exclusive.\n"
            "In multi-agent mode each agent's observation is the observation group named after it, "
            "and there is no central-value critic to give `obs_groups['critic']` to.\n"
            "Fix: drop `obs_groups` (set it to `{}`) to train the agents, or drop `agents` (set it to `[]`) "
            "for a single-agent run."
        )
    repeated = sorted({name for name in agents if agents.count(name) > 1})
    if repeated:
        raise ValueError(
            f"RlGamesPpoRunnerOptions.agents names {repeated!r} more than once: {agents!r}.\n"
            "Each agent owns one observation group, one reward group and one action term.\n"
            "Fix: list every agent exactly once, in the action manager's term order."
        )
