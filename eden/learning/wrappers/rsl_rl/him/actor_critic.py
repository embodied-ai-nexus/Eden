"""HIM actor-critic: OpenHomie's ``HIMActorCritic`` (``HomieRL/rsl_rl/rsl_rl/modules/him_actor_critic.py``).

Flat ground (no terrain encoder). The actor sees only the newest frame of the history plus the
estimator's outputs — ``frame ⊕ velocity(3) ⊕ latent(32)`` — while the whole history feeds the
estimator. The critic is a plain MLP over the privileged observation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from eden.learning.wrappers.rsl_rl.him.estimator import VELOCITY_DIM, HIMEstimator, _get_activation


_NO_OBS_NORMALIZATION = (
    "HIM uses the task's fixed observation scales; set obs_normalization=False on the runner's actor and critic."
)


# What HIMActorCritic implements: MLP actor and critic sharing one activation, and a Gaussian whose std is
# a raw (``scalar``) parameter with no clamp — rsl-rl's default ``std_range`` is that "effectively unclamped".
_HIM_MODEL = "MLPModel"
_HIM_STD_TYPE = "scalar"
_UNCLAMPED_STD_RANGE = (1e-6, 1e6)


def _unsupported(field: str, value, supported, why: str) -> ValueError:
    return ValueError(
        f"HIMActorCritic does not implement {field}={value!r}; it supports only {supported!r} ({why}). Set "
        f"{field}={supported!r} in the runner options rather than relying on HIM to ignore it."
    )


def check_model_cfg(actor_cfg: dict, critic_cfg: dict) -> float:
    """Refuse the rsl-rl-lib 4 / 5 model settings HIM would otherwise ignore; return the initial action std.

    Reads both formats Eden hands the runner: 5's ``distribution_cfg`` and 4's flattened
    ``init_noise_std`` / ``noise_std_type`` / ``state_dependent_std`` (``translate_runner_dict``).
    """
    for name, cfg in (("actor", actor_cfg), ("critic", critic_cfg)):
        if cfg.get("class_name", _HIM_MODEL) != _HIM_MODEL:
            raise _unsupported(f"{name}.class_name", cfg["class_name"], _HIM_MODEL, "HIM's networks are MLPs")
    if critic_cfg.get("activation", actor_cfg["activation"]) != actor_cfg["activation"]:
        raise _unsupported(
            "critic.activation", critic_cfg["activation"], actor_cfg["activation"], "HIM shares the actor's activation"
        )
    if "distribution_cfg" in actor_cfg:  # rsl-rl-lib 5
        dist = actor_cfg["distribution_cfg"]
        if dist is None:
            raise _unsupported("actor.distribution_cfg", None, "GaussianDistribution", "HIM samples its actions")
        if dist.get("class_name", "GaussianDistribution") != "GaussianDistribution":
            raise _unsupported(
                "actor.distribution_cfg.class_name",
                dist["class_name"],
                "GaussianDistribution",
                "a state-independent std",
            )
        std_type, init_std = dist.get("std_type", _HIM_STD_TYPE), dist["init_std"]
        std_range = tuple(float(v) for v in dist.get("std_range", _UNCLAMPED_STD_RANGE))
        if std_range != _UNCLAMPED_STD_RANGE:
            raise _unsupported(
                "actor.distribution_cfg.std_range", std_range, _UNCLAMPED_STD_RANGE, "HIM does not clamp its std"
            )
        std_field = "actor.distribution_cfg.std_type"
    else:  # rsl-rl-lib 4, flattened by translate_runner_dict
        if not actor_cfg.get("stochastic", True):
            raise _unsupported("actor.stochastic", False, True, "HIM samples its actions")
        if actor_cfg.get("state_dependent_std", False):
            raise _unsupported("actor.state_dependent_std", True, False, "a state-independent std")
        std_type, init_std = actor_cfg.get("noise_std_type", _HIM_STD_TYPE), actor_cfg["init_noise_std"]
        std_field = "actor.noise_std_type"
    if std_type != _HIM_STD_TYPE:
        raise _unsupported(std_field, std_type, _HIM_STD_TYPE, "HIM learns the std itself, not its log")
    return float(init_std)


def single_group(obs_groups: dict, obs_set: str) -> str:
    """The one observation group ``obs_groups[obs_set]`` names; HIM reads each set from a single group."""
    groups = obs_groups[obs_set]
    if len(groups) != 1:
        raise ValueError(
            f"HIM reads its {obs_set} observations from exactly one group, got obs_groups[{obs_set!r}]={groups!r}. "
            f"Concatenate them into one observation group."
        )
    return groups[0]


class HIMActorCritic(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_one_step_obs: int,
        actor_history_length: int,
        num_actions: int,
        actor_hidden_dims=(512, 256, 256),
        critic_hidden_dims=(512, 256, 256),
        activation: str = "elu",
        init_noise_std: float = 1.0,
        dynamic_latent_dim: int = 32,
        obs_group: str = "policy",
        target_skip_dim: int = 3,
        critic_obs_group: str = "critic",
    ):
        super().__init__()
        act = _get_activation(activation)

        self.num_actor_obs = num_actor_obs
        self.num_critic_obs = num_critic_obs
        self.num_one_step_obs = num_one_step_obs
        self.actor_history_length = actor_history_length
        self.actor_proprioceptive_obs_length = actor_history_length * num_one_step_obs
        self.num_actions = num_actions
        self.dynamic_latent_dim = dynamic_latent_dim
        self.obs_group = obs_group
        self.critic_obs_group = critic_obs_group

        # Estimator: history -> velocity(3) + latent(dynamic_latent_dim)
        self.estimator = HIMEstimator(
            temporal_steps=actor_history_length,
            num_one_step_obs=num_one_step_obs,
            num_height_points=0,
            latent_dim=dynamic_latent_dim,
            target_skip_dim=target_skip_dim,
        )

        # Actor: current_obs(num_one_step_obs) ⊕ vel(3) ⊕ latent -> action
        mlp_input_dim_a = num_one_step_obs + VELOCITY_DIM + dynamic_latent_dim
        actor_layers: list[nn.Module] = [nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]), act]
        for i in range(len(actor_hidden_dims)):
            if i == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[i], num_actions))
            else:
                actor_layers += [nn.Linear(actor_hidden_dims[i], actor_hidden_dims[i + 1]), act]
        self.actor = nn.Sequential(*actor_layers)

        # Critic: privileged obs -> value
        critic_layers: list[nn.Module] = [nn.Linear(num_critic_obs, critic_hidden_dims[0]), act]
        for i in range(len(critic_hidden_dims)):
            if i == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[i], 1))
            else:
                critic_layers += [nn.Linear(critic_hidden_dims[i], critic_hidden_dims[i + 1]), act]
        self.critic = nn.Sequential(*critic_layers)

        # State-independent action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution: Normal | None = None
        # A call, as rsl-rl's own GaussianDistribution makes it. OpenHomie *assigns* False here, which
        # leaves validation on and replaces torch's method with a bool, so every rsl-rl distribution
        # built later in the process fails with "'bool' object is not callable".
        Normal.set_default_validate_args(False)

    @classmethod
    def from_groups(
        cls,
        obs,
        actor_group: str,
        critic_group: str,
        num_actions: int,
        *,
        actor_hidden_dims,
        critic_hidden_dims,
        activation: str,
        init_noise_std: float,
        target_skip_dim: int,
    ) -> HIMActorCritic:
        """Size the networks off the observation TensorDict, checking it has the layout HIM assumes.

        The ``actor_group`` holds a whole number of stacked frames, the ``critic_group`` one frame followed
        by the true base linear velocity (``VELOCITY_DIM``).
        """
        num_actor_obs = int(obs[actor_group].shape[-1])
        num_critic_obs = int(obs[critic_group].shape[-1])
        frame = num_critic_obs - VELOCITY_DIM
        if frame <= 0 or num_actor_obs % frame:
            raise ValueError(
                f"HIM needs the {critic_group!r} group to be one actor frame + the {VELOCITY_DIM}-d base linear "
                f"velocity, and the {actor_group!r} group a whole number of such frames; got actor width "
                f"{num_actor_obs} and critic width {num_critic_obs} (frame {frame})."
            )
        return cls(
            num_actor_obs=num_actor_obs,
            num_critic_obs=num_critic_obs,
            num_one_step_obs=frame,
            actor_history_length=num_actor_obs // frame,
            num_actions=num_actions,
            actor_hidden_dims=tuple(actor_hidden_dims),
            critic_hidden_dims=tuple(critic_hidden_dims),
            activation=activation,
            init_noise_std=float(init_noise_std),
            obs_group=actor_group,
            critic_obs_group=critic_group,
            target_skip_dim=target_skip_dim,
        )

    @classmethod
    def from_rsl_rl_v3(cls, obs, obs_groups: dict, num_actions: int, **policy_cfg) -> HIMActorCritic:
        """rsl-rl-lib 3's construction call, ``policy_class(obs, obs_groups, num_actions, **policy_cfg)``.

        Reached through the ``policy.class_name`` that ``HIMPPOAlgorithmOptions.adapt_runner_dict`` writes.
        """
        if policy_cfg.get("actor_obs_normalization") or policy_cfg.get("critic_obs_normalization"):
            raise ValueError(_NO_OBS_NORMALIZATION)
        std_type = policy_cfg.get("noise_std_type", _HIM_STD_TYPE)
        if std_type != _HIM_STD_TYPE:
            raise _unsupported(
                "policy.noise_std_type", std_type, _HIM_STD_TYPE, "HIM learns the std itself, not its log"
            )
        return cls.from_groups(
            obs,
            single_group(obs_groups, "policy"),
            single_group(obs_groups, "critic"),
            num_actions,
            actor_hidden_dims=policy_cfg["actor_hidden_dims"],
            critic_hidden_dims=policy_cfg["critic_hidden_dims"],
            activation=policy_cfg["activation"],
            init_noise_std=policy_cfg["init_noise_std"],
            target_skip_dim=policy_cfg["target_skip_dim"],
        )

    def reset(self, dones=None) -> None:
        pass

    def forward(self, obs) -> torch.Tensor:
        """The deterministic action for an observation TensorDict: what rsl-rl-lib 4/5's ``get_inference_policy``
        serves.
        """
        return self.act_inference(obs)

    @property
    def action_mean(self) -> torch.Tensor:
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        return self.distribution.entropy().sum(dim=-1)

    @property
    def output_std(self) -> torch.Tensor:
        """Current action-noise std (used for logging by the runner)."""
        return self.std.detach()

    def _actor_input(self, obs_history: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            vel, dynamic_latent = self.estimator(obs_history[:, : self.actor_proprioceptive_obs_length])
        return torch.cat((obs_history[:, -self.num_one_step_obs :], vel, dynamic_latent), dim=-1)

    def update_distribution(self, obs_history: torch.Tensor) -> None:
        action_mean = self.actor(self._actor_input(obs_history))
        self.distribution = Normal(action_mean, action_mean * 0.0 + self.std)

    def act(self, obs_history: torch.Tensor, **kwargs) -> torch.Tensor:
        self.update_distribution(obs_history)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, obs_history, **kwargs) -> torch.Tensor:
        """The deterministic action, from the history tensor or the whole observation TensorDict.

        rsl-rl-lib 3's ``get_inference_policy`` hands out this method itself, called on the TensorDict.
        """
        if not isinstance(obs_history, torch.Tensor):
            obs_history = obs_history[self.obs_group]
        return self.actor(self._actor_input(obs_history))

    def evaluate(self, critic_observations: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.critic(critic_observations)

    def update_estimator(self, obs_history: torch.Tensor, next_critic_obs: torch.Tensor, lr: float | None = None):
        return self.estimator.update(obs_history[:, : self.actor_proprioceptive_obs_length], next_critic_obs, lr)
