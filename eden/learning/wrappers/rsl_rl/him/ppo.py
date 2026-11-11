"""HIMPPO — OpenHomie's ``HIMPPO`` (``HomieRL/rsl_rl/rsl_rl/algorithms/him_ppo.py``) for rsl-rl-lib 3–5.

Standard PPO plus three things from HIMLoco / OpenHomie:

* **rollout-time symmetry** — each env step stores the real transition and its mirror, so GAE runs
  over doubled on-policy data;
* **a self-supervised estimator** — the HIM estimator trains each mini-batch on the real and mirrored
  observations, with its own optimizer (``HIMActorCritic.update_estimator``);
* **mirror losses** — an actor loss ``a(flip(o)) ≈ flip(a(o))`` and a critic loss
  ``V(flip(o)) ≈ V(o)``, each scaled by ``symmetry_scale``.

Task-agnostic: the mirror is the task's rsl-rl ``data_augmentation_func`` and the observation layout
it needs is documented on :class:`~eden.learning.wrappers.rsl_rl.him.options.HIMPPOAlgorithmOptions`.
Runs on rsl-rl-lib 3, 4 and 5. The 4 / 5 ``OnPolicyRunner`` builds it from
``runner_options.algorithm.class_name`` through :meth:`HIMPPO.construct_algorithm`; the 3 runner builds the
actor-critic and the algorithm itself, which :meth:`HIMPPOAlgorithmOptions.adapt_runner_dict
<eden.learning.wrappers.rsl_rl.him.options.HIMPPOAlgorithmOptions.adapt_runner_dict>` routes to
:meth:`HIMActorCritic.from_rsl_rl_v3` and :meth:`HIMPPO.from_rsl_rl_v3`. Either way it is driven through
the same ``act`` / ``process_env_step`` / ``compute_returns`` / ``update`` calls as rsl-rl's PPO.
Single-process only: there is no gradient all-reduce.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn
import torch.optim as optim
from rsl_rl.utils import resolve_callable, resolve_obs_groups
from tensordict import TensorDict

from eden.learning.wrappers.rsl_rl.him.actor_critic import (
    _NO_OBS_NORMALIZATION,
    HIMActorCritic,
    check_model_cfg,
    single_group,
)
from eden.learning.wrappers.rsl_rl.him.storage import HIMRolloutStorage


_NOT_DISTRIBUTED = (
    "HIMPPO is single-process: it has no gradient all-reduce or parameter broadcast. Train without torchrun."
)

_NEEDS_FINAL_OBSERVATIONS = (
    "HIMPPO needs extras['final_observations'] for the estimator's target at episode ends; set "
    "env_options.record_final_observations=True."
)

# The algorithm-options entries HIMPPO.__init__ takes.
_ALG_KEYS = (
    "use_flip",
    "num_learning_epochs",
    "num_mini_batches",
    "clip_param",
    "gamma",
    "lam",
    "value_loss_coef",
    "entropy_coef",
    "learning_rate",
    "max_grad_norm",
    "use_clipped_value_loss",
    "schedule",
    "desired_kl",
    "symmetry_scale",
)


class HIMOptimizer:
    """The PPO optimizer, carrying the estimator's own optimizer in its state.

    HIMPPO steps the two separately, but rsl-rl-lib 3's runner checkpoints ``alg.optimizer.state_dict()``
    and nothing else; folding the estimator's state in here is what lets a v3 resume restore both. Every
    other use (``param_groups``, ``zero_grad``, ``step``) is the PPO optimizer's.
    """

    def __init__(self, ppo: optim.Optimizer, estimator: optim.Optimizer) -> None:
        self.ppo = ppo
        self.estimator = estimator

    @property
    def param_groups(self) -> list[dict]:
        return self.ppo.param_groups

    def zero_grad(self) -> None:
        self.ppo.zero_grad()

    def step(self) -> None:
        self.ppo.step()

    def state_dict(self) -> dict:
        return {"ppo": self.ppo.state_dict(), "estimator": self.estimator.state_dict()}

    def load_state_dict(self, state: dict) -> None:
        self.ppo.load_state_dict(state["ppo"])
        self.estimator.load_state_dict(state["estimator"])


class HIMPPO:
    actor_critic: HIMActorCritic

    def __init__(
        self,
        actor_critic: HIMActorCritic,
        use_flip: bool = True,
        num_learning_epochs: int = 5,
        num_mini_batches: int = 4,
        clip_param: float = 0.2,
        gamma: float = 0.99,
        lam: float = 0.95,
        value_loss_coef: float = 1.0,
        entropy_coef: float = 0.01,
        learning_rate: float = 1e-3,
        max_grad_norm: float = 1.0,
        use_clipped_value_loss: bool = True,
        schedule: str = "adaptive",
        desired_kl: float = 0.01,
        device: str = "cpu",
        symmetry_scale: float = 1.0,
        env=None,
        data_augmentation_func: Callable | None = None,
    ):
        if data_augmentation_func is None:
            raise ValueError(
                "HIMPPO stores every transition's mirror, so it needs the task's mirror: set the algorithm options' "
                "data_augmentation_func to a 'module:function' with rsl-rl's symmetry signature."
            )
        self.env = env
        self.data_augmentation_func = data_augmentation_func
        self.device = device
        self.use_flip = use_flip
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate

        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage: HIMRolloutStorage | None = None
        self.optimizer = HIMOptimizer(
            optim.Adam(self.actor_critic.parameters(), lr=learning_rate), self.actor_critic.estimator.optimizer
        )
        self.transition = HIMRolloutStorage.Transition()
        self.transition_sym = HIMRolloutStorage.Transition()
        self.symmetry_scale = symmetry_scale

        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        # Read by rsl-rl-lib's runners: RND is logged when an algorithm has it, and v3 reaches the
        # model as ``alg.policy`` (v4/v5 through ``get_policy()``).
        self.rnd = None
        self.intrinsic_rewards = None
        self.policy = actor_critic
        self.actor_obs_group = actor_critic.obs_group
        self.critic_obs_group = actor_critic.critic_obs_group

    @classmethod
    def construct_algorithm(cls, obs, env, cfg: dict, device: str) -> HIMPPO:
        """rsl-rl-lib 4 / 5's construction hook: build the actor-critic, the storage and the algorithm.

        The mirror is run once on these observations' widths, so a layout it does not cover fails here,
        before a rollout.
        """
        if cfg.get("multi_gpu"):
            raise NotImplementedError(_NOT_DISTRIBUTED)
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], ["actor", "critic"])
        actor_cfg, critic_cfg = cfg["actor"], cfg["critic"]
        if actor_cfg.get("obs_normalization") or critic_cfg.get("obs_normalization"):
            raise ValueError(_NO_OBS_NORMALIZATION)
        init_std = check_model_cfg(actor_cfg, critic_cfg)
        # Without the pre-reset critic observations every rollout would fail on its first step; refuse here.
        env_options = getattr(getattr(env, "unwrapped", env), "env_options", None)
        if env_options is not None and not getattr(env_options, "record_final_observations", True):
            raise ValueError(_NEEDS_FINAL_OBSERVATIONS)
        actor_critic = HIMActorCritic.from_groups(
            obs,
            single_group(cfg["obs_groups"], "actor"),
            single_group(cfg["obs_groups"], "critic"),
            env.num_actions,
            actor_hidden_dims=actor_cfg["hidden_dims"],
            critic_hidden_dims=critic_cfg["hidden_dims"],
            activation=actor_cfg["activation"],
            init_noise_std=init_std,
            target_skip_dim=cfg["algorithm"]["target_skip_dim"],
        )
        return cls._build(actor_critic, cfg["algorithm"], device, env, env.num_envs, cfg["num_steps_per_env"])

    @classmethod
    def from_rsl_rl_v3(
        cls, actor_critic: HIMActorCritic, storage, device: str, multi_gpu_cfg=None, **alg_cfg
    ) -> HIMPPO:
        """rsl-rl-lib 3's construction call, ``alg_class(actor_critic, storage, device=, **alg_cfg, multi_gpu_cfg=)``.

        Reached through the ``algorithm.class_name`` that ``HIMPPOAlgorithmOptions.adapt_runner_dict`` writes;
        the runner has already built ``actor_critic`` (``HIMActorCritic.from_rsl_rl_v3``). Its stock rollout
        storage is only read for its size. v3 hands the algorithm no env, so the mirror gets ``env=None``.
        """
        if multi_gpu_cfg:
            raise NotImplementedError(_NOT_DISTRIBUTED)
        return cls._build(actor_critic, alg_cfg, device, None, storage.num_envs, storage.num_transitions_per_env)

    @classmethod
    def _build(cls, actor_critic: HIMActorCritic, alg_cfg: dict, device: str, env, num_envs: int, num_steps: int):
        alg = cls(
            actor_critic,
            device=device,
            env=env,
            data_augmentation_func=resolve_callable(alg_cfg["data_augmentation_func"]),
            **{k: alg_cfg[k] for k in _ALG_KEYS},
        )
        # Fail here, not mid-rollout, when the task's mirror does not cover this layout.
        alg._mirror_obs(alg.actor_obs_group, torch.zeros(1, actor_critic.num_actor_obs, device=device))
        alg._mirror_obs(alg.critic_obs_group, torch.zeros(1, actor_critic.num_critic_obs, device=device))
        alg._mirror_actions(torch.zeros(1, actor_critic.num_actions, device=device))
        alg.init_storage(
            num_envs, num_steps, [actor_critic.num_actor_obs], [actor_critic.num_critic_obs], [actor_critic.num_actions]
        )
        return alg

    def _mirror_obs(self, group: str, x: torch.Tensor) -> torch.Tensor:
        """The mirror of one observation group's batch, through the task's augmentation function."""
        n = x.shape[0]
        aug, _ = self.data_augmentation_func(env=self.env, obs=TensorDict({group: x}, batch_size=[n]), actions=None)
        return aug[group][n:]

    def _actor_symmetry_loss(self, obs_batch: torch.Tensor) -> torch.Tensor:
        """``symmetry_scale * mean ||pi(mirror(o)) - mirror(pi(o))||^2``, with the mirrored action as the target.

        The target is detached, as OpenHomie's ``flip_g1_actions(...).detach()``: only the policy on the
        mirrored observation is pulled toward it, not the action it is compared with.
        """
        mirrored = self.actor_critic.act_inference(self._mirror_obs(self.actor_obs_group, obs_batch))
        target = self._mirror_actions(self.actor_critic.act_inference(obs_batch)).detach()
        return self.symmetry_scale * torch.mean(torch.sum(torch.square(mirrored - target), dim=-1))

    def _mirror_actions(self, actions: torch.Tensor) -> torch.Tensor:
        _, aug = self.data_augmentation_func(env=self.env, obs=None, actions=actions)
        return aug[actions.shape[0] :]

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape):
        self.storage = HIMRolloutStorage(
            num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape, self.device
        )

    def train_mode(self):
        self.actor_critic.train()

    def eval_mode(self):
        self.actor_critic.eval()

    def get_policy(self):
        return self.actor_critic

    def save(self) -> dict:
        """rsl-rl-lib 4 / 5's checkpoint contents (v3's runner saves ``policy`` and ``optimizer`` itself)."""
        return {
            "actor_critic_state_dict": self.actor_critic.state_dict(),
            "estimator_optimizer_state_dict": self.optimizer.estimator.state_dict(),
            "optimizer_state_dict": self.optimizer.ppo.state_dict(),
        }

    def load(self, loaded_dict: dict, load_cfg: dict | None = None, strict: bool = True) -> bool:
        """Load what ``load_cfg`` selects (everything when ``None``); return whether to restore the iteration.

        The weights always load strictly, whatever ``strict`` says. Eden's checkpoint loader passes
        ``strict=False`` so stock PPO can take older rsl-rl layouts, but a HIM checkpoint has one layout: a
        missing or extra key means the wrong checkpoint, and loading past it would leave those weights at
        their random init. ``ValueError``, not ``KeyError``, so the loader does not reword it as an
        rsl-rl version mismatch.
        """
        if load_cfg is None:
            load_cfg = {"actor": True, "critic": True, "optimizer": True, "iteration": True}
        if "actor_critic_state_dict" not in loaded_dict:
            raise ValueError(
                f"Not a HIMPPO checkpoint from rsl-rl-lib 4/5: no 'actor_critic_state_dict' among "
                f"{sorted(loaded_dict)}. A stock-PPO or rsl-rl-lib 3 checkpoint cannot initialize HIMPPO; "
                f"pass a checkpoint this algorithm saved."
            )
        if load_cfg.get("actor") or load_cfg.get("critic"):
            state = loaded_dict["actor_critic_state_dict"]
            expected = set(self.actor_critic.state_dict())
            missing, unexpected = sorted(expected - set(state)), sorted(set(state) - expected)
            if missing or unexpected:
                raise ValueError(
                    f"HIMPPO checkpoint does not match this actor-critic: missing {missing}, unexpected "
                    f"{unexpected}. It was saved for a different observation layout or network; train or "
                    f"pick a checkpoint from this task's config."
                )
            self.actor_critic.load_state_dict(state, strict=True)
        if load_cfg.get("optimizer"):
            self.optimizer.ppo.load_state_dict(loaded_dict["optimizer_state_dict"])
            self.optimizer.estimator.load_state_dict(loaded_dict["estimator_optimizer_state_dict"])
            self.learning_rate = self.optimizer.param_groups[0]["lr"]
        return bool(load_cfg.get("iteration", False))

    def _act(self, obs: torch.Tensor, critic_obs: torch.Tensor) -> torch.Tensor:
        # Real transition
        self.transition.actions = self.actor_critic.act(obs).detach()
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs

        # Mirrored transition
        obs_sym = self._mirror_obs(self.actor_obs_group, obs)
        critic_obs_sym = self._mirror_obs(self.critic_obs_group, critic_obs)
        self.transition_sym.actions = self.actor_critic.act(obs_sym).detach()
        self.transition_sym.values = self.actor_critic.evaluate(critic_obs_sym).detach()
        self.transition_sym.actions_log_prob = self.actor_critic.get_actions_log_prob(
            self.transition_sym.actions
        ).detach()
        self.transition_sym.action_mean = self.actor_critic.action_mean.detach()
        self.transition_sym.action_sigma = self.actor_critic.action_std.detach()
        self.transition_sym.observations = obs_sym
        self.transition_sym.critic_observations = critic_obs_sym
        return self.transition.actions

    def act(self, obs) -> torch.Tensor:
        return self._act(obs[self.actor_obs_group], obs[self.critic_obs_group])

    def process_env_step(self, obs, rewards, dones, extras) -> None:
        """Record the step; ``obs`` is the observation *after* it, post-reset for the envs that ended.

        The estimator's target is the true next critic observation, so the ended envs take their
        pre-reset one from ``extras["final_observations"]`` — OpenHomie's
        ``next_critic_obs[termination_ids] = termination_privileged_obs``.
        """
        final = extras.get("final_observations")
        if final is None:
            raise KeyError(_NEEDS_FINAL_OBSERVATIONS)
        next_critic_obs = torch.where(
            dones.bool().unsqueeze(-1), final[self.critic_obs_group].to(self.device), obs[self.critic_obs_group]
        )
        self._process_env_step(rewards, dones, extras, next_critic_obs)

    def compute_returns(self, obs) -> None:
        last_values = self.actor_critic.evaluate(obs[self.critic_obs_group]).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def _process_env_step(self, rewards, dones, infos, next_critic_obs):
        self.transition.next_critic_observations = next_critic_obs.clone()
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones

        self.transition_sym.next_critic_observations = self._mirror_obs(self.critic_obs_group, next_critic_obs)
        self.transition_sym.rewards = rewards.clone()
        self.transition_sym.dones = dones

        # Bootstrapping on time outs
        if "time_outs" in infos:
            time_outs = infos["time_outs"].unsqueeze(1).to(self.device)
            self.transition.rewards += self.gamma * torch.squeeze(self.transition.values * time_outs, 1)
            self.transition_sym.rewards += self.gamma * torch.squeeze(self.transition_sym.values * time_outs, 1)

        self.storage.add_transitions(self.transition)
        self.storage.add_transitions(self.transition_sym)
        self.transition.clear()
        self.transition_sym.clear()
        self.actor_critic.reset(dones)

    def update(self):
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_estimation_loss = 0.0
        mean_swap_loss = 0.0
        mean_actor_sym_loss = 0.0
        mean_critic_sym_loss = 0.0

        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for (
            obs_batch,
            critic_obs_batch,
            actions_batch,
            next_critic_obs_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
        ) in generator:
            self.actor_critic.act(obs_batch)
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
            value_batch = self.actor_critic.evaluate(critic_obs_batch)
            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy

            # Adaptive-KL learning-rate schedule
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)
                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # Estimator update (on real ⊕ mirrored obs)
            if self.use_flip:
                estimator_obs = torch.cat((obs_batch, self._mirror_obs(self.actor_obs_group, obs_batch)), dim=0)
                estimator_next_critic_obs = torch.cat(
                    (next_critic_obs_batch, self._mirror_obs(self.critic_obs_group, next_critic_obs_batch)), dim=0
                )
            else:
                estimator_obs = obs_batch
                estimator_next_critic_obs = next_critic_obs_batch
            estimation_loss, swap_loss = self.actor_critic.update_estimator(
                estimator_obs, estimator_next_critic_obs, lr=self.learning_rate
            )

            # Surrogate loss
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            # Symmetry (mirror) losses
            if self.use_flip:
                flipped_critic_obs_batch = self._mirror_obs(self.critic_obs_group, critic_obs_batch)
                actor_sym_loss = self._actor_symmetry_loss(obs_batch)
                critic_sym_loss = self.symmetry_scale * torch.mean(
                    torch.square(
                        self.actor_critic.evaluate(flipped_critic_obs_batch)
                        - self.actor_critic.evaluate(critic_obs_batch).detach()
                    )
                )
                loss = (
                    surrogate_loss
                    + self.value_loss_coef * value_loss
                    - self.entropy_coef * entropy_batch.mean()
                    + actor_sym_loss
                    + critic_sym_loss
                )
            else:
                loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_estimation_loss += estimation_loss
            mean_swap_loss += swap_loss
            if self.use_flip:
                mean_actor_sym_loss += actor_sym_loss.item()
                mean_critic_sym_loss += critic_sym_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        self.storage.clear()
        return {
            "value_loss": mean_value_loss / num_updates,
            "surrogate_loss": mean_surrogate_loss / num_updates,
            "estimation_loss": mean_estimation_loss / num_updates,
            "swap_loss": mean_swap_loss / num_updates,
            "actor_sym_loss": mean_actor_sym_loss / num_updates,
            "critic_sym_loss": mean_critic_sym_loss / num_updates,
        }
