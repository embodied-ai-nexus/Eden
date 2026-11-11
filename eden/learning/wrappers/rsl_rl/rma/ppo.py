"""PPO for the RMA teacher (rsl-rl-lib >= 5.5.1): concurrent estimator, privileged regularization, DAgger, mirror loss.

Extends rsl_rl's PPO (BSD-3) with the teacher-specific mechanisms from "Extreme Parkour with Legged Robots"
(Cheng et al. 2023):

1. A base-velocity **estimator** (inside the actor model, own Adam) trained concurrently to predict the
   privileged-explicit chunk from proprioception. With ``train_with_estimated_states`` the actor *acts* on the
   estimated chunk during rollouts while storage keeps the ground truth — the estimator's regression target and what
   the critic and the PPO update consume.
2. A **privileged regularization** loss pulling the privileged-latent encoder toward the (frozen) history encoder's
   output, with a coefficient ramped by ``priv_reg_coef_schedule``.
3. **DAgger iterations**: every ``dagger_update_freq``-th learning iteration the rollout is collected with the
   history (adaptation) latent driving the actor, and after the PPO update the history encoder is regressed onto the
   privileged latent over that rollout. The cadence lives here (the iteration counter is saved in checkpoints), so
   rsl_rl's stock ``OnPolicyRunner`` drives the whole teacher.
4. An optional **mirror-equivariance** loss ``|| pi(mirror(obs)) - mirror(pi(obs)) ||^2`` over a task-supplied
   left-right mirror (``mirror_class_name``), which removes the asymmetric gaits an RMA teacher otherwise learns.

Deliberate choices: the PPO loss recomputes the distribution through the privileged path on ground-truth observations
even for rollouts collected with the estimator / history latent; the gradient clip covers actor and critic jointly;
the regularization schedule's counter advances on PPO *and* DAgger updates; DAgger runs before the storage is cleared,
over the same rollout the PPO update used.
"""

from __future__ import annotations

from itertools import chain

import torch
import torch.nn as nn
from tensordict import TensorDict

from eden.learning.wrappers.rsl_rl import MIN_RSL_RL_VERSION
from eden.options.learning.rsl_rl.translate import require_rsl_rl_version

require_rsl_rl_version(MIN_RSL_RL_VERSION, __name__, "PPO over separate actor/critic models, built by resolve_class")

from rsl_rl.algorithms import PPO  # noqa: E402  (after the version guard, on purpose)
from rsl_rl.utils import resolve_callable, resolve_optimizer  # noqa: E402

from eden.learning.wrappers.rsl_rl.rma.model import RmaActorModel  # noqa: E402

#: ``class_name`` string rsl_rl resolves to :class:`RmaPPO`.
RMA_PPO_CLASS_NAME = "eden.learning.wrappers.rsl_rl.rma.ppo:RmaPPO"


class RmaPPO(PPO):
    """PPO + estimator + privileged-latent regularization + DAgger (+ mirror loss) for an :class:`RmaActorModel`.

    The extra keyword arguments are documented on
    :class:`~eden.learning.wrappers.rsl_rl.rma.options.RslRlRmaPpoAlgorithmOptions`; the rest go to rsl_rl's PPO.
    """

    actor: RmaActorModel

    #: Read by ``eden train`` before it builds the env, under a distributed launch.
    distributed_unsupported = (
        "RmaPPO trains on one GPU: its estimator and history-encoder optimizers have no gradient reduction, so a "
        "distributed launch would train them independently per rank. Launch the RMA teacher as one process "
        "(without torchrun)."
    )

    def __init__(
        self,
        actor: RmaActorModel,
        critic,
        storage,
        *,
        estimator_learning_rate: float = 1.0e-4,
        train_with_estimated_states: bool = True,
        dagger_update_freq: int = 20,
        priv_reg_coef_schedule: tuple[float, float, int, int] | list = (0.0, 0.1, 2000, 3000),
        symmetry_coef: float = 0.0,
        mirror_class_name: str | None = None,
        optimizer: str = "adam",
        **ppo_kwargs,
    ) -> None:
        if not isinstance(actor, RmaActorModel):
            raise TypeError(
                f"RmaPPO trains an RmaActorModel, got {type(actor).__name__}; set the runner's actor to "
                "RslRlRmaActorOptions(...) (eden.learning.wrappers.rsl_rl.rma)."
            )
        for key in ("rnd_cfg", "symmetry_cfg", "multi_gpu_cfg"):
            if ppo_kwargs.get(key) is not None:
                raise ValueError(
                    f"RmaPPO does not support {key} (got {ppo_kwargs[key]!r}): leave it None. Its mirror loss is "
                    "symmetry_coef + mirror_class_name, and it trains on one GPU — launch without torchrun."
                )
        if ppo_kwargs.get("use_mixed_precision"):
            raise ValueError("RmaPPO does not support use_mixed_precision=True; leave it False.")
        if critic.is_recurrent:
            raise ValueError(
                f"RmaPPO needs a feed-forward critic, got the recurrent {type(critic).__name__}: its update draws flat "
                "mini-batches without hidden states. Use RslRlCriticOptions (an MLP) for the critic."
            )
        if dagger_update_freq < 1:
            raise ValueError(f"RmaPPO(dagger_update_freq={dagger_update_freq}) must be >= 1.")
        if symmetry_coef > 0.0 and mirror_class_name is None:
            raise ValueError(
                f"RmaPPO(symmetry_coef={symmetry_coef}) needs mirror_class_name='module.path:Class' — a class built "
                "as Class(dims, device) with mirror_obs(flat) / mirror_action(actions) for the task's layout "
                "(e.g. 'eden.tasks.extreme_parkour.symmetry:ParkourMirror'); or set symmetry_coef=0.0."
            )
        super().__init__(actor, critic, storage, optimizer=optimizer, **ppo_kwargs)
        # PPO's optimizer without the estimator, which trains on its own regression loss (and optimizer) below.
        self.optimizer = resolve_optimizer(optimizer)(
            chain(self.actor.policy_parameters(), self.critic.parameters()), lr=self.learning_rate
        )
        self.estimator_optimizer = torch.optim.Adam(self.actor.estimator.parameters(), lr=estimator_learning_rate)
        self.hist_encoder_optimizer = torch.optim.Adam(
            self.actor.net.history_encoder.parameters(), lr=self.learning_rate
        )
        self.train_with_estimated_states = train_with_estimated_states
        self.dagger_update_freq = dagger_update_freq
        self.priv_reg_coef_schedule = tuple(priv_reg_coef_schedule)
        self.symmetry_coef = symmetry_coef
        self.mirror = resolve_callable(mirror_class_name)(self.actor.dims, self.device) if symmetry_coef > 0.0 else None
        # Drives the priv-reg ramp; advances on every PPO update and every DAgger update.
        self.counter = 0
        # The learning iteration whose rollout is being collected (the runner's `it`); picks DAgger iterations.
        self.iteration = 0

    @property
    def is_dagger_iteration(self) -> bool:
        """Whether the current rollout uses the history latent and is followed by a DAgger update."""
        return self.iteration % self.dagger_update_freq == 0

    def act(self, obs: TensorDict) -> torch.Tensor:
        """Sample actions — through the estimator and, on DAgger iterations, the history latent — and record them."""
        self.transition.hidden_states = (self.actor.get_hidden_state(), self.critic.get_hidden_state())
        self.transition.actions = self.actor(
            obs,
            stochastic_output=True,
            hist_encoding=self.is_dagger_iteration,
            estimate=self.train_with_estimated_states,
        ).detach()
        self.transition.values = self.critic(obs).detach()
        self.transition.actions_log_prob = self.actor.get_output_log_prob(self.transition.actions).detach()
        self.transition.distribution_params = tuple(p.detach() for p in self.actor.output_distribution_params)
        self.transition.observations = obs  # ground truth: the estimator's target, the critic's input
        return self.transition.actions

    def update(self) -> dict[str, float]:
        """PPO epochs with the estimator, priv-reg and mirror terms; then DAgger on DAgger iterations."""
        actor, dims = self.actor, self.actor.dims
        mean_value_loss = mean_surrogate_loss = mean_entropy = 0.0
        mean_priv_reg_loss = mean_estimator_loss = mean_symmetry_loss = 0.0
        priv_reg_coef = 0.0
        start, end, ramp_start, ramp_len = self.priv_reg_coef_schedule

        for batch in self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs):
            if self.normalize_advantage_per_mini_batch:  # else compute_returns normalized them over the rollout
                with torch.no_grad():
                    batch.advantages = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1e-8)
            # PPO always trains through the privileged path on the stored ground truth; the history encoder trains
            # in the DAgger update only.
            actor(batch.observations, stochastic_output=True, hist_encoding=False, estimate=False)
            actions_log_prob = actor.get_output_log_prob(batch.actions)
            values = self.critic(batch.observations)
            distribution_params = actor.output_distribution_params
            mu = actor.output_mean
            entropy = actor.output_entropy
            flat = actor.get_flat_obs(batch.observations)

            # Privileged regularization toward the frozen history latent, coefficient ramped by the counter.
            priv_latent = actor.net.infer_priv_latent(flat)
            with torch.no_grad():
                hist_latent = actor.net.infer_hist_latent(flat)
            priv_reg_loss = (priv_latent - hist_latent).norm(p=2, dim=1).mean()
            priv_reg_coef = start + min(max(self.counter - ramp_start, 0) / ramp_len, 1.0) * (end - start)

            # Estimator step on its own optimizer: the stored obs carry the ground-truth priv_explicit.
            predicted = actor.estimator(flat[:, dims.prop_slice])
            estimator_loss = (predicted - flat[:, dims.priv_explicit_slice]).pow(2).mean()
            self.estimator_optimizer.zero_grad()
            estimator_loss.backward()
            nn.utils.clip_grad_norm_(actor.estimator.parameters(), self.max_grad_norm)
            self.estimator_optimizer.step()

            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl_mean = actor.get_kl_divergence(batch.old_distribution_params, distribution_params).mean()
                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            ratio = torch.exp(actions_log_prob - torch.squeeze(batch.old_actions_log_prob))
            surrogate = -torch.squeeze(batch.advantages) * ratio
            surrogate_clipped = -torch.squeeze(batch.advantages) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            if self.use_clipped_value_loss:
                value_clipped = batch.values + (values - batch.values).clamp(-self.clip_param, self.clip_param)
                value_loss = torch.max((values - batch.returns).pow(2), (value_clipped - batch.returns).pow(2)).mean()
            else:
                value_loss = (batch.returns - values).pow(2).mean()

            # Mirror equivariance on the action mean; both sides carry gradient.
            symmetry_loss = torch.zeros((), device=self.device)
            if self.mirror is not None:
                mu_mirrored = actor.mean(self.mirror.mirror_obs(flat), hist_encoding=False, estimate=False)
                symmetry_loss = (mu_mirrored - self.mirror.mirror_action(mu)).pow(2).mean()

            loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy.mean()
                + priv_reg_coef * priv_reg_loss
                + self.symmetry_coef * symmetry_loss
            )
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(chain(actor.policy_parameters(), self.critic.parameters()), self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy.mean().item()
            mean_priv_reg_loss += priv_reg_loss.item()
            mean_estimator_loss += estimator_loss.item()
            mean_symmetry_loss += symmetry_loss.item()

        self.critic.update_normalization(self.storage.observations.flatten(0, 1))
        self.counter += 1
        num_updates = self.num_learning_epochs * self.num_mini_batches
        loss_dict = {
            "value": mean_value_loss / num_updates,
            "surrogate": mean_surrogate_loss / num_updates,
            "entropy": mean_entropy / num_updates,
            "priv_reg": mean_priv_reg_loss / num_updates,
            "priv_reg_coef": priv_reg_coef,
            "estimator": mean_estimator_loss / num_updates,
        }
        if self.mirror is not None:
            loss_dict["symmetry"] = mean_symmetry_loss / num_updates
        if self.is_dagger_iteration:
            loss_dict["hist_latent"] = self.update_dagger()
        self.storage.clear()
        self.iteration += 1
        return loss_dict

    def update_dagger(self) -> float:
        """Regress the history (adaptation) encoder onto the privileged latent over the stored rollout."""
        actor = self.actor
        mean_loss = 0.0
        for batch in self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs):
            flat = actor.get_flat_obs(batch.observations)
            with torch.no_grad():
                priv_latent = actor.net.infer_priv_latent(flat)
            hist_latent_loss = (priv_latent - actor.net.infer_hist_latent(flat)).norm(p=2, dim=1).mean()
            self.hist_encoder_optimizer.zero_grad()
            hist_latent_loss.backward()
            nn.utils.clip_grad_norm_(actor.net.history_encoder.parameters(), self.max_grad_norm)
            self.hist_encoder_optimizer.step()
            mean_loss += hist_latent_loss.item()
        self.counter += 1
        return mean_loss / (self.num_learning_epochs * self.num_mini_batches)

    def save(self) -> dict:
        saved = super().save()
        saved["estimator_optimizer_state_dict"] = self.estimator_optimizer.state_dict()
        saved["hist_encoder_optimizer_state_dict"] = self.hist_encoder_optimizer.state_dict()
        saved["rma_counter"] = self.counter
        return saved

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        """PPO's load — with the actor checked key for key whatever ``strict`` says — plus the RMA state.

        Eden loads with ``strict=False`` to tolerate drift, but a partially loaded RMA actor (another model's
        checkpoint, e.g. a plain MLP actor's ``mlp.*``) would run with random encoders or estimator.
        """
        if (load_cfg is None or load_cfg.get("actor")) and "actor_state_dict" in loaded_dict:
            expected, found = set(self._raw_actor.state_dict()), set(loaded_dict["actor_state_dict"])
            if expected != found:
                raise ValueError(
                    "This checkpoint's actor_state_dict is not an RmaActorModel's: missing "
                    f"{sorted(expected - found)[:6]}{' ...' if len(expected - found) > 6 else ''}, unexpected "
                    f"{sorted(found - expected)[:6]}{' ...' if len(found - expected) > 6 else ''}. It was saved by "
                    "another actor model (or an RMA actor with other layers); load it with the task that trained it."
                )
        load_iteration = super().load(loaded_dict, load_cfg, strict)
        if load_cfg is None or load_cfg.get("optimizer"):
            for key, opt in (
                ("estimator_optimizer_state_dict", self.estimator_optimizer),
                ("hist_encoder_optimizer_state_dict", self.hist_encoder_optimizer),
            ):
                if key in loaded_dict:  # absent from a checkpoint another algorithm saved: keep the fresh state
                    opt.load_state_dict(loaded_dict[key])
        if load_iteration:
            # The runner resumes at the saved iteration number, so the DAgger cadence and the ramp continue.
            self.iteration = loaded_dict["iter"]
            self.counter = loaded_dict.get("rma_counter", self.counter)
        return load_iteration
