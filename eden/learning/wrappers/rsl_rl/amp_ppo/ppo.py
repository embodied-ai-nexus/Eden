"""AMP-augmented PPO (rsl-rl-lib >= 5.5.1): PPO plus an adversarial motion prior over a window of joint positions.

Ported from the HUSKY skateboarding release (``TeleHuman/humanoid_skateboarding``), whose vendored rsl_rl fork trains
the prior over the push-phase joint trajectories. Select it with
:class:`~eden.learning.wrappers.rsl_rl.amp_ppo.options.RslRlAmpPpoAlgorithmOptions` as the runner's ``algorithm``.

The environment publishes an extra observation group (``amp_obs_group``, ``"amp"`` by default) laid out as
``[joint_pos x num_frames history (oldest first), style_gate]`` — it rides the observation TensorDict into
:meth:`process_env_step` the same way rsl_rl's own RND extension consumes its ``rnd_state`` group. Per step, the
discriminator's style reward ``coef * clamp(1 - 0.25 (d-1)^2, 0) * scale`` is added to the environment reward for envs
whose gate is set (HUSKY gates it on the push phase, the only phase its expert clips cover), and the window is pushed
into the policy replay buffer. Per learning iteration, the discriminator takes ``num_learning_epochs *
num_mini_batches`` LSGAN updates (expert target +1, policy target -1, gradient penalty at the expert samples), with its
optimizer's learning rate slaved to PPO's KL-adaptive rate — mirroring the source, where the discriminator shared the
PPO optimizer.

Under a distributed launch (``multi_gpu_cfg``) every rank keeps one discriminator: :meth:`broadcast_parameters` syncs
it and the frame normalizer from rank 0, its gradients are averaged like PPO's, and the normalizer folds the frames of
every rank, so each rank's policy is scored by the same discriminator. Each rank's replay buffer holds its own envs'
windows (the per-rank batches together form the global batch, as for PPO).

A checkpoint without the AMP state (one trained as plain PPO) is refused unless ``amp_warm_start=True``: resuming it
would add the style reward under a value function trained without it. Replay (``eden inference`` / ``eden deploy``)
loads AMP checkpoints under stock PPO, through the options' ``for_inference()``.

Deviations from the source (both judged non-load-bearing):
- the discriminator trains after the PPO epochs on a separate optimizer (same update count/batch size, lr slaved to
  the KL-adaptive rate) instead of sharing PPO's optimizer,
- the frame normalizer updates from raw frames (the source normalized twice).
"""

from __future__ import annotations

import torch
from tensordict import TensorDict

from eden.learning.wrappers.rsl_rl import MIN_RSL_RL_VERSION
from eden.options.learning.rsl_rl.translate import require_rsl_rl_version

require_rsl_rl_version(
    MIN_RSL_RL_VERSION,
    __name__,
    "PPO over separate actor/critic models, and reduce_gradients_in_buckets for the discriminator under DDP",
)

from rsl_rl.algorithms import PPO  # noqa: E402  (after the version guard, on purpose)
from rsl_rl.utils import reduce_gradients_in_buckets  # noqa: E402

from eden.learning.wrappers.rsl_rl.amp_ppo.amp import (  # noqa: E402
    AmpDiscriminator,
    AmpMotionSampler,
    AmpReplayBuffer,
    FrameNormalizer,
)

#: ``class_name`` string rsl_rl resolves to :class:`AmpPPO`.
AMP_PPO_CLASS_NAME = "eden.learning.wrappers.rsl_rl.amp_ppo.ppo:AmpPPO"

#: What :meth:`AmpPPO.save` adds to PPO's checkpoint.
_AMP_CHECKPOINT_KEYS = ("amp_discriminator_state_dict", "amp_optimizer_state_dict", "amp_normalizer_state_dict")


class AmpPPO(PPO):
    """PPO + an adversarial motion prior on gated windows of joint positions.

    The ``amp_*`` keyword arguments are documented on
    :class:`~eden.learning.wrappers.rsl_rl.amp_ppo.options.RslRlAmpPpoAlgorithmOptions`; the rest go to rsl_rl's PPO.
    """

    def __init__(
        self,
        actor,
        critic,
        storage,
        *,
        amp_motion_file: str,
        amp_obs_group: str = "amp",
        amp_num_frames: int = 5,
        amp_reward_coef: float = 5.0,
        amp_reward_scale: float = 0.02,
        amp_replay_size: int = 100_000,
        amp_discr_hidden_dims: tuple[int, ...] = (256, 256),
        amp_grad_pen_scale: float = 5.0,
        amp_trunk_weight_decay: float = 1e-3,
        amp_head_weight_decay: float = 1e-1,
        amp_warm_start: bool = False,
        **ppo_kwargs,
    ) -> None:
        super().__init__(actor, critic, storage, **ppo_kwargs)
        self.amp_obs_group = amp_obs_group
        self.amp_warm_start = amp_warm_start
        self.amp_num_frames = amp_num_frames
        self.amp_reward_coef = amp_reward_coef
        self.amp_reward_scale = amp_reward_scale
        self.amp_grad_pen_scale = amp_grad_pen_scale

        self.amp_expert = AmpMotionSampler(amp_motion_file, amp_num_frames, self.device)
        window_dim = self.amp_expert.frame_dim * amp_num_frames
        self.discriminator = AmpDiscriminator(window_dim, tuple(amp_discr_hidden_dims)).to(self.device)
        self.amp_normalizer = FrameNormalizer(self.amp_expert.frame_dim).to(self.device)
        self.amp_replay = AmpReplayBuffer(window_dim, amp_replay_size, self.device)
        self.amp_optimizer = torch.optim.Adam(
            [
                {"params": self.discriminator.trunk.parameters(), "weight_decay": amp_trunk_weight_decay},
                {"params": self.discriminator.head.parameters(), "weight_decay": amp_head_weight_decay},
            ],
            lr=self.learning_rate,
        )
        self._window_dim = window_dim
        # Accumulated as a tensor to avoid a GPU sync inside the rollout loop.
        self._style_reward_sum = torch.zeros((), device=self.device)
        self._style_reward_steps = 0

    @staticmethod
    def construct_algorithm(obs: TensorDict, env, cfg: dict, device: str) -> AmpPPO:
        """rsl_rl's PPO construction, then the AMP group checked against the env's first observation."""
        alg = PPO.construct_algorithm(obs, env, cfg, device)
        if alg.amp_obs_group not in obs.keys():
            raise ValueError(
                f"AmpPPO reads the observation group amp_obs_group={alg.amp_obs_group!r}, which the env does not "
                f"provide (groups: {sorted(obs.keys())}). Add it to the task's observation_options, or point "
                "amp_obs_group at the group that carries the AMP window and gate."
            )
        alg._split_amp_obs(obs)  # raises, naming the expected layout, on a mis-sized group
        return alg

    # ------------------------------------------------------------------
    # Rollout hooks
    # ------------------------------------------------------------------

    def _split_amp_obs(self, obs: TensorDict) -> tuple[torch.Tensor, torch.Tensor]:
        amp = obs[self.amp_obs_group]
        if amp.shape[-1] != self._window_dim + 1:
            raise ValueError(
                f"'{self.amp_obs_group}' obs group must be [window ({self._window_dim} = {self.amp_num_frames} frames x "
                f"{self.amp_expert.frame_dim} joints), style gate (1)], got dim {amp.shape[-1]} — check the group's "
                f"terms and history_length against amp_num_frames={self.amp_num_frames} and amp_motion_file."
            )
        return amp[:, : self._window_dim], amp[:, self._window_dim] > 0.5

    def process_env_step(
        self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict[str, torch.Tensor]
    ) -> None:
        """Add the gated style reward and bank the policy window, then defer to PPO."""
        raw_windows, gate = self._split_amp_obs(obs)
        frames = raw_windows.view(-1, self.amp_num_frames, self.amp_expert.frame_dim)
        style = self.discriminator.style_reward(self.amp_normalizer.normalize(frames).flatten(1), self.amp_reward_coef)
        style = style * gate.float() * self.amp_reward_scale
        rewards = rewards + style

        self.amp_replay.insert(raw_windows)
        self._style_reward_sum = self._style_reward_sum + style.mean()
        self._style_reward_steps += 1
        super().process_env_step(obs, rewards, dones, extras)

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(self) -> dict[str, float]:
        """Run the PPO update, then the discriminator's LSGAN updates."""
        batch_size = self.storage.num_envs * self.storage.num_transitions_per_env // self.num_mini_batches
        loss_dict = super().update()

        # The source trained the discriminator inside the PPO optimizer, so its
        # learning rate followed the KL-adaptive schedule; mirror that here.
        for group in self.amp_optimizer.param_groups:
            group["lr"] = self.learning_rate

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_amp_loss = mean_grad_pen = mean_expert_d = mean_policy_d = 0.0
        for _ in range(num_updates):
            policy_windows = self.amp_replay.sample(batch_size).view(-1, self.amp_num_frames, self.amp_expert.frame_dim)
            expert_windows = self.amp_expert.sample(batch_size)
            self.amp_normalizer.update(self._gather(policy_windows[:, -1]))
            self.amp_normalizer.update(self._gather(expert_windows[:, -1]))
            policy_in = self.amp_normalizer.normalize(policy_windows).flatten(1)
            expert_in = self.amp_normalizer.normalize(expert_windows).flatten(1)

            policy_d = self.discriminator(policy_in)
            expert_d = self.discriminator(expert_in)
            expert_loss = torch.nn.functional.mse_loss(expert_d, torch.ones_like(expert_d))
            policy_loss = torch.nn.functional.mse_loss(policy_d, -torch.ones_like(policy_d))
            amp_loss = 0.5 * (expert_loss + policy_loss)
            grad_pen = self.discriminator.gradient_penalty(expert_in, self.amp_grad_pen_scale)

            self.amp_optimizer.zero_grad()
            (amp_loss + grad_pen).backward()
            if self.is_multi_gpu:
                reduce_gradients_in_buckets(
                    self.discriminator.parameters(), self.gpu_world_size, self.grad_reduce_bucket_mb
                )
            self.amp_optimizer.step()

            mean_amp_loss += amp_loss.item()
            mean_grad_pen += grad_pen.item()
            mean_expert_d += expert_d.mean().item()
            mean_policy_d += policy_d.mean().item()

        loss_dict["amp"] = mean_amp_loss / num_updates
        loss_dict["amp_grad_pen"] = mean_grad_pen / num_updates
        loss_dict["amp_expert_d"] = mean_expert_d / num_updates
        loss_dict["amp_policy_d"] = mean_policy_d / num_updates
        loss_dict["amp_style_reward"] = float(self._style_reward_sum) / max(self._style_reward_steps, 1)
        self._style_reward_sum = torch.zeros((), device=self.device)
        self._style_reward_steps = 0
        return loss_dict

    def _gather(self, frames: torch.Tensor) -> torch.Tensor:
        """``frames`` from every rank, concatenated (``frames`` itself on one process)."""
        if not self.is_multi_gpu:
            return frames
        gathered = [torch.empty_like(frames) for _ in range(self.gpu_world_size)]
        torch.distributed.all_gather(gathered, frames.contiguous())
        return torch.cat(gathered)

    # ------------------------------------------------------------------
    # Distributed training
    # ------------------------------------------------------------------

    def broadcast_parameters(self) -> None:
        """PPO's broadcast, plus the discriminator (spectral-norm vectors included) and the frame normalizer."""
        super().broadcast_parameters()
        for module in (self.discriminator, self.amp_normalizer):
            for tensor in module.state_dict().values():
                torch.distributed.broadcast(tensor, src=0)

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save(self) -> dict:
        saved = super().save()
        saved["amp_discriminator_state_dict"] = self.discriminator.state_dict()
        saved["amp_optimizer_state_dict"] = self.amp_optimizer.state_dict()
        saved["amp_normalizer_state_dict"] = self.amp_normalizer.state_dict()
        return saved

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        """PPO's load, plus the discriminator, its optimizer and the frame normalizer.

        A checkpoint without them was trained as plain PPO. Continuing it under AMP changes the reward under a value
        function trained without the style term, so it is refused unless ``amp_warm_start`` says that is the intent
        (the discriminator then starts fresh).
        """
        missing = [key for key in _AMP_CHECKPOINT_KEYS if key not in loaded_dict]
        if missing and not self.amp_warm_start:
            raise ValueError(
                f"This checkpoint has no AMP state (missing {missing}; it has {sorted(loaded_dict)}), so it was "
                "trained as plain PPO. Loading it into AmpPPO would add the style reward to a value function trained "
                "without it. To start AMP from its weights with a fresh discriminator, set amp_warm_start=True "
                "(--override 'runner_options: {algorithm: {amp_warm_start: true}}'); to replay it, use "
                "`eden inference`, which loads it under stock PPO."
            )
        resumed = super().load(loaded_dict, load_cfg, strict)
        if not missing:
            self.discriminator.load_state_dict(loaded_dict["amp_discriminator_state_dict"], strict=strict)
            if load_cfg is None or load_cfg.get("optimizer"):
                self.amp_optimizer.load_state_dict(loaded_dict["amp_optimizer_state_dict"])
            self.amp_normalizer.load_state_dict(loaded_dict["amp_normalizer_state_dict"])
        return resumed

    def train_mode(self) -> None:
        super().train_mode()
        self.discriminator.train()

    def eval_mode(self) -> None:
        super().eval_mode()
        self.discriminator.eval()
