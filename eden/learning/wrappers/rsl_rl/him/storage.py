"""HIM rollout storage — OpenHomie's ``HIMRolloutStorage`` (``HomieRL/rsl_rl/rsl_rl/storage/him_rollout_storage.py``).

Differs from a stock rollout buffer in two ways required by ``HIMPPO``:

* capacity is **doubled** (``num_transitions_per_env *= 2``) because each env step stores
  both the real transition and its sagittal mirror, so GAE is computed over doubled
  on-policy data (``compute_returns`` reshapes with the ``// 2`` stacked axis);
* it stores ``next_critic_observations`` per transition (the estimator's SwAV / velocity
  targets need the *next* privileged obs).
"""

from __future__ import annotations

import torch


class HIMRolloutStorage:
    class Transition:
        """One (real or mirrored) env-step transition awaiting insertion into the buffer."""

        def __init__(self):
            self.observations = None
            self.critic_observations = None
            self.actions = None
            self.rewards = None
            self.dones = None
            self.values = None
            self.actions_log_prob = None
            self.action_mean = None
            self.action_sigma = None
            self.next_critic_observations = None

        def clear(self):
            self.__init__()

    def __init__(self, num_envs, num_transitions_per_env, obs_shape, privileged_obs_shape, actions_shape, device="cpu"):
        self.device = device
        num_transitions_per_env *= 2  # real + mirrored transition per env step
        self.obs_shape = obs_shape
        self.privileged_obs_shape = privileged_obs_shape
        self.actions_shape = actions_shape

        self.observations = torch.zeros(num_transitions_per_env, num_envs, *obs_shape, device=self.device)
        self.privileged_observations = torch.zeros(
            num_transitions_per_env, num_envs, *privileged_obs_shape, device=self.device
        )
        self.next_privileged_observations = torch.zeros(
            num_transitions_per_env, num_envs, *privileged_obs_shape, device=self.device
        )
        self.rewards = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.actions = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.dones = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device).byte()

        self.actions_log_prob = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.values = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.returns = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.advantages = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.mu = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.sigma = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)

        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs
        self.step = 0

    def add_transitions(self, transition: "HIMRolloutStorage.Transition") -> None:
        if self.step >= self.num_transitions_per_env:
            raise AssertionError("Rollout buffer overflow")
        self.observations[self.step].copy_(transition.observations)
        self.privileged_observations[self.step].copy_(transition.critic_observations)
        self.next_privileged_observations[self.step].copy_(transition.next_critic_observations)
        self.actions[self.step].copy_(transition.actions)
        self.rewards[self.step].copy_(transition.rewards.view(-1, 1))
        self.dones[self.step].copy_(transition.dones.view(-1, 1))
        self.values[self.step].copy_(transition.values)
        self.actions_log_prob[self.step].copy_(transition.actions_log_prob.view(-1, 1))
        self.mu[self.step].copy_(transition.action_mean)
        self.sigma[self.step].copy_(transition.action_sigma)
        self.step += 1

    def clear(self) -> None:
        self.step = 0

    def compute_returns(self, last_values, gamma, lam) -> None:
        num_transitions_per_env = self.num_transitions_per_env // 2
        advantage = 0
        resize = lambda x: x.view(num_transitions_per_env, 2, -1, 1)  # noqa: E731
        for step in reversed(range(num_transitions_per_env)):
            if step == num_transitions_per_env - 1:
                next_values = last_values
            else:
                next_values = resize(self.values)[step + 1]
            next_is_not_terminal = 1.0 - resize(self.dones)[step].float()
            delta = resize(self.rewards)[step] + next_is_not_terminal * gamma * next_values - resize(self.values)[step]
            advantage = delta + next_is_not_terminal * gamma * lam * advantage
            resize(self.returns)[step] = advantage + resize(self.values)[step]
        self.advantages = self.returns - self.values
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(num_mini_batches * mini_batch_size, requires_grad=False, device=self.device)

        observations = self.observations.flatten(0, 1)
        critic_observations = self.privileged_observations.flatten(0, 1)
        next_critic_observations = self.next_privileged_observations.flatten(0, 1)
        actions = self.actions.flatten(0, 1)
        values = self.values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)
        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        old_mu = self.mu.flatten(0, 1)
        old_sigma = self.sigma.flatten(0, 1)

        for _ in range(num_epochs):
            for i in range(num_mini_batches):
                batch_idx = indices[i * mini_batch_size : (i + 1) * mini_batch_size]
                yield (
                    observations[batch_idx],
                    critic_observations[batch_idx],
                    actions[batch_idx],
                    next_critic_observations[batch_idx],
                    values[batch_idx],
                    advantages[batch_idx],
                    returns[batch_idx],
                    old_actions_log_prob[batch_idx],
                    old_mu[batch_idx],
                    old_sigma[batch_idx],
                )
