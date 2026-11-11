"""Train the Reacher task with a single-file, CleanRL-style PPO loop on Eden's gymnasium vector env.

---
tier: recipe
capabilities: [rl.cleanrl, rl.gymnasium, task.reaching, robot.yam]
hardware: cpu
optional_deps: []
run: uv run python examples/recipes/rl_libraries/cleanrl_ppo.py -c
---

CleanRL ships algorithms as standalone scripts rather than a library, so its entry point into Eden is
the gymnasium env itself: ``gym.make_vec("Eden/reacher", num_envs=N)`` returns torch tensors on the
sim's device, and this loop (after CleanRL's ``ppo_continuous_action_isaacgym.py``) never leaves it.
The one Eden-specific line bootstraps a timed-out episode from ``info["final_obs"]``, the observation
the episode ended on — the returned one already starts the next episode.

Usage:
    uv run python examples/recipes/rl_libraries/cleanrl_ppo.py              # GPU, 4096 envs
    uv run python examples/recipes/rl_libraries/cleanrl_ppo.py -c -b 64 -s 20
"""

import argparse

import eden as en
import genesis as gs
import gymnasium as gym
import torch
from torch import nn

ROLLOUT_STEPS = 24
LEARNING_EPOCHS = 5
MINI_BATCHES = 4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_COEF = 0.2
ENT_COEF = 0.005
VF_COEF = 1.0
MAX_GRAD_NORM = 1.0
LEARNING_RATE = 1e-3


def mlp(in_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(in_dim, 256), nn.ELU(), nn.Linear(256, 128), nn.ELU(), nn.Linear(128, out_dim))


class Agent(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int):
        super().__init__()
        self.critic = mlp(obs_dim, 1)
        self.actor_mean = mlp(obs_dim, action_dim)
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).flatten()

    def get_action_and_value(self, obs: torch.Tensor, action: torch.Tensor | None = None):
        mean = self.actor_mean(obs)
        dist = torch.distributions.Normal(mean, self.actor_logstd.expand_as(mean).exp())
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action).sum(1), dist.entropy().sum(1), self.get_value(obs)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-c", "--cpu", action="store_true", help="Simulate and train on the CPU.")
    parser.add_argument("-b", "--num-envs", type=int, default=4096, help="Number of parallel envs.")
    parser.add_argument("-s", "--steps", type=int, default=150, help="PPO iterations.")
    args = parser.parse_args()

    en.init(backend=gs.cpu if args.cpu else gs.gpu, log_root_path="logs/reacher_cleanrl", performance_mode=True)

    from eden.envs.wrappers.gymnasium_env import register_gymnasium_envs

    register_gymnasium_envs()
    num_envs = 4 if en.is_smoke() else args.num_envs
    iterations = 1 if en.is_smoke() else args.steps
    # Staggered episode starts, as `eden train` uses: without them every env times out on the same step.
    envs = gym.make_vec("Eden/reacher", num_envs=num_envs, random_episode_starts=True, seed=0)
    device = envs.device

    obs_dim = envs.single_observation_space["policy"].shape[0]
    action_dim = envs.single_action_space.shape[0]
    agent = Agent(obs_dim, action_dim).to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=LEARNING_RATE, eps=1e-5)

    obs_buf = torch.zeros(ROLLOUT_STEPS, num_envs, obs_dim, device=device)
    actions_buf = torch.zeros(ROLLOUT_STEPS, num_envs, action_dim, device=device)
    logprobs_buf, rewards_buf, dones_buf, values_buf = (
        torch.zeros(ROLLOUT_STEPS, num_envs, device=device) for _ in range(4)
    )

    obs, _ = envs.reset(seed=0)
    next_obs = obs["policy"]
    for iteration in range(1, iterations + 1):
        reward_sum = torch.zeros((), device=device)
        episode_return_sum = torch.zeros((), device=device)
        episode_count = torch.zeros((), device=device)
        episode_log = {}
        for step in range(ROLLOUT_STEPS):
            obs_buf[step] = next_obs
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
            actions_buf[step], logprobs_buf[step], values_buf[step] = action, logprob, value

            obs, reward, terminated, truncated, info = envs.step(action)
            # A timed-out episode did not end: bootstrap it from the observation it was cut off at. `where`, not a
            # product, so the rows without a valid final observation (zero, or a solver error's) add nothing.
            with torch.no_grad():
                final_value = agent.get_value(info["final_obs"]["policy"])
            time_out = truncated & ~terminated & info["_final_obs"]
            rewards_buf[step] = reward + GAMMA * torch.where(time_out, final_value, 0.0)
            dones_buf[step] = (terminated | truncated).float()
            next_obs = obs["policy"]
            reward_sum += reward.mean()
            episode_return_sum += info["episode"]["r"].sum()
            episode_count += info["_episode"].sum()
            # Eden's per-term episode statistics; empty on a step where no episode ended.
            episode_log = info["log"] or episode_log

        # Generalized advantage estimation; dones_buf[t] ends the episode after step t.
        with torch.no_grad():
            next_value = agent.get_value(next_obs)
            advantages = torch.zeros_like(rewards_buf)
            last_gae = torch.zeros(num_envs, device=device)
            for t in reversed(range(ROLLOUT_STEPS)):
                next_values = next_value if t == ROLLOUT_STEPS - 1 else values_buf[t + 1]
                not_done = 1.0 - dones_buf[t]
                delta = rewards_buf[t] + GAMMA * next_values * not_done - values_buf[t]
                last_gae = delta + GAMMA * GAE_LAMBDA * not_done * last_gae
                advantages[t] = last_gae
            returns = advantages + values_buf

        b_obs, b_actions = obs_buf.reshape(-1, obs_dim), actions_buf.reshape(-1, action_dim)
        b_logprobs, b_advantages = logprobs_buf.reshape(-1), advantages.reshape(-1)
        b_returns, b_values = returns.reshape(-1), values_buf.reshape(-1)
        batch_size = b_obs.shape[0]
        minibatch_size = batch_size // MINI_BATCHES
        for _ in range(LEARNING_EPOCHS):
            for indices in torch.randperm(batch_size, device=device).split(minibatch_size):
                _, new_logprob, entropy, new_value = agent.get_action_and_value(b_obs[indices], b_actions[indices])
                ratio = (new_logprob - b_logprobs[indices]).exp()
                mb_advantages = b_advantages[indices]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                policy_loss = torch.max(
                    -mb_advantages * ratio, -mb_advantages * ratio.clamp(1 - CLIP_COEF, 1 + CLIP_COEF)
                ).mean()
                value_clipped = b_values[indices] + (new_value - b_values[indices]).clamp(-CLIP_COEF, CLIP_COEF)
                value_loss = torch.max(
                    (new_value - b_returns[indices]) ** 2, (value_clipped - b_returns[indices]) ** 2
                ).mean()
                loss = policy_loss + VF_COEF * value_loss - ENT_COEF * entropy.mean()
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), MAX_GRAD_NORM)
                optimizer.step()

        message = f"iteration {iteration}/{iterations}: mean reward per step {reward_sum.item() / ROLLOUT_STEPS:.4f}"
        if episode_count.item() > 0:
            message += f", {int(episode_count.item())} episodes ended, mean return {(episode_return_sum / episode_count).item():.3f}"
            message += "".join(f", {k} {v:.3f}" for k, v in episode_log.items() if k.startswith("Episode_Reward/"))
        en.logger.info(message)

    envs.close()


if __name__ == "__main__":
    main()
