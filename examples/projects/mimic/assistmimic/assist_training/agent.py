"""rl_games pieces for AssistMimic: PPO with its own critic learning rate, and a ``progress.csv`` log.

rl_games' continuous PPO trains one Adam over the whole model at one learning rate. AssistMimic in Eden trains
the role actors at ``learning_rate`` and the critic at ``critic_learning_rate`` (a key this agent adds to the
runner's ``config``); :class:`AssistA2CAgent` is that agent, registered over rl_games' ``a2c_continuous``.
"""

from __future__ import annotations

import csv
import os
import time

import torch
from rl_games.algos_torch.a2c_continuous import A2CAgent
from rl_games.common import datasets
from rl_games.common.algo_observer import AlgoObserver
from torch import optim


class PairShuffledDataset(datasets.PPODataset):
    """rl_games' PPO dataset, with the order of the envs' trajectories shuffled before every mini-epoch.

    rl_games' continuous PPO cuts each mini-epoch's minibatches as the same contiguous slices of the batch
    (rows ``agent * horizon + t``). This permutes whole blocks of ``block`` rows instead — one env's
    caregiver-and-recipient trajectories, ``2 * horizon`` rows — so every minibatch is a fresh random set of envs
    and still holds as many caregiver rows as recipient ones (``AssistActorCritic`` splits it in half by role).
    """

    def __init__(self, *args, block: int, **kwargs):
        super().__init__(*args, **kwargs)
        if self.batch_size % block or self.minibatch_size % block:
            raise ValueError(
                f"PairShuffledDataset: batch {self.batch_size} and minibatch {self.minibatch_size} must be multiples "
                f"of one env's two-agent trajectories ({block} rows)."
            )
        self.block = block

    def apply_permutation(self):
        perm = torch.randperm(self.batch_size // self.block, device=self.device)
        idx = (perm[:, None] * self.block + torch.arange(self.block, device=self.device)).flatten()
        for key, value in self.values_dict.items():
            if key in self.special_names or value is None:
                continue
            if isinstance(value, dict):
                self.values_dict[key] = {k: v[idx] for k, v in value.items()}
            else:
                self.values_dict[key] = value[idx]


class AssistA2CAgent(A2CAgent):
    """rl_games' ``a2c_continuous`` with the actors at ``learning_rate`` and the critic at ``critic_learning_rate``,
    and minibatches of envs drawn afresh each mini-epoch (:class:`PairShuffledDataset`).

    The model's ``a2c_network`` must be an :class:`~assist_training.network.AssistActorCritic`.
    """

    def __init__(self, base_name, params):
        super().__init__(base_name, params)
        net = self.model.a2c_network
        if self.num_agents != 2:
            raise ValueError(
                f"AssistA2CAgent trains a caregiver and a recipient per env; the env has {self.num_agents}."
            )
        if "critic_learning_rate" not in self.config:
            raise KeyError(
                "AssistA2CAgent: the runner config has no 'critic_learning_rate'; set "
                "runner_dict['params']['config']['critic_learning_rate'] before runner.load()."
            )
        self.critic_lr = float(self.config["critic_learning_rate"])
        grouped = {id(p) for p in (*net.actors.parameters(), *net.critic.parameters())}
        missed = [n for n, p in self.model.named_parameters() if id(p) not in grouped]
        if missed:
            raise ValueError(f"AssistA2CAgent: parameters {missed} are in neither the actor nor the critic group.")
        self.optimizer = optim.Adam(
            [
                {"params": net.actors.parameters(), "lr": self.last_lr},
                {"params": net.critic.parameters(), "lr": self.critic_lr},
            ],
            eps=1e-08,
            weight_decay=self.weight_decay,
            fused=True,
        )
        self.dataset = PairShuffledDataset(
            self.batch_size,
            self.minibatch_size,
            self.is_discrete,
            self.is_rnn,
            self.ppo_device,
            self.seq_length,
            permute=True,
            block=2 * self.horizon_length,
        )

    def _set_group_lrs(self) -> None:
        actors, critic = self.optimizer.param_groups
        actors["lr"], critic["lr"] = self.last_lr, self.critic_lr

    # rl_games writes ``last_lr`` into every param group after each step and each schedule update.
    def update_lr(self, lr):
        super().update_lr(lr)
        self._set_group_lrs()

    def train_actor_critic(self, input_dict):
        result = super().train_actor_critic(input_dict)
        self._set_group_lrs()
        return result

    # A restored optimizer carries the checkpoint's learning rates; this run's apply from its first step.
    def set_full_state_weights(self, weights, set_epoch=True):
        super().set_full_state_weights(weights, set_epoch=set_epoch)
        self._set_group_lrs()


def register_agent(runner) -> None:
    """Make ``runner`` build :class:`AssistA2CAgent` for ``algo: a2c_continuous``."""
    runner.algo_factory.register_builder("a2c_continuous", lambda **kwargs: AssistA2CAgent(**kwargs))


class ProgressLog(AlgoObserver):
    """One ``progress.csv`` row per epoch: episode length, termination fraction, each role's per-step reward.

    The env must be two agents interleaved by env (``RlGamesPpoRunnerOptions.agents``), so a rollout's reward
    rows ``e * 2 + r`` are role ``r`` of env ``e``. Rewards are the shaped ones (``reward_scale``) the agent
    trained on. ``extra_log(env)`` adds columns.
    """

    def __init__(self, path: str, env, extra_log=None):
        super().__init__()
        self.path, self.env, self.extra_log = path, env, extra_log
        self._new = not os.path.exists(path)

    def after_init(self, algo):
        self.algo = algo
        self.motion = self.env.command_manager.get_term("motion")
        dev = algo.ppo_device
        self.ep_term, self.ep_done = torch.zeros((), device=dev), torch.zeros((), device=dev)
        self.rew = torch.zeros(2, device=dev)
        self._t = time.time()

    def process_infos(self, infos, done_indices):
        # ``done_indices`` are each finished env's first agent row (rl_games takes every ``num_agents``-th).
        time_outs = infos["time_outs"][done_indices.squeeze(-1)]
        self.ep_done += time_outs.numel()
        self.ep_term += (~time_outs.bool()).sum()

    def after_steps(self):
        rewards = self.algo.experience_buffer.tensor_dict["rewards"]  # (horizon, 2N, 1)
        self.rew = rewards.view(rewards.shape[0], -1, 2).mean(dim=(0, 1))

    def after_print_stats(self, frame, epoch_num, total_time):
        now = time.time()
        algo = self.algo
        psi = self.motion.psi
        values = [self.ep_term, self.ep_done, self.rew[0], self.rew[1], algo.vec_env.nan_envs]
        if psi is not None:
            values += [psi.stored_total, psi.restored_total]
        values = torch.stack([v.float() for v in values]).tolist()
        ep_term, ep_done, rew_caregiver, rew_recipient, nan_envs = values[:5]
        row = {
            "iter": epoch_num - 1,
            "ep_len": float(algo.game_lengths.get_mean()) if algo.game_lengths.current_size > 0 else 0.0,
            "term_frac": ep_term / max(ep_done, 1.0),
            "rew_caregiver": rew_caregiver,
            "rew_recipient": rew_recipient,
            "fps": algo.curr_frames / max(now - self._t, 1e-9),
            "nan_envs": int(nan_envs),
        }
        if psi is not None:
            row["psi_stored"], row["psi_restarts"] = int(values[5]), int(values[6])
        if self.extra_log is not None:
            row.update(self.extra_log(self.env))
        self._t = now
        self.ep_term.zero_()
        self.ep_done.zero_()
        with open(self.path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if self._new:
                w.writeheader()
                self._new = False
            w.writerow(row)
        print(" | ".join(f"{k} {v:.4g}" if isinstance(v, float) else f"{k} {v}" for k, v in row.items()), flush=True)
