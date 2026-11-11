"""Eden's two-agent rl_games wrapper plus the NaN quarantine AssistMimic's training needs."""

from __future__ import annotations

import torch

from eden.envs.wrappers.rl_games_env import RlGamesVecEnvWrapper


class QuarantineVecEnv(RlGamesVecEnvWrapper):
    """Ends and resets any env whose observation or reward came back non-finite, paying it nothing.

    The physics can hand back NaN without Genesis flagging a solver error (a violent contact, or NaN left in the
    solver for a step after a reset), and rl_games has no guard: one NaN reward makes every advantage of the
    rollout NaN, and the value normalizer's statistics with them, for good. Such an env ends here as terminated
    (no value bootstrap), with reward 0, and is reset at once; its observation is recomputed from the reset.
    ``nan_envs`` counts them. Checking costs one GPU->CPU read per step, as the trainer this replaces did.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nan_envs = torch.zeros((), dtype=torch.long, device=self.device)

    def step(self, actions: torch.Tensor):
        obs, rew, dones, extras = super().step(actions)
        a = self.get_number_of_agents()
        finite = torch.isfinite(obs).all(dim=1) & torch.isfinite(rew)
        bad = ~finite.view(self.num_envs, a).all(dim=1)
        if bad.any():
            self.nan_envs += bad.sum()
            obs_dict, _ = self.unwrapped.reset(bad)
            obs = self._agent_obs(obs_dict)
            bad_rows = bad.repeat_interleave(a)
            rew = torch.where(bad_rows, 0.0, rew)
            dones = dones | bad_rows.to(dones.dtype)
            extras["time_outs"] = extras["time_outs"] & ~bad_rows
        return obs, rew, dones, extras
