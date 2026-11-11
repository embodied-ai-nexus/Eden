"""VecEnv wrapper that feeds ConTrack's CMDP-PPO (``rsl_rl_contrack``) its per-step reward-group extras.

ConTrack's CMDP-PPO consumes three per-step reward-group scalars from
``extras`` — ``rg`` (task), ``rs`` (style), ``rp`` (penalty) — and trains a
two-channel critic on ``(rg + rp, rs + rp)``. :class:`ContrackVecEnvWrapper`
forwards :meth:`RewardManager.get_group_reward` for the task's declared reward
groups (see ``RewardGroupOptions`` in the task config); the per-episode group
returns are logged by the reward manager itself (``Episode_Reward_Group/<g>``
in the reset extras).

It also applies ``runner_options.check_for_nan`` (the fork's runner has no NaN check of its own), and maintains the
object-tracking-error running sums the
ConTrack ``OnPolicyRunner`` logs unconditionally (``it_obj_pos_err_sum`` /
``it_obj_rot_err_sum`` / ``it_step_count``).

Needs no backend: :class:`~eden.learning.wrappers.rsl_rl_contrack.runner.ContrackOnPolicyRunner` wraps the env in it.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

import torch

from eden.envs.wrappers.rsl_rl_env import RslRlVecEnvWrapper
from eden.utils.geom import quat_error_magnitude

if TYPE_CHECKING:
    from tensordict import TensorDict

    from eden.envs.base import RLEnvBase
    from eden.learning.wrappers.rsl_rl_contrack.options import RslRlContrackRunnerOptions

CMDP_GROUPS = ("rg", "rs", "rp")


class ContrackVecEnvWrapper(RslRlVecEnvWrapper):
    """RslRl wrapper that injects the per-step CMDP reward-group extras.

    Requires the wrapped env's reward terms to be declared in exactly the CMDP
    groups (``rg`` / ``rs`` / ``rp`` via ``RewardGroupOptions``) — enforced at
    construction so a term outside the three channels (which the CMDP critic
    would never see) fails before training starts.
    """

    # Entity whose tracking error feeds the runner's obj-err logs; optional —
    # envs without an entity of this name log zeros.
    object_entity_name: str = "object"

    def __init__(self, env: RLEnvBase, options: RslRlContrackRunnerOptions):
        super().__init__(env, options)
        groups = set(self.unwrapped.reward_manager.group_term_names)
        if groups != set(CMDP_GROUPS):
            raise ValueError(
                f"ConTrack CMDP-PPO requires the reward terms to be partitioned into exactly the groups "
                f"{CMDP_GROUPS}; the env declares {sorted(groups)}. Declare every reward term inside "
                f"rg/rs/rp RewardGroupOptions (a term outside them would be invisible to the CMDP critic)."
            )

        # Read by rsl_rl_contrack.OnPolicyRunner.log() — zeroed there each iteration.
        self.it_obj_pos_err_sum = torch.zeros((), device=self.device)
        self.it_obj_rot_err_sum = torch.zeros((), device=self.device)
        self.it_step_count = 0

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        obs, rew, dones, extras = super().step(actions)
        reward_manager = self.unwrapped.reward_manager
        for group in CMDP_GROUPS:
            extras[group] = reward_manager.get_group_reward(group)
        if self.runner_options.check_for_nan:
            _raise_on_nan(obs, rew, {group: extras[group] for group in CMDP_GROUPS})
        self._accumulate_object_tracking_error(self.unwrapped)
        return obs, rew, dones, extras

    def _accumulate_object_tracking_error(self, env: RLEnvBase) -> None:
        if self.object_entity_name not in env.entities:
            self.it_step_count += 1
            return
        obj = env.entities[self.object_entity_name]
        ref = env.motion_manager.data[self.object_entity_name]
        self.it_obj_pos_err_sum += torch.norm(obj.get_pos() - ref.base_pos_w[:, 0], dim=-1).mean()
        self.it_obj_rot_err_sum += quat_error_magnitude(obj.get_quat(), ref.base_quat_w[:, 0]).mean()
        self.it_step_count += 1


def _raise_on_nan(obs: TensorDict, rewards: torch.Tensor, group_rewards: dict[str, torch.Tensor]) -> None:
    """Raise, naming every offending tensor, when an env output the CMDP-PPO consumes holds NaN."""
    tensors = {f"obs[{key!r}]": value for key, value in obs.items()}
    tensors.update({"rewards": rewards, **{f"extras[{group!r}]": value for group, value in group_rewards.items()}})
    flags = torch.stack([torch.isnan(value).any() for value in tensors.values()]).tolist()  # one host sync
    bad = [name for name, flag in zip(tensors, flags) if flag]
    if bad:
        raise ValueError(
            f"NaN in the env output {bad} (runner_options.check_for_nan=True). A reward or observation term produced "
            "it; find it with `eden smoke --task contrack`, rather than turning check_for_nan off."
        )
