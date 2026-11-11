"""EMA-smoothed joint-position action with a startup hold.

Port of ``wuji-mjlab``'s ``JointPositionOffsetEMAAction``:

    raw_target_t = clamp(default_pos + clamp(a_t, -1, 1) * scale, soft_limits)
    target_t     = alpha * raw_target_t + (1 - alpha) * target_{t-1}

plus a warmup window at the start of each episode during which the hand simply holds its
default pose, so the fingers settle onto the cube before the policy starts moving them.
"""

from __future__ import annotations

import torch

from eden.managers import ACTION_TERM_REGISTRY
from eden.managers.terms.actions.joint_actions import ImplicitPDController
from eden.utils.string import term_subject


@ACTION_TERM_REGISTRY.register()
class JointPositionOffsetEma(ImplicitPDController):
    """Implicit-PD position control with action clipping, EMA smoothing and a warmup hold.

    Three details are load-bearing; getting any one of them wrong breaks training:

    * **The raw action is clipped to [-1, 1] for the target only, never in place.** That
      bounds the commanded target (without it an out-of-range target makes
      ``control_dofs_pos`` drive the joint into its stop at full effort -- bang-bang
      fingers) while leaving ``self._raw_action`` unclipped, which is what the
      ``action_rate`` penalty reads. Both halves matter, and this is exactly upstream's
      arrangement: a runner-level ``clip_actions`` would clip the action before the env
      sees it at all, so the penalty would stop growing outside the box and the policy
      could park its mean out there for free. See the runner note in ``config.py``.
    * **The target is clamped to the soft DOF limits**, the equivalent of a MuJoCo position
      actuator's ``ctrlrange``. Eden derives the soft range with the same 0.9 factor about
      the midpoint that mjlab/IsaacLab use, so this matches upstream exactly.
    * **The EMA state is seeded from, and reset to, the default pose**, so step 0 of an
      episode commands a hold rather than a jump from zero.

    Parameters
    ----------
    ema_alpha:
        Blend weight on the newly commanded target. 1.0 disables smoothing.
    warmup_time_s:
        Episode time during which the default pose is commanded regardless of the action.
    """

    ema_alpha: float = 0.5
    warmup_time_s: float = 0.4

    def build(self) -> None:
        super().build()
        if not 0.0 < self.ema_alpha <= 1.0:
            raise ValueError(f"{term_subject(self)}: ema_alpha must be in (0, 1], got {self.ema_alpha}.")
        if self.warmup_time_s < 0.0:
            raise ValueError(f"{term_subject(self)}: warmup_time_s must be >= 0, got {self.warmup_time_s}.")

        # ``effective_offset`` already folds in ``default_dofs_pos`` for
        # ``reference_source=DEFAULT``; broadcast it so the EMA state is per-env.
        self._default_target = self.effective_offset.expand(self.num_envs, -1).clone()
        self._ema_target = self._default_target.clone()

        # Columns rather than ``get_soft_dofs_pos_limits``: ``compute`` clamps against these every step, and the
        # stored ``(1 or num_envs, n, 2)`` keeps a shared row broadcast instead of one copy per env.
        soft_limits = self.entity.soft_dofs_pos_limits[:, self.entity.find_named_dofs(self.dofs_name).columns]
        self._target_lower = soft_limits[..., 0].expand(self.num_envs, -1)
        self._target_upper = soft_limits[..., 1].expand(self.num_envs, -1)

    def compute(self, actions: torch.Tensor) -> None:
        self._prev_action[:] = self._raw_action
        self._raw_action[:] = actions

        raw_target = self._compute_pos_target(self._raw_action.clamp(-1.0, 1.0))
        raw_target = raw_target.clamp(self._target_lower, self._target_upper)
        self._ema_target = self.ema_alpha * raw_target + (1.0 - self.ema_alpha) * self._ema_target

        in_warmup = (self._env.episode_length_buf * self._env.dt < self.warmup_time_s).unsqueeze(-1)
        self._processed_action = torch.where(in_warmup, self._default_target, self._ema_target)
        # Carry the held pose into the EMA state too, so the first post-warmup step blends
        # from the default pose rather than from a target the hand never tracked.
        self._ema_target = self._processed_action
        if self._modifier is not None:
            self._processed_action = self._modifier.modify_processed_action(self._processed_action)

    def reset(self, envs_idx: (slice | torch.Tensor) | None = None) -> None:
        super().reset(envs_idx)
        if envs_idx is None:
            envs_idx = slice(None)
        self._ema_target[envs_idx] = self._default_target[envs_idx]


__all__ = ["JointPositionOffsetEma"]
