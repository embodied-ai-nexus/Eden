"""Diagnostic metrics for in-hand cube reorientation.

Port of ``wuji-mjlab``'s ``metrics.py``. None of these shape the reward; they exist because
this task fails in several distinguishable ways that all look like "low reward":

* the policy freezes to avoid the action-rate and torque penalties (``dofs_vel_rms`` near
  zero, ``hand_object_contact`` steady),
* it jams its fingers into each other (``self_collision`` high -- from the metric term
  in ``contacts.py``),
* it saturates the weak distal actuators and stops making progress
  (``torque_saturation`` near 1),
* it lifts the cube out of the hand rather than rotating it in place
  (``cube_height_above_palm`` climbing, ``cage_escape`` non-zero),
* or it reorients fine but too slowly (``goal_interval`` high while
  ``goal_reach_count`` still rises).

The names above are the task config's metric keys. Two upstream metrics are not re-declared:
``fingertip_contact_count`` is the value of the ``hand_object_contact`` term and
``cage_escape_frequency`` is derivable from the ``cage_escape`` term, both in their own modules.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from genesis.utils.geom import inv_transform_by_quat

from eden.managers import METRIC_TERM_REGISTRY, MetricTerm
from eden.tasks.reorient.constants import PALM_LINK

if TYPE_CHECKING:
    from eden.envs.base import EnvBase
    from eden.options.managers.metrics import MetricTermOptions


class _HandDofsMetric(MetricTerm):
    """Shared plumbing for the joint-space metrics.

    Parameters
    ----------
    entity_name:
        Scene entity name of the hand.
    """

    entity_name: str = "robot"
    contributes_to_success: bool = False

    def build(self) -> None:
        entity = self._env.entities[self.entity_name]
        _, dofs_idx = entity.find_named_dofs_idx_local(entity.dofs_name, preserve_order=True)
        self._dofs_idx = torch.as_tensor(dofs_idx, dtype=torch.long, device=self.device)


@METRIC_TERM_REGISTRY.register()
class DofsVelRms(_HandDofsMetric):
    """RMS joint velocity. Near zero means the policy has learned to hold still."""

    def compute(self) -> torch.Tensor:
        vel = self.resolve_entity(self.entity_name).get_dofs_vel(dofs_idx_local=self._dofs_idx)
        return torch.sqrt(torch.mean(vel * vel, dim=-1))


@METRIC_TERM_REGISTRY.register()
class DofsAccRms(_HandDofsMetric):
    """RMS joint acceleration, from finite-differenced velocity.

    A jerky policy scores high here while looking fine on ``dofs_vel_rms``. The previous
    velocity is a term-owned buffer -- upstream keeps it in an env-level cache and needs a
    dedicated reset event to clear it.
    """

    def __init__(self, env: EnvBase, options: MetricTermOptions):
        super().__init__(env, options)
        self._prev_vel: torch.Tensor | None = None

    def compute(self) -> torch.Tensor:
        vel = self.resolve_entity(self.entity_name).get_dofs_vel(dofs_idx_local=self._dofs_idx)
        if self._prev_vel is None:
            # Allocated outside inference mode so ``reset`` can write to it later (see
            # ``rewards.ActionRateL2Smooth`` for the same constraint).
            with torch.inference_mode(False):
                self._prev_vel = torch.zeros_like(vel)
        acc = (vel - self._prev_vel) / self._env.dt
        self._prev_vel.copy_(vel)
        return torch.sqrt(torch.mean(acc * acc, dim=-1))

    def reset(self, envs_idx: torch.Tensor | None = None) -> None:
        if self._prev_vel is None:
            return
        if envs_idx is None:
            self._prev_vel.zero_()
        else:
            self._prev_vel[envs_idx] = 0.0


@METRIC_TERM_REGISTRY.register()
class TorqueSaturationRatio(_HandDofsMetric):
    """Fraction of DOFs commanding more than ``threshold_frac`` of their effort limit.

    Parameters
    ----------
    entity_name:
        Scene entity name of the hand.
    threshold_frac:
        Fraction of the effort limit that counts as saturated.
    """

    threshold_frac: float = 0.9

    def build(self) -> None:
        super().build()
        entity = self._env.entities[self.entity_name]
        _, upper = entity.get_dofs_force_range(dofs_idx_local=self._dofs_idx)
        self._limits = upper.clamp_min(1e-3)

    def compute(self) -> torch.Tensor:
        torque = self.resolve_entity(self.entity_name).get_dofs_control_force(dofs_idx_local=self._dofs_idx)
        return (torque.abs() > self.threshold_frac * self._limits).float().mean(dim=-1)


@METRIC_TERM_REGISTRY.register()
class CubeHeightAbovePalm(MetricTerm):
    """Object offset along the palm normal, in the palm frame.

    Palm-frame rather than world-z so it is invariant to the wrist randomization: a large
    positive value always means "lifted off the palm", regardless of how the hand is tilted.

    Parameters
    ----------
    hand_entity_name:
        Scene entity name of the hand.
    object_entity_name:
        Scene entity name of the object.
    palm_link:
        Link pattern selecting the palm.
    """

    hand_entity_name: str = "robot"
    object_entity_name: str = "cube"
    palm_link: str = PALM_LINK
    contributes_to_success: bool = False

    def build(self) -> None:
        hand = self._env.entities[self.hand_entity_name]
        self._env.entities[self.object_entity_name]
        _, palm_idx = hand.find_named_links_idx_local(self.palm_link, preserve_order=True)
        self._palm_idx = torch.as_tensor(palm_idx, dtype=torch.long, device=self.device)

    def compute(self) -> torch.Tensor:
        hand = self.resolve_entity(self.hand_entity_name)
        obj = self.resolve_entity(self.object_entity_name)
        palm_pos = hand.get_links_pos(ls_idx_local=self._palm_idx).squeeze(1)
        palm_quat = hand.get_links_quat(ls_idx_local=self._palm_idx).squeeze(1)
        # Palm-local +X is the palm normal for this hand (see the robot options' axis note).
        return inv_transform_by_quat(obj.get_pos() - palm_pos, palm_quat)[:, 0]


@METRIC_TERM_REGISTRY.register()
def goal_reach_count(env: EnvBase, *, command_name: str = "reorient") -> torch.Tensor:
    """Goals banked so far this episode. The headline task-progress number."""
    return env.command_manager.get_term(command_name).goal_reach_count.float()


@METRIC_TERM_REGISTRY.register()
def goal_interval(env: EnvBase, *, command_name: str = "reorient") -> torch.Tensor:
    """Steps since the last goal switch.

    A sawtooth, so its episode mean is roughly half the true interval when switching is
    regular -- and grows without bound when the policy stops reaching goals at all. Read as
    a trend, not an absolute.
    """
    return env.command_manager.get_term(command_name).goal_timer.float()


@METRIC_TERM_REGISTRY.register()
def cube_survival_steps(env: EnvBase) -> torch.Tensor:
    """Current episode length. Its mean rises as the policy stops dropping the cube."""
    return env.episode_length_buf.float()


__all__ = [
    "CubeHeightAbovePalm",
    "DofsAccRms",
    "DofsVelRms",
    "TorqueSaturationRatio",
    "cube_survival_steps",
    "goal_interval",
    "goal_reach_count",
]
