"""Reward terms for in-hand cube reorientation.

Port of ``wuji-mjlab``'s ``rewards.py``. Three of upstream's nine reward terms map onto
Eden built-ins and are not re-implemented here: the combined 1st+2nd-order action-rate
penalty is ``rewards.ActionRateL2Smooth``, the finger self-collision penalty and the cage
escape penalty are ``rewards.MetricReward`` pulls off the metric terms in
``contacts.py`` / ``cage.py``.

The shaping worth understanding before tuning any weight: ``orientation_alignment`` pays a
*plateau* -- full credit anywhere inside the success threshold, decaying linearly with the
error beyond it -- while ``hold_escalation`` pays an amount that grows the longer the cube is held on
target. Together they mean "get there, then stay there", which is what makes a single
episode produce a stream of reorientations rather than one lucky alignment.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from eden.managers import REWARD_TERM_REGISTRY, RewardTerm
from eden.tasks.reorient.constants import TIP_LINKS
from eden.utils.geom import quat_error_magnitude
from eden.utils.string import term_subject

if TYPE_CHECKING:
    from eden.envs.base import EnvBase
    from eden.options.managers.rewards import RewardTermOptions


def tolerance_linear(value: torch.Tensor, bound: float, margin: float) -> torch.Tensor:
    """Plateau-and-decay kernel: 1 inside ``[0, bound]``, linear to 0 at ``bound + margin``.

    Parameters
    ----------
    value:
        Non-negative error.
    bound:
        Half-width of the full-credit plateau.
    margin:
        Distance past ``bound`` over which the credit decays to zero.

    Returns
    -------
    torch.Tensor
        Credit in [0, 1], same shape as ``value``.
    """
    if margin <= 0.0:
        return (value <= bound).float()
    return torch.where(value <= bound, torch.ones_like(value), (1.0 - (value - bound) / margin).clamp(min=0.0))


@REWARD_TERM_REGISTRY.register()
def orientation_alignment(
    env: EnvBase,
    *,
    command_name: str = "reorient",
    object_entity_name: str = "cube",
    bound: float = 0.2,
    margin: float = torch.pi,
) -> torch.Tensor:
    """Plateau-and-decay credit on the cube's orientation error against the goal.

    A plateau rather than a peak: inside ``bound`` (the same threshold the command term
    counts as success) every orientation is worth the same, so the policy is not pushed to
    chase the last few degrees at the cost of dropping the cube. Outside, the credit decays
    linearly to zero at ``bound + margin``; with the default ``margin = pi`` that lies past the
    largest possible error, so there is usable gradient even from a completely wrong orientation.
    """
    goal_quat = env.command_manager.get_term(command_name).command
    ori_error = quat_error_magnitude(env.entities[object_entity_name].get_quat(), goal_quat)
    return tolerance_linear(ori_error, bound=bound, margin=margin)


@REWARD_TERM_REGISTRY.register()
def hold_escalation(env: EnvBase, *, command_name: str = "reorient") -> torch.Tensor:
    """Time-escalating credit for holding the cube on target inside the success window.

    Returns the in-window hold counter (0 outside the window or off target), so each
    successive on-target step is worth more than the last. This replaces upstream's earlier
    sparse +100-on-success plus a flat per-step hold bonus: the escalation delivers a
    comparable total per goal cycle while giving a gradient that distinguishes "held for two
    steps" from "held for fifteen".

    Reads the command's counters **live**, with no snapshot: the success machine is measured in
    ``InHandReorientCommand.post_physics_refresh`` at phase 5 and this reward runs at phase 9 of
    the same step, so the step that earns the cycle's largest counter is paid before the goal
    switch at phase 17 clears it. Upstream runs the machine in ``_update_command`` and carries
    three ``reward_*`` shadow buffers to reach back across that boundary -- see the hook rule in
    ``docs/guides/terms.md``, whose worked example is this term.
    """
    command = env.command_manager.get_term(command_name)
    return (command.in_success_window & command.within_threshold).float() * command.reward_hold_counter


@REWARD_TERM_REGISTRY.register()
def palm_detach(env: EnvBase, *, metric_name: str = "hand_object_contact") -> torch.Tensor:
    """1.0 while the cube is held by the distal phalanges *and* clear of the palm.

    Both halves are needed. Rewarding only "not touching the palm" pays a cube that has
    left the hand entirely; rewarding only "touching the fingertips" pays a cube rolling
    across the palm with a finger resting on it. Together they describe the fingertip grasp
    that in-hand reorientation actually requires.
    """
    term = env.metric_manager.get_term(metric_name)
    return (~term.palm_or_proximal_contact & term.distal_contact).float()


@REWARD_TERM_REGISTRY.register()
def dofs_pos_deviation_l2(env: EnvBase, *, entity_name: str = "robot") -> torch.Tensor:
    """Squared deviation of the joints from the entity's default pose.

    L2, not the L1 of ``rewards.similar_to_default``: upstream uses a squared kernel here,
    which barely penalises the small excursions normal reorientation needs while pulling
    hard against a finger driven to an extreme.
    """
    entity = env.entities[entity_name]
    return torch.sum(torch.square(entity.get_dofs_pos() - entity.default_dofs_pos), dim=-1)


@REWARD_TERM_REGISTRY.register()
class NormalizedTorque(RewardTerm):
    """Squared PD control torque, normalized per DOF by its effort limit.

    Normalizing matters on this hand: the effort ceilings span 0.15-0.65 Nm, so an
    unnormalized sum would let the policy freely saturate the weak distal joints while a
    single strong proximal joint dominated the penalty.

    Parameters
    ----------
    entity_name:
        Scene entity name of the actuated entity.
    """

    entity_name: str = "robot"

    def build(self) -> None:
        entity = self._env.entities[self.entity_name]
        _, dofs_idx = entity.find_named_dofs_idx_local(entity.dofs_name, preserve_order=True)
        self._dofs_idx = torch.as_tensor(dofs_idx, dtype=torch.long, device=self.device)
        _, upper = entity.get_dofs_force_range(dofs_idx_local=self._dofs_idx)
        self._limits = upper.clamp_min(1e-3)

    def compute(self) -> torch.Tensor:
        torque = self.resolve_entity(self.entity_name).get_dofs_control_force(dofs_idx_local=self._dofs_idx)
        return torch.sum(torch.square(torque / self._limits), dim=-1)


@REWARD_TERM_REGISTRY.register()
class FingertipSlide(RewardTerm):
    """Relative speed between each contacting fingertip and the object, summed.

    Penalizes the cube *sliding* under the fingertips as opposed to being rolled by them:
    a rolling contact has near-zero relative velocity at the contact point, a slipping one
    does not. Gated on contact, so moving a finger through free space costs nothing.

    Parameters
    ----------
    entity_name:
        Scene entity name of the hand.
    object_entity_name:
        Scene entity name of the object.
    tip_links:
        Link patterns for the fingertip frames whose velocities are read. Must resolve in
        the same order as the contact flags on the ``metric_name`` term.
    metric_name:
        Metric term supplying the per-fingertip contact flags.
    """

    entity_name: str = "robot"
    object_entity_name: str = "cube"
    tip_links: list[str] = [TIP_LINKS]
    metric_name: str = "hand_object_contact"

    def __init__(self, env: EnvBase, options: RewardTermOptions):
        super().__init__(env=env, options=options)
        self._tip_idx: torch.Tensor | None = None

    def build(self) -> None:
        entity = self._env.entities[self.entity_name]
        self._env.entities[self.object_entity_name]
        _, tip_idx = entity.find_named_links_idx_local(self.tip_links, preserve_order=True)
        if len(tip_idx) == 0:
            raise ValueError(
                f"{term_subject(self)}: tip_links={self.tip_links!r} matched no links. "
                f"Available links: {[link.name for link in entity.links]}"
            )
        self._tip_idx = torch.as_tensor(tip_idx, dtype=torch.long, device=self.device)
        # The metric manager is constructed before the reward manager (``_manager_order``), so a
        # bad ``metric_name`` raises here rather than mid-rollout.
        self._contact = self._env.metric_manager.get_term(self.metric_name)

    def compute(self) -> torch.Tensor:
        hand = self.resolve_entity(self.entity_name)
        obj = self.resolve_entity(self.object_entity_name)
        tip_vel = hand.get_links_vel(ls_idx_local=self._tip_idx)  # (N, n_tips, 3)
        rel_speed = torch.norm(tip_vel - obj.get_vel().unsqueeze(1), dim=-1)  # (N, n_tips)
        in_contact = self._contact.tip_contact.float()
        if in_contact.shape[1] != rel_speed.shape[1]:
            raise ValueError(
                f"{term_subject(self)}: {in_contact.shape[1]} contact flags from metric '{self.metric_name}' but "
                f"{rel_speed.shape[1]} fingertip links from tip_links={self.tip_links!r}; the two must line up."
            )
        return torch.sum(rel_speed * in_contact, dim=-1)


__all__ = [
    "FingertipSlide",
    "NormalizedTorque",
    "dofs_pos_deviation_l2",
    "hold_escalation",
    "orientation_alignment",
    "palm_detach",
    "tolerance_linear",
]
