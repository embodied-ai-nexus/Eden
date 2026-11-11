"""AssistMimic observation terms for one agent of the caregiver/recipient pair.

Each agent's observation group is, in AssistMimic's order (``HumanoidImInterxHelpUp._compute_observations``)::

    [ self (358) | imitation v6 (576) | self wrist contact (2) | self wrist force (6) |
      partner v2 (294) | partner rel. to own wrists (144) | partner wrist contact (2) |
      previous action (69) | role (1) ]

The first 934 dims are exactly PHC's input, which the pretrained ``phc_3`` actor consumes; the rest are
AssistMimic's auxiliary features, reduced from SMPL-X fingers to SMPL wrists (a "wrist" is the wrist and
hand links together, since SMPL's hand is one rigid palm). AssistMimic's future-trajectory features are
omitted: its network drops them before the actor and the critic.
"""

from __future__ import annotations

import torch
from genesis.utils.geom import inv_quat

from eden.managers.observation_manager import OBSERVATION_TERM_REGISTRY, ObservationTerm
from eden.managers.terms.utils import resolve_link_idx
from eden.tasks.assistmimic.custom_terms.commands import NUM_BODIES, PARTNER, ROLES
from eden.tasks.assistmimic.custom_terms.phc_math import heading_quat_inv, imitation_obs_v6, rotate, self_obs
from eden.utils.geom import quat_mul, quat_to_tan_norm
from eden.utils.string import term_subject

WRIST_LINKS = (("L_Wrist", "L_Hand"), ("R_Wrist", "R_Hand"))


def wrist_hand_idx(entity, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Link indices of the left and right wrist, and of the left and right hand (``WRIST_LINKS``)."""
    return (
        resolve_link_idx(entity, [w for w, _ in WRIST_LINKS], device),
        resolve_link_idx(entity, [h for _, h in WRIST_LINKS], device),
    )


def body_state(entity) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Link pose, COM linear velocity and angular velocity, ``(N, 24, *)``, world frame, quats wxyz.

    Isaac Gym's rigid-body linear velocity is the COM's (verified against a PHC rollout), so PHC's observations
    and rewards use it, not the link-origin velocity.
    """
    return (
        entity.get_links_pos(),
        entity.get_links_quat(),
        entity.get_links_vel(ref="link_com"),
        entity.get_links_ang(),
    )


def _rotate_inv(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Express ``v`` (N, B, 3) in the frame of ``q`` (N, 4)."""
    return rotate(inv_quat(q)[:, None], v)


class _AgentTerm(ObservationTerm):
    entity_name: str = "caregiver"

    def build(self) -> None:
        entity = self.resolve_entity(self.entity_name)
        if entity.num_links != NUM_BODIES:
            raise ValueError(
                f"{term_subject(self)}: the entity has {entity.num_links} links, expected the "
                f"{NUM_BODIES} of "
                "PHC's SMPL humanoid (eden.options.robots.smpl_humanoid.PhcSmplHumanoid)."
            )


@OBSERVATION_TERM_REGISTRY.register()
class PhcSelfObservation(_AgentTerm):
    """PHC's proprioception (``compute_humanoid_observations_smpl_max``), 358 dims."""

    def compute(self) -> torch.Tensor:
        return self_obs(*body_state(self.resolve_entity(self.entity_name)))


@OBSERVATION_TERM_REGISTRY.register()
class PhcImitationObservation(_AgentTerm):
    """PHC's imitation observation v6 against this agent's reference one step ahead, 576 dims.

    Parameters
    ----------
    entity_name : str
        The agent.
    command_name : str
        The :class:`~eden.tasks.assistmimic.custom_terms.commands.AssistMimicPairMotion` term.
    role : str
        ``"caregiver"`` or ``"recipient"`` — which reference track of the pair this agent follows.
    partner_entity_name : str
        Set it to turn on AssistMimic's **Dynamic Reference Retargeting**
        (``_adjust_caregiver_hand_reference``, ``enable_adjust_caregiver_hand_reference``): while the pair is
        within ``retarget_distance``, each of this agent's hands is asked for where its partner's nearest body
        *actually is* rather than where the clip says it should be, by shifting that hand's reference by the
        partner's own tracking error there. A caregiver can then keep hold of a recipient who has slipped off
        the reference, which is exactly what happens once the recipient is being lifted. Only the caregiver
        gets it (AssistMimic masks on ``env_id % 2 == 0``), and only on HHI-Assist: Inter-X help-up leaves
        ``enable_adjust_caregiver_hand_reference`` at ``False``. Empty disables it.

        This moves the *observation* reference only. The imitation **reward** keeps grading against the
        unshifted clip (AssistMimic reads the motion library again in ``_compute_reward_with_hand_masking``),
        so the retargeting can never pay the agent for the partner's tracking error.
    retarget_distance : float
        Pair root distance [m] within which the retargeting applies (``force_tracking_distance_threshold``).
    """

    command_name: str = "motion"
    role: str = "caregiver"
    partner_entity_name: str = ""
    retarget_distance: float = 1.3

    def build(self) -> None:
        super().build()
        if self.role not in PARTNER:
            raise ValueError(
                f"{term_subject(self)}: role '{self.role}' is not one of {ROLES}; it picks the reference track "
                "(AssistMimicPairMotion.obs_ref) this agent observes."
            )
        self._env.command_manager.get_term(self.command_name)
        if not self.partner_entity_name:
            return
        entity = self.resolve_entity(self.entity_name)
        self.resolve_entity(self.partner_entity_name)
        self._partner_role = PARTNER[self.role]
        # One (wrist, hand) index pair per side: SMPL's whole hand below the elbow, AssistMimic's 16 finger
        # links plus the wrist reduced to the two links this body has.
        self._hand_idx = torch.stack(
            [resolve_link_idx(entity, list(pair), self.device) for pair in WRIST_LINKS]
        )  # (2, 2), column 0 is the wrist the nearest partner body is measured from
        self._rows = torch.arange(self.num_envs, device=self.device)

    def _retargeted(self, command, ref_pos: torch.Tensor) -> torch.Tensor:
        """``ref_pos`` with each hand's rows shifted by the partner's tracking error at its nearest body."""
        partner = self.resolve_entity(self.partner_entity_name)
        partner_ref = command.obs_ref[self._partner_role].pos  # (N, 24, 3)
        partner_sim = partner.get_links_pos()
        own_root = self.resolve_entity(self.entity_name).get_links_pos()[:, 0]
        near = (own_root - partner_sim[:, 0]).norm(dim=-1) <= self.retarget_distance  # (N,)
        out = ref_pos.clone()
        for side in self._hand_idx:
            wrist_ref = ref_pos[:, side[0]]  # (N, 3)
            nearest = torch.cdist(wrist_ref[:, None], partner_ref)[:, 0].argmin(dim=-1)  # (N,)
            error = partner_sim[self._rows, nearest] - partner_ref[self._rows, nearest]  # (N, 3)
            out[:, side] += torch.where(near[:, None], error, torch.zeros_like(error))[:, None]
        return out

    def compute(self) -> torch.Tensor:
        command = self._env.command_manager.get_term(self.command_name)
        ref = command.obs_ref[self.role]
        ref_pos = self._retargeted(command, ref.pos) if self.partner_entity_name else ref.pos
        return imitation_obs_v6(
            *body_state(self.resolve_entity(self.entity_name)), ref_pos, ref.rot, ref.vel, ref.ang_vel
        )


class _WristTerm(_AgentTerm):
    def build(self) -> None:
        super().build()
        self._wrist_idx, self._hand_idx = wrist_hand_idx(self.resolve_entity(self.entity_name), self.device)

    def wrist_forces(self, entity) -> torch.Tensor:
        """Net contact force on each wrist (wrist + hand link), world frame, ``(N, 2, 3)``."""
        forces = entity.get_links_net_contact_force()
        return forces[:, self._wrist_idx] + forces[:, self._hand_idx]


@OBSERVATION_TERM_REGISTRY.register()
class WristContact(_WristTerm):
    """AssistMimic's hand-contact flag, per wrist: force above a threshold *and* near one of the partner's wrists.

    Parameters
    ----------
    entity_name : str
        The agent whose wrists are checked.
    partner_entity_name : str
        The other humanoid.
    force_threshold : float
        Contact-force magnitude [N] (AssistMimic ``hand_contact_threshold``).
    proximity : float
        Max distance [m] to the partner's nearest wrist (``hand_contact_proximity_threshold``).
    """

    partner_entity_name: str = "recipient"
    force_threshold: float = 10.0
    proximity: float = 0.2

    def build(self) -> None:
        super().build()
        self.resolve_entity(self.partner_entity_name)

    def compute(self) -> torch.Tensor:
        entity = self.resolve_entity(self.entity_name)
        partner = self.resolve_entity(self.partner_entity_name)
        force_contact = self.wrist_forces(entity).norm(dim=-1) > self.force_threshold
        own = entity.get_links_pos(ls_idx_local=self._wrist_idx)
        other = partner.get_links_pos(ls_idx_local=self._wrist_idx)
        near = torch.cdist(own, other).min(dim=-1).values < self.proximity
        return (force_contact & near).float()


@OBSERVATION_TERM_REGISTRY.register()
class WristForce(_WristTerm):
    """Contact force on each wrist in the agent's root frame (AssistMimic ``_compute_hand_force_obs``), 6 dims."""

    def compute(self) -> torch.Tensor:
        entity = self.resolve_entity(self.entity_name)
        return _rotate_inv(entity.get_links_quat()[:, 0], self.wrist_forces(entity)).flatten(1)


@OBSERVATION_TERM_REGISTRY.register()
class PartnerObservation(_AgentTerm):
    """AssistMimic ``partner_obs_v2``: the partner's bodies in the agent's root frame, 294 dims.

    Partner body positions relative to the agent's root, their linear velocities and rotations (tan-norm), all
    rotated into the agent's full root orientation (not just its heading), plus the partner root's
    orientation with its heading removed.
    """

    partner_entity_name: str = "recipient"

    def build(self) -> None:
        super().build()
        self.resolve_entity(self.partner_entity_name)

    def compute(self) -> torch.Tensor:
        entity = self.resolve_entity(self.entity_name)
        partner = self.resolve_entity(self.partner_entity_name)
        root_pos, root_rot = entity.get_links_pos()[:, 0], entity.get_links_quat()[:, 0]
        p_pos, p_rot, p_vel, _ = body_state(partner)
        n, b = p_pos.shape[:2]
        rel_pos = _rotate_inv(root_rot, p_pos - root_pos[:, None])
        rel_vel = _rotate_inv(root_rot, p_vel)
        rel_rot = quat_mul(inv_quat(root_rot)[:, None].expand(n, b, 4).reshape(-1, 4), p_rot.reshape(-1, 4))
        partner_root_upright = quat_mul(heading_quat_inv(p_rot[:, 0]), p_rot[:, 0])
        return torch.cat(
            [
                rel_pos.reshape(n, -1),
                rel_vel.reshape(n, -1),
                quat_to_tan_norm(rel_rot).reshape(n, -1),
                quat_to_tan_norm(partner_root_upright),
            ],
            dim=-1,
        )


@OBSERVATION_TERM_REGISTRY.register()
class PartnerRelativeToWrists(_WristTerm):
    """AssistMimic ``_compute_wrist_relative_partner_obs``: partner bodies in each own wrist frame, 144 dims."""

    partner_entity_name: str = "recipient"

    def build(self) -> None:
        super().build()
        self.resolve_entity(self.partner_entity_name)

    def compute(self) -> torch.Tensor:
        entity = self.resolve_entity(self.entity_name)
        p_pos = self.resolve_entity(self.partner_entity_name).get_links_pos()
        w_pos = entity.get_links_pos(ls_idx_local=self._wrist_idx)
        w_rot = entity.get_links_quat(ls_idx_local=self._wrist_idx)
        out = [_rotate_inv(w_rot[:, i], p_pos - w_pos[:, i : i + 1]).flatten(1) for i in range(2)]
        return torch.cat(out, dim=-1)


@OBSERVATION_TERM_REGISTRY.register()
class RoleFlag(ObservationTerm):
    """Constant role label (AssistMimic ``env_id % 2``: 0 caregiver, 1 recipient).

    Parameters
    ----------
    value : float
        The label.
    """

    value: float = 0.0

    def build(self) -> None:
        self._flag = torch.full((self.num_envs, 1), self.value, device=self.device)

    def compute(self) -> torch.Tensor:
        return self._flag
