"""AssistMimic's pair termination: the episode ends for both humanoids when either loses its reference."""

from __future__ import annotations

import torch

from eden.managers.termination_manager import TERMINATION_TERM_REGISTRY, TerminationTerm
from eden.managers.terms.utils import resolve_link_idx
from eden.tasks.assistmimic.custom_terms.commands import ROLES

#: ``reset_bodies`` of ``env_im_interx_helpup.yaml``, all of them bodies of the SMPL humanoid.
RESET_BODIES = [
    "Pelvis", "L_Hip", "R_Hip", "Torso", "Spine", "Chest", "Neck", "Head", "L_Thorax", "L_Shoulder",
    "R_Thorax", "R_Shoulder", "L_Knee", "R_Knee", "L_Ankle", "R_Ankle",
]  # fmt: skip


@TERMINATION_TERM_REGISTRY.register()
class PairTrackingFailure(TerminationTerm):
    """Either agent's mean body distance to its reference exceeds its threshold (PHC ``use_mean`` reset).

    Checked from the second step of an episode on, a restored one's included (PHC skips ``progress_buf <= 1``).
    AssistMimic resets both envs of a pair when either fails; here both humanoids share one env, so that is the
    natural outcome.

    Parameters
    ----------
    command_name : str
        The :class:`~eden.tasks.assistmimic.custom_terms.commands.AssistMimicPairMotion` term.
    caregiver_entity_name, recipient_entity_name : str
        The two humanoids.
    caregiver_threshold, recipient_threshold : float
        ``terminationDistance`` / ``terminationDistanceRecipient`` [m].
    bodies_name : list[str]
        ``reset_bodies``.
    """

    command_name: str = "motion"
    caregiver_entity_name: str = "caregiver"
    recipient_entity_name: str = "recipient"
    caregiver_threshold: float = 0.25
    recipient_threshold: float = 0.25
    bodies_name: list[str] = RESET_BODIES

    def build(self) -> None:
        self._env.command_manager.get_term(self.command_name)
        self._names = {"caregiver": self.caregiver_entity_name, "recipient": self.recipient_entity_name}
        self._thresholds = {"caregiver": self.caregiver_threshold, "recipient": self.recipient_threshold}
        self._idx = resolve_link_idx(self.resolve_entity(self.caregiver_entity_name), self.bodies_name, self.device)
        self.resolve_entity(self.recipient_entity_name)

    def compute(self) -> torch.Tensor:
        command = self._env.command_manager.get_term(self.command_name)
        failed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        for role in ROLES:
            pos = self.resolve_entity(self._names[role]).get_links_pos(ls_idx_local=self._idx)
            dist = (pos - command.grade_ref[role].pos[:, self._idx]).norm(dim=-1).mean(dim=-1)
            failed |= dist > self._thresholds[role]
        return failed & (self._env.episode_length_buf > 1)


@TERMINATION_TERM_REGISTRY.register()
class ReferenceMotionEnd(TerminationTerm):
    """The clip has been played to its last frame (PHC ``pass_time``) — a truncation.

    Parameters
    ----------
    command_name : str
        The :class:`~eden.tasks.assistmimic.custom_terms.commands.AssistMimicPairMotion` term.
    """

    command_name: str = "motion"
    time_out: bool = True

    def build(self) -> None:
        self._env.command_manager.get_term(self.command_name)

    def compute(self) -> torch.Tensor:
        command = self._env.command_manager.get_term(self.command_name)
        return command.motion_time() >= command.clip_length
