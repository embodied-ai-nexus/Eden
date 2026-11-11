"""FlashDexRetarget's failure terminations: object drop, fingertip divergence, blow-ups."""

from __future__ import annotations

import torch

from eden.managers.termination_manager import TERMINATION_TERM_REGISTRY, TerminationTerm
from eden.tasks.flashdexretarget.custom_terms.common import HandTermMixin


@TERMINATION_TERM_REGISTRY.register()
class FDRObjectDropped(HandTermMixin, TerminationTerm):
    """An object is more than ``drop_margin`` below its reference height.

    Parameters
    ----------
    drop_margin : float
        Metres below the reference (FlashDexRetarget: 0.10).
    """

    drop_margin: float = 0.10

    def build(self) -> None:
        self._build_state()

    def compute(self) -> torch.Tensor:
        z = self.state.object_pose()[0][..., 2]
        return (z < self.state.ref_object(0)["pos"][..., 2] - self.drop_margin).any(dim=-1)


@TERMINATION_TERM_REGISTRY.register()
class FDRFingertipDiverged(HandTermMixin, TerminationTerm):
    """A robot fingertip is more than ``threshold`` from its MANO fingertip, after ``grace_steps`` of the episode.

    Parameters
    ----------
    threshold : float
        Metres (FlashDexRetarget: 0.2).
    grace_steps : int
        Control steps after a reset the check waits (FlashDexRetarget: 15).
    """

    threshold: float = 0.2
    grace_steps: int = 15

    def build(self) -> None:
        self._build_state()

    def compute(self) -> torch.Tensor:
        err = (self.state.mano_tips(0) - self.state.tips_pos()).norm(dim=-1)
        return (err > self.threshold).flatten(1).any(dim=-1) & (self._env.episode_length_buf >= self.grace_steps)


@TERMINATION_TERM_REGISTRY.register()
class FDRVelocitySanity(HandTermMixin, TerminationTerm):
    """A physics blow-up: a joint-velocity norm above ``max_joint_vel``, or an object faster than ``max_obj_lin_vel``
    / ``max_obj_ang_vel``.

    Parameters
    ----------
    max_joint_vel : float
        rad/s (FlashDexRetarget: 200).
    max_obj_lin_vel : float
        m/s (FlashDexRetarget: 100).
    max_obj_ang_vel : float
        rad/s (FlashDexRetarget: 200).
    """

    max_joint_vel: float = 200.0
    max_obj_lin_vel: float = 100.0
    max_obj_ang_vel: float = 200.0

    def build(self) -> None:
        self._build_state()

    def compute(self) -> torch.Tensor:
        robot = self._env.entities[self.entity_name]
        lin, ang = self.state.object_vel()
        return (
            (robot.get_dofs_vel().norm(dim=-1) > self.max_joint_vel)
            | (lin.norm(dim=-1) > self.max_obj_lin_vel).any(dim=-1)
            | (ang.norm(dim=-1) > self.max_obj_ang_vel).any(dim=-1)
        )
