"""FlashDexRetarget's clip sampling."""

from __future__ import annotations

from typing import TYPE_CHECKING

from eden.managers.motion_manager import MOTION_TERM_REGISTRY, MotionTerm

if TYPE_CHECKING:
    from eden.envs.base import EnvBase


@MOTION_TERM_REGISTRY.register()
class FDRMotionTerm(MotionTerm):
    """The clip sampling: a training episode starts at a uniform frame of its clip (``motion_start_from_middle_range``,
    FlashDexRetarget's [0, 0.99 T)), an evaluation episode (an env built with ``eval_mode=True``, ``eden inference``)
    at the clip's first frame, so that a completed evaluation episode tracked its whole clip.
    """

    def __init__(self, env: EnvBase, options) -> None:
        super().__init__(env, options)
        if not env.training:
            self.start_at_timestep_zero_prob = 1.0
