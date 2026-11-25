"""Built-in motion terms: the base :class:`~eden.managers.motion_manager.MotionTerm` and the terms written on it."""

from eden.managers.motion_manager import MOTION_TERM_REGISTRY, MotionTerm
from eden.managers.terms.motions.kinematic import KinematicLinksMotion

# Registered here, not where it is defined: see the note at ``MOTION_TERM_REGISTRY``.
MOTION_TERM_REGISTRY.register()(MotionTerm)

__all__ = ["KinematicLinksMotion", "MotionTerm"]
