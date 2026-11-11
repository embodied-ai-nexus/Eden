"""Built-in curriculum terms."""

from eden.managers.terms.curricula.locomotion import AverageEpisodeLengthTracker
from eden.managers.terms.curricula.motion import MotionBasePoseThresholdCurriculum
from eden.managers.terms.curricula.rewards import PenaltyCurriculum, StageRewardWeightCurriculum
from eden.managers.terms.curricula.success_guided import SuccessGuidedSampling
from eden.managers.terms.curricula.velocity import VelocityCommandCurriculum

__all__ = [
    "AverageEpisodeLengthTracker",
    "MotionBasePoseThresholdCurriculum",
    "PenaltyCurriculum",
    "StageRewardWeightCurriculum",
    "SuccessGuidedSampling",
    "VelocityCommandCurriculum",
]
