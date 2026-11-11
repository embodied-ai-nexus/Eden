"""FlashDexRetarget's task terms: ports of FlashDexRetarget's (``src/envs``) onto Eden's managers, one module per
manager (``observations``, ``actions``, ``rewards``, ``terminations``, ``motions``, ``metrics``, ``events``,
``curricula``). ``common.py`` is the one to read first: the hand / object state every term reads.

The robot is one fixed-base articulation holding one or two hands (6 actuated forearm joints + fingers each); every
hand has an object (one shared, or one each) and a MANO reference. Per hand side, the terms read:

* the robot's **palm** frame (the MJCF ``<side>_palm`` site: the palm link's origin, rotated by the site's
  orientation), **fingertips** (the MJCF ``track_hand_<side>_<finger>_tip`` sites, as link + offset), mapped
  **finger bodies** and **contact links** — :class:`~eden.tasks.flashdexretarget.custom_terms.common.HandObjectState`;
* the **object** the hand holds, in its mesh-origin frame (``relative=True``), and its SDF grid and surface points per
  object variant;
* the **references** from the motion manager: ``mano_<side>`` (wrist base + 20 MANO joints) and the object entity's
  own sequence, at frame 0 (the current step) and frame 1 (the next); the future-trajectory observation reads up to
  frame ``n_future``.

Class terms all take ``hand_model`` (``xhand`` / ``sharpa``), ``sides``, ``objects`` (side -> object entity) and
``dataset`` (the ``motion.pt``, for the object geometry):
:class:`~eden.tasks.flashdexretarget.config.FlashDexRetargetDataMod` fills them.
"""

from eden.tasks.flashdexretarget.custom_terms.actions import FDRHandAction, FDRObjectAssist
from eden.tasks.flashdexretarget.custom_terms.curricula import FDRAssistCurriculum
from eden.tasks.flashdexretarget.custom_terms.events import FDRHandReset
from eden.tasks.flashdexretarget.custom_terms.metrics import (
    FDREpisodeCompleted,
    FDRObjectPosError,
    FDRObjectRotError,
    FDRTipPosError,
)
from eden.tasks.flashdexretarget.custom_terms.motions import FDRMotionTerm
from eden.tasks.flashdexretarget.custom_terms.observations import (
    FDRFutureTrajectory,
    FDRHandObservation,
    FDRTargetError,
)
from eden.tasks.flashdexretarget.custom_terms.rewards import (
    FDRActionPenalty,
    FDRJointBCReward,
    FDRObjectPointCloudReward,
    FDRTipDistanceBoundReward,
    FDRTipTransReward,
    FDRWristRotReward,
    FDRWristTransReward,
)
from eden.tasks.flashdexretarget.custom_terms.terminations import (
    FDRFingertipDiverged,
    FDRObjectDropped,
    FDRVelocitySanity,
)

__all__ = [
    "FDRActionPenalty",
    "FDRAssistCurriculum",
    "FDREpisodeCompleted",
    "FDRFingertipDiverged",
    "FDRFutureTrajectory",
    "FDRHandAction",
    "FDRHandObservation",
    "FDRHandReset",
    "FDRJointBCReward",
    "FDRMotionTerm",
    "FDRObjectAssist",
    "FDRObjectDropped",
    "FDRObjectPointCloudReward",
    "FDRObjectPosError",
    "FDRObjectRotError",
    "FDRTargetError",
    "FDRTipDistanceBoundReward",
    "FDRTipPosError",
    "FDRTipTransReward",
    "FDRVelocitySanity",
    "FDRWristRotReward",
    "FDRWristTransReward",
]
