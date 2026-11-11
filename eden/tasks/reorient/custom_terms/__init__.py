"""Task-local manager terms for in-hand cube reorientation.

Split by term kind. The two modules worth reading first are ``commands.py`` (the goal
state machine that defines the task) and ``cage.py`` / ``contacts.py`` (the shared
computations every other term pulls from).
"""

from eden.tasks.reorient.custom_terms.actions import JointPositionOffsetEma
from eden.tasks.reorient.custom_terms.cage import CageEscape
from eden.tasks.reorient.custom_terms.commands import InHandReorientCommand
from eden.tasks.reorient.custom_terms.contacts import HandObjectContact, SelfCollisionCount
from eden.tasks.reorient.custom_terms.curricula import AdaptiveEpisodeCurriculum, ReorientSuccessCurriculum
from eden.tasks.reorient.custom_terms.events import (
    ObjectVelocityDisturbance,
    RandomizeDofsArmature,
    RandomizeDofsDamping,
    RandomizeGravityTilt,
    ResetDofsByPatternOffset,
    ResetObjectInPalm,
)
from eden.tasks.reorient.custom_terms.metrics import (
    CubeHeightAbovePalm,
    DofsAccRms,
    DofsVelRms,
    TorqueSaturationRatio,
    cube_survival_steps,
    goal_interval,
    goal_reach_count,
)
from eden.tasks.reorient.custom_terms.observations import (
    CubePosInTag,
    DofsPosNormalized,
    DofsPosTargetError,
    FingertipPositions,
    GoalRotErrorInTag,
    PalmRot6d,
    PerturbationDirection,
    PerturbationVelocity,
    RandomizedParams,
    cage_counter_progress,
    command_state_progress,
    object_ang_vel,
    object_lin_vel,
)
from eden.tasks.reorient.custom_terms.rewards import (
    FingertipSlide,
    NormalizedTorque,
    dofs_pos_deviation_l2,
    hold_escalation,
    orientation_alignment,
    palm_detach,
)
from eden.tasks.reorient.custom_terms.terminations import cage_drop

__all__ = [
    "AdaptiveEpisodeCurriculum",
    "CageEscape",
    "CubeHeightAbovePalm",
    "CubePosInTag",
    "DofsAccRms",
    "DofsPosNormalized",
    "DofsPosTargetError",
    "DofsVelRms",
    "FingertipPositions",
    "FingertipSlide",
    "GoalRotErrorInTag",
    "HandObjectContact",
    "InHandReorientCommand",
    "JointPositionOffsetEma",
    "NormalizedTorque",
    "ObjectVelocityDisturbance",
    "PalmRot6d",
    "PerturbationDirection",
    "PerturbationVelocity",
    "RandomizeDofsArmature",
    "RandomizeDofsDamping",
    "RandomizeGravityTilt",
    "RandomizedParams",
    "ReorientSuccessCurriculum",
    "ResetDofsByPatternOffset",
    "ResetObjectInPalm",
    "SelfCollisionCount",
    "TorqueSaturationRatio",
    "cage_counter_progress",
    "cage_drop",
    "command_state_progress",
    "cube_survival_steps",
    "dofs_pos_deviation_l2",
    "goal_interval",
    "goal_reach_count",
    "hold_escalation",
    "object_ang_vel",
    "object_lin_vel",
    "orientation_alignment",
    "palm_detach",
]
