"""MDP terms shared by the robosuite task ports, split by manager.

``common`` holds the grasp-frame, object-point and pad-contact helpers; ``observations`` /
``rewards`` / ``metrics`` / ``terminations`` hold the registered terms. Every term that touches the robot
declares one field — ``rig``, the scene slot its arm + gripper composite sits in
(:class:`~eden.tasks.robosuite.custom_terms.common.RigMixin`) — and reads the arm, the gripper, the mount link, the
pad links and both DOF lists off that composite in ``build()``. That is what makes ``--robot`` a scene edit: no term
holds a copy of the rig, so a swap has nothing to rewrite and nothing it can leave stale.
"""

from eden.tasks.robosuite.custom_terms.metrics import (
    EntitiesInContact,
    PadsGrasping,
)
from eden.tasks.robosuite.custom_terms.observations import (
    ArmDofsPosSinCos,
    ArmDofsVel,
    EefPos,
    EefQuat,
    GripperDofsPos,
    GripperDofsVel,
    GripperToObject,
    ObjectLinkPos,
    ObjectPointPos,
    ObjectToObject,
)
from eden.tasks.robosuite.custom_terms.rewards import HoverOverPoint, ReachTanh
from eden.tasks.robosuite.custom_terms.terminations import entity_below_height

__all__ = [
    "ArmDofsPosSinCos",
    "ArmDofsVel",
    "EefPos",
    "EefQuat",
    "EntitiesInContact",
    "GripperDofsPos",
    "GripperDofsVel",
    "GripperToObject",
    "HoverOverPoint",
    "ObjectLinkPos",
    "ObjectPointPos",
    "ObjectToObject",
    "PadsGrasping",
    "ReachTanh",
    "entity_below_height",
]
