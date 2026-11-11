"""G1Inspire = Unitree G1 29DoF body + an Inspire RH56DFQ hand welded to each wrist (Weave's G1-Inspire-HOI rig).

The composite form of Weave's merged ``g1_29dof_rev_1_0_with_inspire_hand_DFQ.urdf``, cut at its two fixed hand
joints so each hand is its own scene entity with its own DOFs while the assembled rig is the one the Weave reference
motions were retargeted onto.

**The body** is :class:`~eden.options.robots.unitree_g1.G1Dofs29NoHands` on the bare-wrist URDF: the same G1 rev 1.0
joints, frames and masses, except that each ``*_wrist_yaw_link`` carries the bare wrist (0.0846 kg) rather than the
no-hand MJCF's wrist-plus-rubber-hand (0.2546 kg), so the arms carry the welded hands' real mass and no phantom
payload on top.

**The mount** is the fixed hand joint the split cut (``*_base_link_joint``): the hand base link sits ``0.0415 m``
along the wrist, rotated rpy ``(0, 0, π/2)`` left and ``(π, 0, -π/2)`` right. Those two rotations are not mirror
images, so the rig sets them per side. :class:`~eden.datasets.weave_dataset.WeaveG1InspireHOIDataset` rebuilds the hand
base pose of the reference from the same constants.

The hands are not gravity-compensated: the G1 is a floating-base humanoid whose arms carry the 0.19 kg per hand.
"""

from __future__ import annotations

import math

from genesis.typing import UnitVec4FType, Vec3FType
from pydantic import Field

from eden.options.entities import RobotOptions
from eden.options.robots.composite.base import COMPOSITE_ROBOT_REGISTRY, HumanoidOptions
from eden.options.robots.inspire_rh56 import InspireRH56DFQ_L, InspireRH56DFQ_R
from eden.options.robots.unitree_g1 import G1Dofs29NoHands

#: The G1 rev 1.0 body of Weave's rig, its hands cut off at the fixed hand joints (bare-wrist inertials).
G1_BARE_WRIST_FILE = "humanoids/unitree_g1/g1_29dof_rev_1_0_bare_wrist.urdf"
#: The wrist links the hands weld to.
G1_INSPIRE_WRIST_LINKS_NAME: tuple[str, str] = ("left_wrist_yaw_link", "right_wrist_yaw_link")
#: Hand base link position in its ``*_wrist_yaw_link`` frame (Weave's ``*_base_link_joint`` origin), m.
G1_INSPIRE_MOUNT_OFFSET: tuple[float, float, float] = (0.0415, 0.0, 0.0)
_H = math.sqrt(0.5)
#: Hand base link orientation in the wrist frame, wxyz: rpy ``(0, 0, π/2)`` left, ``(π, 0, -π/2)`` right.
G1_INSPIRE_LEFT_MOUNT_QUAT: tuple[float, float, float, float] = (_H, 0.0, 0.0, _H)
G1_INSPIRE_RIGHT_MOUNT_QUAT: tuple[float, float, float, float] = (0.0, _H, -_H, 0.0)


@COMPOSITE_ROBOT_REGISTRY.register()
class G1Inspire(HumanoidOptions):
    """Unitree G1 + two Inspire RH56DFQ hands, a 53-DOF bimanual humanoid (29 body + 2 x 12, 6 of each hand coupled).

    Expands into three scene entities — the body under
    :attr:`~eden.options.robots.composite.base.HumanoidOptions.robot_name` plus a hand per wrist::

        scene = en.scenes.Empty(robot=G1Inspire())
        # -> entities "robot", "left_hand", "right_hand"

    A hand's six mimic followers are driven in software, so a position controller commands only
    :attr:`~eden.options.robots.composite.base.HumanoidOptions.left_hand_actuated_dofs` and passes the couplings
    through::

        ImplicitPDController.configure(entity_name=rig.left_hand_name, dofs_name=rig.left_hand_actuated_dofs,
                                       mimic_dofs=rig.left_hand_mimic_dofs)
    """

    robot: RobotOptions = Field(default_factory=lambda: G1Dofs29NoHands(file=G1_BARE_WRIST_FILE))
    #: Free-base because a welded child is posed by its parent link.
    left_hand: RobotOptions = Field(default_factory=lambda: InspireRH56DFQ_L(is_fixed_base=False))
    right_hand: RobotOptions = Field(default_factory=lambda: InspireRH56DFQ_R(is_fixed_base=False))

    left_ee_link: str = G1_INSPIRE_WRIST_LINKS_NAME[0]
    right_ee_link: str = G1_INSPIRE_WRIST_LINKS_NAME[1]
    mount_offset: Vec3FType = G1_INSPIRE_MOUNT_OFFSET
    left_mount_quat: UnitVec4FType | None = G1_INSPIRE_LEFT_MOUNT_QUAT
    right_mount_quat: UnitVec4FType | None = G1_INSPIRE_RIGHT_MOUNT_QUAT

    #: Distal link of each digit, thumb-to-pinky, read off each hand's metadata.
    left_grasp_links: list[str] = list(InspireRH56DFQ_L.metadata.target_hand_links)
    right_grasp_links: list[str] = list(InspireRH56DFQ_R.metadata.target_hand_links)
