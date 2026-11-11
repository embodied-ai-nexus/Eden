"""OliWuji = LimX Oli (hands stripped) + a Wuji Hand2 beta2 bolted to each wrist flange.

A :class:`~eden.options.robots.composite.base.HumanoidOptions` rig: a humanoid root with two dexterous
hands, rather than the one-arm-one-end-effector shape of
:class:`~eden.options.robots.composite.base._ArmCompositeOptions`. Both welds are identities, so
this module is entirely defaults — the entity names, the pelvis knob, the mount transform, the DOF
shortcuts and the two welds all come from the base.

**The flange.** Carried by the asset, not by this file:
:class:`~eden.options.robots.limx_oli.OliNoHands` declares a pure ``*_hand_mount`` frame on each
exposed wrist flange, so the hands seat on it with no offset and nothing here re-derives one.

**The orientation.** Also the identity, because both assets already share the wrist convention:
Oli's arm and its moulded hand run along -Z, which is the Wuji finger axis (``up=(0, 0, -1)``);
and the moulded hand is wide across X (8.6 cm), thin across Y (5.5 cm), with its distal geometry
drifting +Y right / -Y left — fingers curling toward the palm — which is the palm normal the
Wuji hand declares per side. Palm inboard, thumb forward.

**The catch.** A Wuji hand's ``up``/``front`` rotate its asset 180 deg at load, and a weld has to
cancel that or the hand ends up pointing back up the forearm — see
:func:`~eden.options.robots.composite.base.weld_quat`, which
:meth:`~eden.options.robots.composite.base.HumanoidOptions._mount_spec` applies to every humanoid weld.

The hands are deliberately *not* gravity-compensated (unlike the industrial-arm rigs here): Oli
is a floating-base humanoid whose arms must carry the payload they are given.
"""

from __future__ import annotations

from pydantic import Field

from eden.options.entities import RobotOptions
from eden.options.robots.composite.base import COMPOSITE_ROBOT_REGISTRY, HumanoidOptions
from eden.options.robots.limx_oli import OliNoHands
from eden.options.robots.wuji_hand2beta2 import WujiHand2Beta2WithMount_L, WujiHand2Beta2WithMount_R


@COMPOSITE_ROBOT_REGISTRY.register()
class OliWuji(HumanoidOptions):
    """LimX Oli + two Wuji Hand2 beta2 hands, a 71-DOF bimanual humanoid (31 body + 2 x 20).

    Added to a scene by name like any other entity; it expands into three — the humanoid
    under :attr:`~eden.options.robots.composite.base.HumanoidOptions.robot_name` plus a hand per
    wrist, each welded to the asset's own ``*_hand_mount`` flange frame::

        scene = en.scenes.Empty(robot=OliWuji(base_pos=(0.0, 0.0, 0.903)))
        # -> entities "robot", "left_hand", "right_hand"

    The rig exposes what a controller needs as shortcut properties over the sub-options, so
    the DOF lists and fingertip links cannot drift from the robot definitions::

        JointPositionAction.configure(entity_name=rig.robot_name, dofs_name=rig.robot_dofs)
        DifferentialIKController.configure(entity_name=rig.robot_name, ee_link_name=rig.right_ee_link)
    """

    #: The hand-stripped variant, which carries the ``*_hand_mount`` flanges and a wrist inertial
    #: recomputed for bare metal, so the arms carry the hands' real 0.69 kg each and no phantom payload.
    robot: RobotOptions = Field(default_factory=OliNoHands)
    #: The ``WithMount`` flavours, whose root link *is* the flange plate. Free-base because a
    #: welded child is posed by its parent link; no friction override, so a task that wants
    #: grippier fingertips says so itself.
    left_hand: RobotOptions = Field(default_factory=lambda: WujiHand2Beta2WithMount_L(is_fixed_base=False))
    right_hand: RobotOptions = Field(default_factory=lambda: WujiHand2Beta2WithMount_R(is_fixed_base=False))

    #: The asset's own ``*_hand_mount`` links, not the parent ``*_wrist_roll_link`` — which is what
    #: makes the inherited ``mount_offset`` an identity rather than a re-derived number.
    left_ee_link: str = "left_hand_mount"
    right_ee_link: str = "right_hand_mount"

    #: Fingertip links per hand, thumb-to-pinky; their centroid is the grasp point. Read off
    #: each hand's metadata, so swapping in a different hand means setting these to match.
    left_grasp_links: list[str] = list(WujiHand2Beta2WithMount_L.metadata.target_hand_links)
    right_grasp_links: list[str] = list(WujiHand2Beta2WithMount_R.metadata.target_hand_links)
