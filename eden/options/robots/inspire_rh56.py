"""Inspire RH56 dexterous hands (left/right): the stock RH56 and the RH56DFQ variant Unitree mounts on the G1.

Both are the same hand: link tree, joint frames and axes, and every link's inertial (0.1918 kg in all) are identical.
What differs is the asset each is published with, and so the joint ranges, the coupling ratios and the meshes:

==========================  ==================================  =========================================
asset                       ``InspireRH56_*``                   ``InspireRH56DFQ_*`` (Weave's G1 rig)
==========================  ==================================  =========================================
finger proximal range       0 – 1.47 rad                        0 – 1.7 rad
thumb distal range          0 – 0.4 rad                         0 – 1.2 rad
couplings                   URDF ``<mimic>`` (Genesis equality) software, via ``mimic_dofs`` (no tags)
thumb int. / distal ratio   1.334 / 0.667                       1.6 / 2.4
finger intermediate ratio   1.06399, offset -0.04545            1.0
==========================  ==================================  =========================================

Six of the twelve joints are mimic followers of the six drivers (``actuated_dofs_name``). ``metadata.mimic_dofs``
publishes the couplings as ``follower -> (driver, multiplier, offset)`` for both. The RH56 asset keeps its
``<mimic>`` tags, so Genesis couples its followers itself and the mapping is descriptive; the DFQ asset has them
stripped, and a task passes the mapping to the joint PD action term, which then commands each follower at
``driver_target * multiplier + offset`` (Weave's ``MimicJointPositionAction``)::

    ImplicitPDController.configure(entity_name="left_hand", dofs_name=InspireRH56DFQ_L.actuated_dofs_name,
                                   mimic_dofs=InspireRH56DFQ_L.metadata.mimic_dofs)
"""

from __future__ import annotations
from typing import ClassVar

from eden.options.catalog import ROBOT_REGISTRY
from eden.options.entities import RobotOptions, MetadataOptions

INSPIRE_DOFS_NAME: list[str] = [
    "thumb_proximal_yaw_joint",
    "thumb_proximal_pitch_joint",
    "thumb_intermediate_joint",
    "thumb_distal_joint",
    "index_proximal_joint",
    "index_intermediate_joint",
    "middle_proximal_joint",
    "middle_intermediate_joint",
    "ring_proximal_joint",
    "ring_intermediate_joint",
    "pinky_proximal_joint",
    "pinky_intermediate_joint",
]
#: The six driven joints; every other joint follows one of them.
INSPIRE_ACTUATED_DOFS_NAME: list[str] = [
    "thumb_proximal_yaw_joint",
    "thumb_proximal_pitch_joint",
    "index_proximal_joint",
    "middle_proximal_joint",
    "ring_proximal_joint",
    "pinky_proximal_joint",
]
#: Distal link of each digit, thumb-to-pinky: the links whose ``*_tip`` frames Genesis merges into them.
INSPIRE_FINGERTIP_LINKS_NAME: list[str] = [
    "thumb_distal",
    "index_intermediate",
    "middle_intermediate",
    "ring_intermediate",
    "pinky_intermediate",
]
_FINGERS = ("index", "middle", "ring", "pinky")


def _mimic_dofs(
    thumb_intermediate: float, thumb_distal: float, finger: float, finger_offset: float
) -> dict[str, tuple[str, float, float]]:
    """``follower -> (driver, multiplier, offset)`` for one asset's coupling ratios."""
    return {
        "thumb_intermediate_joint": ("thumb_proximal_pitch_joint", thumb_intermediate, 0.0),
        "thumb_distal_joint": ("thumb_proximal_pitch_joint", thumb_distal, 0.0),
        **{f"{f}_intermediate_joint": (f"{f}_proximal_joint", finger, finger_offset) for f in _FINGERS},
    }


def _metadata(side: str, mimic_dofs: dict[str, tuple[str, float, float]]) -> MetadataOptions:
    return MetadataOptions(
        side=side,
        target_hand_links=list(INSPIRE_FINGERTIP_LINKS_NAME),
        mimic_dofs=mimic_dofs,
    )


# WARNING: THIS IS A PLACEHOLDER FOR THE INSPIRE ROBOT.
# TODO: the inspire setup is unstable, do not use it for now
# We need the physics implementation for non-backdrivable / high joint friction robots.
class _InspireRH56(RobotOptions):
    dofs_name: list[str] = list(INSPIRE_DOFS_NAME)
    default_dofs_pos: dict[str, float] = {"*": 0.0}
    default_dofs_kp: dict[str, float] = {
        "*": 500.0,
    }
    default_dofs_kd: dict[str, float] = {
        "*": 50.0,
    }

    # Per-joint armature set to a realistic finger rotor inertia (~3.3e-3). URDF carries no armature field, so otherwise
    # every joint takes Genesis' non-physical 0.1 morph default (~30x heavier); it dominates inertia only at low kd.
    default_dofs_armature: dict[str, float] = {"*": 1.0 / 300}
    actuated_dofs_name: ClassVar[list[str]] = list(INSPIRE_ACTUATED_DOFS_NAME)


#: The couplings the RH56 asset's own ``<mimic>`` tags declare.
_RH56_MIMIC_DOFS = _mimic_dofs(thumb_intermediate=1.334, thumb_distal=0.667, finger=1.06399, finger_offset=-0.04545)


@ROBOT_REGISTRY.register()
class InspireRH56_R(_InspireRH56):
    file: str = "hands/inspire_hand/inspire_hand_right.urdf"
    metadata: ClassVar[MetadataOptions] = _metadata("right", _RH56_MIMIC_DOFS)


@ROBOT_REGISTRY.register()
class InspireRH56_L(_InspireRH56):
    file: str = "hands/inspire_hand/inspire_hand_left.urdf"
    metadata: ClassVar[MetadataOptions] = _metadata("left", _RH56_MIMIC_DOFS)


class _InspireRH56DFQ(_InspireRH56):
    """The RH56DFQ as Unitree mounts it on the G1, with Weave's ``fingers`` actuator group on all twelve joints.

    The asset is split out of Weave's ``g1_29dof_rev_1_0_with_inspire_hand_DFQ.urdf``, ``<mimic>`` tags stripped (see
    the module docstring). Gains are Weave's, followers included: kp 5, kd 0.5, armature 0.01, effort 1. The armature
    is set explicitly — URDF has no armature element, so leaving it to the loader would fill 0.1, 10x Weave's and
    enough to change how a kp-5 finger closes.
    """

    default_dofs_kp: dict[str, float] = {"*": 5.0}
    default_dofs_kd: dict[str, float] = {"*": 0.5}
    default_dofs_armature: dict[str, float] = {"*": 0.01}
    default_dofs_force_limits: dict[str, float] = {"*": 1.0}


#: The couplings of Weave's merged URDF (its ``<mimic>`` tags, which the split strips).
_DFQ_MIMIC_DOFS = _mimic_dofs(thumb_intermediate=1.6, thumb_distal=2.4, finger=1.0, finger_offset=0.0)


@ROBOT_REGISTRY.register()
class InspireRH56DFQ_L(_InspireRH56DFQ):
    """Left Inspire RH56DFQ hand."""

    file: str = "hands/inspire_hand/inspire_hand_dfq_left.urdf"
    metadata: ClassVar[MetadataOptions] = _metadata("left", _DFQ_MIMIC_DOFS)


@ROBOT_REGISTRY.register()
class InspireRH56DFQ_R(_InspireRH56DFQ):
    """Right Inspire RH56DFQ hand."""

    file: str = "hands/inspire_hand/inspire_hand_dfq_right.urdf"
    metadata: ClassVar[MetadataOptions] = _metadata("right", _DFQ_MIMIC_DOFS)
