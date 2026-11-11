"""Droid = Franka arm + Robotiq 2F85 (after mjlab's ``Droid``)."""

from __future__ import annotations

from pydantic import Field

from eden.options.entities import RobotOptions
from eden.options.materials import RigidMaterialOptions
from eden.options.robots.composite.base import COMPOSITE_ROBOT_REGISTRY, ManipulatorOptions
from eden.options.robots.franka_emika_panda import FrankaEmikaPandaNoHand
from eden.options.robots.robotiq_2f85 import Robotiq2f85

# Arm home pose proven to let the Franka reach down and grasp a cube on the
# ground plane (also shared by the dexterous ``manipulation_dex_lift`` task).
_FRANKA_HOME = {
    "joint1": 0.0,
    "joint2": -0.569,
    "joint3": 0.0,
    "joint4": -2.810,
    "joint5": 0.0,
    "joint6": 3.037,
    "joint7": 0.741,
}


def _droid_arm() -> RobotOptions:
    """The hand-less Panda: the arm every policy trained on the Droid was trained on.

    Pass ``Droid(arm=FrankaResearch3())`` for the FR3 mjlab's Droid uses.
    """
    return FrankaEmikaPandaNoHand(
        default_dofs_pos=dict(_FRANKA_HOME),
        material=RigidMaterialOptions(gravity_compensation=1.0),
    )


def _droid_gripper() -> RobotOptions:
    """Robotiq 2F85 whose *only* colliding geoms are the fingertip pads.

    Shared verbatim by every manipulator rig (the UR composites import it, and so
    does ``PolarisDroid``) — the same gripper on every arm is what makes the grasp
    point, gripper action, and every gripper-keyed reward embodiment-invariant.
    Otherwise the policy wedges the object against the inner linkage/palm (and it
    interpenetrates) instead of pinching it between the fingers. The flat box pads
    (friction 1.2) and force-limited drivers come from :class:`Robotiq2f85`.
    """
    return Robotiq2f85(
        batch_fixed_verts=True,
        is_fixed_base=False,
        material=RigidMaterialOptions(gravity_compensation=1.0),
        collision_link_patterns=[".*pad"],
    )


@COMPOSITE_ROBOT_REGISTRY.register()
class Droid(ManipulatorOptions):
    """Franka arm + Robotiq 2F85, after mjlab's ``Droid``.

    The arm is the hand-less Panda (:func:`_droid_arm`), not the FR3 mjlab pairs the gripper
    with: the rig was built and tuned against that Panda, and keeping it avoids silently
    re-tuning every policy trained here. One keyword gets the faithful pairing:
    ``Droid(arm=FrankaResearch3())``.

    Construct directly like any other Options preset — ``Droid(base_pos=(0, 0, 0.4))``
    — and add it to a scene by name (``en.scenes.Empty(robot=Droid(...))``); the
    scene expands it into the ``robot`` (arm) and ``gripper`` entities welded at the
    Franka's ``attachment`` link.
    """

    arm: RobotOptions = Field(default_factory=_droid_arm)
    gripper: RobotOptions = Field(default_factory=_droid_gripper)
    ee_link: str = "attachment"
    grasp_links: list[str] = ["left_pad", "right_pad"]
