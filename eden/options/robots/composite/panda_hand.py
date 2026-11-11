"""Franka Emika Panda arm + Franka Hand, as a composite rig.

The same robot as :class:`~eden.options.robots.FrankaEmikaPanda` (one MJCF, arm and fingers in
one entity) but assembled the way every other manipulator rig is: the arm-only
:class:`~eden.options.robots.FrankaEmikaPandaNoHand` with the :class:`~eden.options.robots.FrankaHand`
welded onto its ``attachment`` flange. That is what lets a task written against the composite
contract — ``arm_name`` / ``gripper_name``, ``ee_link``, ``grasp_links``, the gripper's tool point
and stroke — swap this rig for a Droid or a UR + Robotiq by registry name alone, with no
single-entity special case. The hand lands exactly where ``panda.xml`` puts it: the ``attachment``
flange both Eden Franka arms expose is 135 degrees about z, and ``franka_hand.xml``'s root carries
the complementary 180 degrees.
"""

from __future__ import annotations

from pydantic import Field

from eden.options.entities import RobotOptions
from eden.options.materials import RigidMaterialOptions
from eden.options.robots.composite.base import COMPOSITE_ROBOT_REGISTRY, ManipulatorOptions
from eden.options.robots.franka_emika_panda import FrankaEmikaPandaNoHand
from eden.options.robots.franka_hand import FrankaHand


def _panda_arm() -> RobotOptions:
    return FrankaEmikaPandaNoHand(material=RigidMaterialOptions(gravity_compensation=1.0))


def _franka_hand() -> RobotOptions:
    """The Franka Hand as a welded child: free-base (its parent link poses it) and gravity-compensated like the arm.

    ``batch_fixed_verts=True`` is what Genesis' ``attach`` requires of a child whose MJCF root has no free
    joint (the same flag the Robotiq carries in :func:`~eden.options.robots.composite.droid._droid_gripper`).
    """
    return FrankaHand(
        is_fixed_base=False, batch_fixed_verts=True, material=RigidMaterialOptions(gravity_compensation=1.0)
    )


@COMPOSITE_ROBOT_REGISTRY.register(name="panda_franka_hand")
class PandaFrankaHand(ManipulatorOptions):
    """Panda arm + Franka Hand, welded at the arm's ``attachment`` flange.

    Registered ``panda_franka_hand`` (``<arm>_<end-effector>``, like ``ur5e_2f85``): ``panda``
    alone is the single-entity :class:`~eden.options.robots.FrankaEmikaPanda` in ``ROBOT_REGISTRY``.
    """

    arm: RobotOptions = Field(default_factory=_panda_arm)
    gripper: RobotOptions = Field(default_factory=_franka_hand)
    ee_link: str = "attachment"
    grasp_links: list[str] = ["left_gripper_pad", "right_gripper_pad"]
