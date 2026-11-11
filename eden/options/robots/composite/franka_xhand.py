"""FrankaXHand = Franka arm + RobotEra XHand1 (left) on the LeFranX coupling, the dexterous ``Droid``."""

from __future__ import annotations

from pydantic import Field

from genesis.typing import UnitVec4FType, Vec3FType

from eden.options.entities import RobotOptions
from eden.options.materials import RigidMaterialOptions
from eden.options.robots.composite.base import COMPOSITE_ROBOT_REGISTRY, DexManipulatorOptions
from eden.options.robots.composite.droid import _FRANKA_HOME
from eden.options.robots.franka_emika_panda import FrankaEmikaPandaNoHand
from eden.options.robots.robotera_xhand import XHand1_L


def _franka_arm() -> RobotOptions:
    # Same Franka + home pose as Droid (imported so the two rigs can't drift).
    return FrankaEmikaPandaNoHand(
        default_dofs_pos=dict(_FRANKA_HOME),
        material=RigidMaterialOptions(gravity_compensation=1.0),
    )


def _xhand_left() -> RobotOptions:
    """XHand1 (left), gravity-compensated, whole hand colliding.

    Unlike the Robotiq gripper (pad-only collision), the dexterous fingers must contact
    the object all over to wrap it, so no ``collision_link_patterns`` filter is applied.
    """
    return XHand1_L(
        is_fixed_base=False,
        material=RigidMaterialOptions(gravity_compensation=1.0),
    )


# The XHand1 (left) root -- the centre of its wrist face -- in the Panda's ``attachment`` frame when the hand hangs on
# the LeFranX right-angle coupling (https://github.com/wengmister/xhand-coupling, see FrankaXHand).
_LEFRANX_MOUNT_OFFSET = (0.0, 0.037, 0.046)
_LEFRANX_MOUNT_QUAT = (0.5, 0.5, -0.5, 0.5)


@COMPOSITE_ROBOT_REGISTRY.register()
class FrankaXHand(DexManipulatorOptions):
    """Franka arm + RobotEra XHand1 (left) on the LeFranX right-angle coupling, a 12-DOF dexterous manipulator.

    The dexterous counterpart of :class:`~eden.options.robots.composite.droid.Droid`:
    same Franka arm and home pose, but the two-finger gripper is swapped for the XHand1
    hand, welded to the Franka's ``attachment`` frame and driven in joint space. The
    grasp point is the fingertip centroid. Construct like any Options preset —
    ``FrankaXHand(base_pos=(0, 0, 0.4))`` — and add it to a scene by name
    (``en.scenes.Empty(robot=FrankaXHand(...))``); the scene expands it into the
    ``robot`` (arm) and ``hand`` entities.

    The hand hangs on the coupling (`wengmister/xhand-coupling <https://github.com/wengmister/xhand-coupling>`_, the
    mount of the LeFranX Franka + XHand teleoperation stack), a plate bolted to the flange plus a holder hanging off
    it. The hand's finger axis is perpendicular to the flange axis and its palm faces along it, so with the flange
    pointing down the fingers are horizontal and the palm faces down. Both CAD parts are exported in one assembly
    frame, which fixes the transform: the hand's Ø38 wrist bore slides over the holder's 13 mm boss and the wrist face
    rests on the seat around it, 46 mm out from the flange face and 37 mm off the flange axis; the fingers point back
    across that axis.

    The clocking about the flange axis is the one measured on a lab robot: at the Franka ready pose
    ``(0, -pi/4, 0, -3pi/4, 0, pi/2, pi/4)`` the fingertips point along the base frame's -y, the palm faces down and
    the thumb is on the base's side. That is one 90-degree bolt step off the coupling's dowel-pin hole. A coupling
    mounted on its pin instead hangs the hand at ``attach_offset=(-0.037, 0.0, 0.046)``,
    ``attach_quat=(0.0, 0.7071068, 0.0, 0.7071068)`` (fingertips along -x at the ready pose). A hand on another
    mount takes that mount's ``attach_offset`` / ``attach_quat`` the same way.

    Only the weld is modelled. The coupling itself (75 x 77 x 95 mm envelope) has no geometry here, so nothing
    collides with it.
    """

    arm: RobotOptions = Field(default_factory=_franka_arm)
    hand: RobotOptions = Field(default_factory=_xhand_left)
    ee_link: str = "attachment"
    # The five XHand1 (left) fingertip links (all carry collision geoms) — the grasp point is
    # their centroid, the EE the shared manipulation terms use. Read off the robot's own
    # metadata so it can't drift from the asset (same source XArm7XHand1 uses).
    grasp_links: list[str] = list(XHand1_L.metadata.target_hand_links)
    # Annotations must stay the base class's Vec3FType / UnitVec4FType: those carry the quaternion normalizer and the
    # NaN guard, which a bare tuple annotation would drop.
    attach_offset: Vec3FType | None = _LEFRANX_MOUNT_OFFSET
    attach_quat: UnitVec4FType | None = _LEFRANX_MOUNT_QUAT
