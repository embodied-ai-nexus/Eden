"""XArm7XHand = UFACTORY xArm7 + RobotEra XHand1 (left/right), ConTrack's rig.

ConTrack ships the arm and hand pre-welded in one URDF
(``assets/urdf/xarm_xhand_{left,right}.urdf``). This composite reproduces that
chain from Eden's own unit robots — :class:`~eden.options.robots.ufactory_xarm7.Xarm7`
plus :class:`~eden.options.robots.robotera_xhand.XHand1_L`/``XHand1_R`` — welded at
the arm flange, so the rig needs no ConTrack-specific asset. Genesis ``attach``
removes the hand's root joint and merges both bodies into a single kinematic tree,
so this is a true fixed joint, not a constraint-based weld.

Equivalence to ConTrack's combined URDF (verified in
``tests/options/robots/composite/test_xarm_xhand.py``):

- The xArm7 chain (``link_base``..``link7``: joint origins, axes, limits, masses,
  inertias, dynamics) is **bit-identical** between Eden's ``arms/ufactory_xarm7/xarm.urdf``
  and the combined file.
- ConTrack welds the hand with ``joint_eef``: ``link7 -> {side}_hand_link`` at
  ``xyz=(0, 0, 0.007) rpy=(3.14, 0, ±1.57)``. Eden's ``link_eef`` is a zero-offset fixed
  child of ``link7``, and Eden's XHand1 assets carry that ``rpy`` rotation *inside* the
  asset (their base frame is the flange-aligned convention), so the mount reduces to
  :data:`_MOUNT_OFFSET` with an identity rotation.
- ``rpy="3.14 0 ±1.57"`` truncates π and π/2, i.e. ConTrack's weld is **1.781 mrad** off the
  exact rotation. Mount at ConTrack's own truncated value and all 38 link positions agree to
  292 nm — the chains are the same geometry. This rig mounts at the *exact* rotation instead,
  which leaves a 0.37 mm systematic offset at the fingertips versus the released clips. That
  is deliberate: this is a general-purpose preset, so it describes the real robot rather than
  reproducing one dataset's typo, and 0.37 mm is ~80x below the tightest std any ConTrack
  reward uses (3 cm contact distance). Override ``attach_quat`` if you need the truncated
  frame bit-for-bit.
- Every hand link **frame** is rotated by a constant 180° (``(1, ±1, 0)/√2``) relative to
  the combined URDF: the two XHand1 revisions use mirrored per-link frame conventions.
  Link positions, joint values, and dynamics are unaffected; only link *quaternions*
  differ, so terms that read a hand link's orientation are not frame-comparable with
  the source implementation.

Known asset-revision deltas (Eden's XHand1 is a later revision than ConTrack's copy):
per-link masses differ by ≤ 8 g (largest: right ``thumb_rota_link2``, 35 g vs 43 g),
the ``*back*``/``*tip*`` massless stubs carry different placeholder inertias, and
``thumb_rota_joint1/2`` cap at 1.570 rad instead of 1.745. The bundled clips peak at
1.505 rad on ``thumb_rota_joint1``, so the reference stays reachable — but the residual
policy's ±0.25 rad finger action clips ~0.07 rad earlier than in the source.
"""

from __future__ import annotations

from functools import partial

from genesis.typing import Vec3FType
from pydantic import Field

from eden.options.entities import RobotOptions
from eden.options.robots.composite.base import COMPOSITE_ROBOT_REGISTRY, DexManipulatorOptions
from eden.options.robots.robotera_xhand import XHand1_L, XHand1_R
from eden.options.robots.ufactory_xarm7 import Xarm7

# ConTrack's ``joint_eef`` translation. Its rotation is omitted on purpose: it equals the
# frame convention already baked into Eden's XHand1 assets (see the module docstring), so
# the mount orientation is identity.
_MOUNT_OFFSET = (0.0, 0.0, 0.007)

# ConTrack's xArm7 home pose (IsaacLab ``robots/xarm_xhand_right.py`` init state), in
# ``joint1``..``joint7`` order. Overrides Eden's tabletop-ready ``Xarm7`` default, which
# was picked for gripper tasks and puts the hand nowhere near the clip's first frame.
_XARM_HOME = (0.0, 0.0, 0.0, 0.22, 3.14, 1.33, 0.0)

# PD gains / effort limits from ConTrack's IsaacLab actuator config
# (``robots/xarm_xhand_right.py``): arm kp=1000 kd=50 effort=50; hand kp=100 kd=10
# effort=1. Both differ from Eden's stock ``Xarm7`` / ``XHand1`` defaults (tuned for
# position-controlled pick-and-place and Quest teleop respectively), so the rig restates
# them rather than inheriting. NOTE: under ``EnvOptions.integrator="euler"`` only, Eden's
# implicit-damping stabilization adds ``kd * substep_dt`` to the armature at build time, so
# the effective armature differs slightly from the configured value; the implicit
# integrators (the default included) keep it as configured.
_ARM_KP, _ARM_KD, _ARM_EFFORT = 1000.0, 50.0, 50.0
_HAND_KP, _HAND_KD, _HAND_EFFORT = 100.0, 10.0, 1.0


def _contrack_arm() -> RobotOptions:
    return Xarm7(
        default_dofs_pos={f"joint{i + 1}": q for i, q in enumerate(_XARM_HOME)},
        default_dofs_kp={"joint[1-7]": _ARM_KP},
        default_dofs_kd={"joint[1-7]": _ARM_KD},
        default_dofs_force_limits={"joint[1-7]": _ARM_EFFORT},
    )


def _contrack_hand(hand_cls: type[RobotOptions]) -> RobotOptions:
    """XHand1, open, with ConTrack's finger actuator config.

    ``is_fixed_base=False`` is what lets Genesis ``attach`` drop the hand's root joint and
    absorb it into the arm's tree; a fixed-base child would instead need
    ``batch_fixed_verts=True``. The whole hand collides (no ``collision_link_patterns``
    filter) — dexterous fingers must contact the object all over to wrap it.
    """
    return hand_cls(
        is_fixed_base=False,
        default_dofs_pos={"*": 0.0},
        default_dofs_kp={"*": _HAND_KP},
        default_dofs_kd={"*": _HAND_KD},
        default_dofs_force_limits={"*": _HAND_EFFORT},
    )


class _XArm7XHand1(DexManipulatorOptions):
    """Everything the two rigs share; the subclasses differ only in the hand's chirality."""

    arm: RobotOptions = Field(default_factory=_contrack_arm)
    ee_link: str = "link_eef"
    # The base class's Vec3FType keeps the NaN guard (a bare tuple annotation would drop it) and admits None.
    attach_offset: Vec3FType | None = _MOUNT_OFFSET


@COMPOSITE_ROBOT_REGISTRY.register(name="xarm7_xhand1_r")
class XArm7XHand1_R(_XArm7XHand1):
    """xArm7 + XHand1 (right), the 19-DOF rig ConTrack's right-hand clips track."""

    hand: RobotOptions = Field(default_factory=partial(_contrack_hand, XHand1_R))
    grasp_links: list[str] = list(XHand1_R.metadata.target_hand_links)


@COMPOSITE_ROBOT_REGISTRY.register(name="xarm7_xhand1_l")
class XArm7XHand1_L(_XArm7XHand1):
    """xArm7 + XHand1 (left), the 19-DOF rig ConTrack's left-hand clips track."""

    hand: RobotOptions = Field(default_factory=partial(_contrack_hand, XHand1_L))
    grasp_links: list[str] = list(XHand1_L.metadata.target_hand_links)
