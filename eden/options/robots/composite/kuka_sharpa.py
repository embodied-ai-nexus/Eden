"""KukaSharpa = KUKA LBR iiwa 14 + SharPa Wave (left), the 29-DOF rig SimToolReal trains on.

SimToolReal's merged 29-DOF URDF (``iiwa14_left_sharpa_adjusted_restricted.urdf``) reaches the hand
through two intermediate frames::

    iiwa14_link_7
      --[iiwa14_joint_ee     xyz=(0, 0, 0.045)  rpy=(0, 0, 0)      ]--> iiwa14_link_ee
      --[iiwa14_sharpa       xyz=(0, 0, 0)      rpy=(0, 0, +15 deg)]--> sharpa_mount
      --[sharpa_mount_joint  xyz=(0, 0, 0.05)   rpy=(0, 0, -90 deg)]--> left_hand_C_MC

Both are pure frames (``sharpa_mount`` carries only a 50 g placeholder mass) and every
rotation is about z, so the chain collapses to the one ``attach_offset`` / ``attach_quat``
weld below. Verified against that URDF: 36/36 links, forward kinematics <= 9e-7 m /
2.5e-4 deg, per-DOF dynamics exact, and a 200-step 29-DOF PD rollout agreeing to 0.002 deg.
Three deliberate differences: the 0.05 kg ``sharpa_mount`` has no counterpart (leaving the
arm's M(q) block 0.5-0.7% low), arm collision is 46 MJCF spheres vs 8 URDF meshes (hand
geometry is identical), and :data:`SIMTOOLREAL_HAND_DOFS_POS_LIMITS` is opt-in.

That agreement also depends on the joint damping and effort limit the iiwa14 MJCF omits;
those are hardware properties so they live on :class:`~eden.options.robots.kuka_iiwa14.Iiwa14`
and are inherited here.
"""

from __future__ import annotations

from genesis.typing import UnitVec4FType, Vec3FType
from pydantic import Field

from eden.options.entities import RobotOptions
from eden.options.materials import RigidMaterialOptions
from eden.options.robots.composite.base import COMPOSITE_ROBOT_REGISTRY, DexManipulatorOptions
from eden.options.robots.kuka_iiwa14 import Iiwa14
from eden.options.robots.sharpa_wave import SharpaWave_L

# ``link7`` -> ``left_hand_C_MC``: the composed chain above. Yaw is exactly 15 - 90 = -75 deg.
SHARPA_MOUNT_OFFSET: tuple[float, float, float] = (0.0, 0.0, 0.095)
SHARPA_MOUNT_QUAT: tuple[float, float, float, float] = (0.7933533402912352, 0.0, 0.0, -0.6087614290087207)

# The five distal phalanges (the *colliding* tips, not SharpaWave_L's massless
# ``left_*_fingertip`` frames one step further out), in SimToolReal's obs order.
_SHARPA_L_TIPS = [
    "left_index_DP",
    "left_middle_DP",
    "left_ring_DP",
    "left_thumb_DP",
    "left_pinky_DP",
]

# Arm home pose the SimToolReal policy resets against (deployment INIT_JOINT_POS).
_IIWA_HOME = {
    "joint1": -1.571,
    "joint2": 1.571,
    "joint3": 0.0,
    "joint4": 1.376,
    "joint5": 0.0,
    "joint6": 1.485,
    "joint7": 1.308,
}

# Isaac Lab actuator gains from SimToolReal's ``scene_utils``, rekeyed onto Eden's stock
# joint names: SimToolReal's URDF renames five proximal joints (``left_thumb_CMC_FE`` ->
# ``left_1_thumb_CMC_FE``, ...) to pin their order, so dicts lifted from it match no DOF here.
SIMTOOLREAL_ARM_KP: dict[str, float] = {
    "joint1": 600.0,
    "joint2": 600.0,
    "joint3": 500.0,
    "joint4": 400.0,
    "joint5": 200.0,
    "joint6": 200.0,
    "joint7": 200.0,
}
SIMTOOLREAL_ARM_KD: dict[str, float] = {
    "joint1": 27.027026473513512,
    "joint2": 27.027026473513512,
    "joint3": 24.672186769721083,
    "joint4": 22.067474708266914,
    "joint5": 9.752538131173853,
    "joint6": 9.147747263670984,
    "joint7": 9.147747263670984,
}
SIMTOOLREAL_HAND_KP: dict[str, float] = {
    "left_thumb_CMC_FE": 6.95,
    "left_thumb_CMC_AA": 13.2,
    "left_thumb_MCP_FE": 4.76,
    "left_thumb_MCP_AA": 6.62,
    "left_thumb_IP": 0.9,
    "left_index_MCP_FE": 4.76,
    "left_index_MCP_AA": 6.62,
    "left_index_PIP": 0.9,
    "left_index_DIP": 0.9,
    "left_middle_MCP_FE": 4.76,
    "left_middle_MCP_AA": 6.62,
    "left_middle_PIP": 0.9,
    "left_middle_DIP": 0.9,
    "left_ring_MCP_FE": 4.76,
    "left_ring_MCP_AA": 6.62,
    "left_ring_PIP": 0.9,
    "left_ring_DIP": 0.9,
    "left_pinky_CMC": 1.38,
    "left_pinky_MCP_FE": 4.76,
    "left_pinky_MCP_AA": 6.62,
    "left_pinky_PIP": 0.9,
    "left_pinky_DIP": 0.9,
}
SIMTOOLREAL_HAND_KD: dict[str, float] = {
    "left_thumb_CMC_FE": 0.28676845,
    "left_thumb_CMC_AA": 0.40845109,
    "left_thumb_MCP_FE": 0.20394083,
    "left_thumb_MCP_AA": 0.24044435,
    "left_thumb_IP": 0.04190723,
    "left_index_MCP_FE": 0.20859232,
    "left_index_MCP_AA": 0.24595532,
    "left_index_PIP": 0.04243185,
    "left_index_DIP": 0.03504461,
    "left_middle_MCP_FE": 0.2085923,
    "left_middle_MCP_AA": 0.24595532,
    "left_middle_PIP": 0.04243185,
    "left_middle_DIP": 0.03504461,
    "left_ring_MCP_FE": 0.20859226,
    "left_ring_MCP_AA": 0.24595528,
    "left_ring_PIP": 0.04243183,
    "left_ring_DIP": 0.0350446,
    "left_pinky_CMC": 0.02782345,
    "left_pinky_MCP_FE": 0.20859229,
    "left_pinky_MCP_AA": 0.24595528,
    "left_pinky_PIP": 0.04243183,
    "left_pinky_DIP": 0.0350446,
}

# Rotor inertia: 0 on the arm (the iiwa's is absorbed by its controller), per-joint on the
# hand (SimToolReal ``HAND_JOINT_ARMATURE``, finer than SharpaWave_L's uniform 1/300).
SIMTOOLREAL_ARM_ARMATURE: dict[str, float] = {f"joint{i}": 0.0 for i in range(1, 8)}

SIMTOOLREAL_HAND_ARMATURE: dict[str, float] = {
    "left_thumb_CMC_FE": 0.0032,
    "left_thumb_CMC_AA": 0.0032,
    "left_thumb_MCP_FE": 0.00265,
    "left_thumb_MCP_AA": 0.00265,
    "left_thumb_IP": 0.0006,
    "left_index_MCP_FE": 0.00265,
    "left_index_MCP_AA": 0.00265,
    "left_index_PIP": 0.0006,
    "left_index_DIP": 0.00042,
    "left_middle_MCP_FE": 0.00265,
    "left_middle_MCP_AA": 0.00265,
    "left_middle_PIP": 0.0006,
    "left_middle_DIP": 0.00042,
    "left_ring_MCP_FE": 0.00265,
    "left_ring_MCP_AA": 0.00265,
    "left_ring_PIP": 0.0006,
    "left_ring_DIP": 0.00042,
    "left_pinky_CMC": 0.0032,
    "left_pinky_MCP_FE": 0.00265,
    "left_pinky_MCP_AA": 0.00265,
    "left_pinky_PIP": 0.0006,
    "left_pinky_DIP": 0.00042,
}

# The "restricted" half of SimToolReal's URDF filename — the only hand-subtree difference from the stock
# asset (finger abduction shrunk 10x, thumb CMC abduction clipped), which is what lets this
# rig be assembled from unit robots at all. Opt-in, since the restriction is SimToolReal's
# training choice rather than SharPa Wave hardware. Note that passing it as the hand's
# ``default_dofs_pos_limits`` only moves Eden's view of the range (soft limits, the
# ``dofs_pos_limits`` penalty) — Genesis takes hard joint stops from the asset alone.
SIMTOOLREAL_HAND_DOFS_POS_LIMITS: dict[str, tuple[float, float]] = {
    "left_thumb_CMC_AA": (-0.3491, 0.1309),
    "left_index_MCP_AA": (-0.03491, 0.03491),
    "left_middle_MCP_AA": (-0.03491, 0.03491),
    "left_ring_MCP_AA": (-0.03491, 0.03491),
    "left_pinky_MCP_AA": (-0.03491, 0.03491),
}


def _iiwa_arm() -> RobotOptions:
    """KUKA iiwa 14 at the SimToolReal home pose, gravity-compensated (the real arm compensates
    internally, and the gains above assume a gravity-free arm — otherwise it sags off pose).
    """
    # Joint damping and the effort limit are iiwa14 hardware properties, so they live on
    # ``Iiwa14`` and are inherited here — the rollout above diverges by ~74 deg without them.
    return Iiwa14(
        default_dofs_pos=dict(_IIWA_HOME),
        default_dofs_kp=dict(SIMTOOLREAL_ARM_KP),
        default_dofs_kd=dict(SIMTOOLREAL_ARM_KD),
        default_dofs_armature=dict(SIMTOOLREAL_ARM_ARMATURE),
        material=RigidMaterialOptions(gravity_compensation=1.0, friction=0.5),
    )


def _sharpa_left() -> RobotOptions:
    """SharPa Wave (left), gravity-compensated, whole hand colliding — like FrankaXHand's hand
    and unlike the pad-only Robotiq, no ``collision_link_patterns`` filter: the fingers must
    contact all over to wrap an object. SimToolReal also bumps the five fingertips to friction
    1.5 at runtime (``link.set_friction``), which no build-time option covers.
    """
    return SharpaWave_L(
        is_fixed_base=False,
        default_dofs_kp=dict(SIMTOOLREAL_HAND_KP),
        default_dofs_kd=dict(SIMTOOLREAL_HAND_KD),
        default_dofs_armature=dict(SIMTOOLREAL_HAND_ARMATURE),
        material=RigidMaterialOptions(gravity_compensation=1.0, friction=0.5),
    )


@COMPOSITE_ROBOT_REGISTRY.register()
class KukaSharpa(DexManipulatorOptions):
    """KUKA iiwa 14 + SharPa Wave (left), a 29-DOF dexterous manipulator (7 arm + 22 hand).

    The SimToolReal rig as two unit robots welded at the arm flange (see the module docstring
    for the mount derivation), so arm and hand each stay swappable. The scene expands it into
    the ``robot`` (arm) and ``hand`` entities::

        scene = en.scenes.Empty(robot=KukaSharpa(base_pos=(0.0, 0.8, 0.0)))

    ``base_pos`` there is SimToolReal's training placement, 0.8 m behind a workspace at the
    world origin.
    """

    arm: RobotOptions = Field(default_factory=_iiwa_arm)
    hand: RobotOptions = Field(default_factory=_sharpa_left)
    # The iiwa MJCF has no body at the tool frame, only an ``attachment_site`` 4.5 cm up the
    # flange z — so the flange is the parent and that offset folds into ``attach_offset``.
    ee_link: str = "link7"
    # Annotations must stay the base class's Vec3FType / UnitVec4FType: those carry the
    # quaternion normalizer and the allow_inf_nan=False guard, which a bare tuple silently drops.
    attach_offset: Vec3FType | None = SHARPA_MOUNT_OFFSET
    attach_quat: UnitVec4FType | None = SHARPA_MOUNT_QUAT
    grasp_links: list[str] = list(_SHARPA_L_TIPS)
