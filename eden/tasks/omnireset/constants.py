"""The OmniReset task as UWLab defines it: its scene, controller, timing and the order its data comes in.

Every value is UWLab's (``uwlab_tasks/manager_based/manipulation/omnireset/config/ur5e_robotiq_2f85/rl_state_cfg.py``
and the props' ``metadata.yaml``) unless its comment says how it was adapted to Genesis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

UWLAB_REPO = "UW-Lab/uwlab-assets"
#: ``isaaclab3`` of ``UW-Lab/uwlab-assets``: the reset datasets (xyzw quaternions) and the policies UWLab's docs pin.
DATASETS_REVISION = "83860532010b2737aa80d6e8621235ee554186f0"

#: Isaac Lab's joint order for the robot (PhysX breadth-first over the articulation): the order of every
#: joint-indexed array in UWLab's data. Genesis orders the gripper joints depth-first, so read them by name.
ISAAC_JOINT_ORDER = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
    "finger_joint",
    "right_outer_knuckle_joint",
    "right_inner_knuckle_joint",
    "left_inner_knuckle_joint",
    "right_inner_finger_knuckle_joint",
    "left_inner_finger_knuckle_joint",
)

#: UWLab's ``Ur5eRobotiq2f85RelativeOSCAction`` (stage-1 training and eval): soft task-space gains.
OSC_SCALE = (0.02, 0.02, 0.02, 0.02, 0.02, 0.2)
OSC_KP = (200.0, 200.0, 200.0, 3.0, 3.0, 3.0)
OSC_DAMPING_RATIO = (3.0, 3.0, 3.0, 1.0, 1.0, 1.0)
OSC_TORQUE_LIMIT = (150.0, 150.0, 150.0, 28.0, 28.0, 28.0)
#: ``ROBOTIQ_GRIPPER_BINARY_ACTIONS``: finger_joint open / closed targets, rad.
GRIPPER_OPEN, GRIPPER_CLOSED = 0.0, 0.785398
#: UR5e joint velocity limits (rad/s); an episode ends past twice these (``abnormal_robot_state``).
ARM_VELOCITY_LIMITS = {
    "shoulder_pan_joint": 1.5708,
    "shoulder_lift_joint": 1.5708,
    "elbow_joint": 1.5708,
    "wrist_1_joint": 3.1415,
    "wrist_2_joint": 3.1415,
    "wrist_3_joint": 3.1415,
}

#: The reset-state families, in the order UWLab's training config lists them (each drawn with p = 0.25).
RESET_FAMILIES = (
    "ObjectAnywhereEEAnywhere",
    "ObjectRestingEEGrasped",
    "ObjectAnywhereEEGrasped",
    "ObjectPartiallyAssembledEEGrasped",
)
#: The one family UWLab evaluates from (``...-State-Play-v0``): every episode starts from it.
EVAL_FAMILY = "ObjectAnywhereEEAnywhere"

# Timing: UWLab's. PhysX steps at 120 Hz and the arm torques are recomputed every step; the policy acts every 12
# steps (10 Hz) for 16 s episodes. Genesis substeps each step 4x (the torque held across them) so the grasp contacts
# stay stiff, see GRASP_SOL_PARAMS. The controller's damping ``J^T Kd J qdot`` is explicit, so its stability rests
# on the wrist's inertia, most of which is the two collider-less gripper knuckles PhysX gives 1 kg each
# (see :class:`~eden.options.robots.OmniResetUR5e2f85`). Without them every episode runs away at 120 Hz. With them, from
# ObjectAnywhereEEAnywhere, Eden succeeds in 95.0 % of 2,048 episodes (95.3 % of 256, 4 runaway ends) against
# UWLab's 98.0 % of 256 in Isaac Sim 6.1 (2 runaway ends). Even in PhysX a few envs are only held by its joint speed
# limits (17/256 sit at 3.14 rad/s after 1 s of zero action), which Genesis does not enforce; the abnormal-robot
# termination ends those episodes in both.
SIM_DT = 1 / 120
SIM_SUBSTEPS = 4
DECIMATION = 12
EPISODE_LENGTH_S = 16.0

#: Contact/equality solver parameters (timeconst, dampratio, dmin, dmax, width, mid, power) for the gripper pads,
#: the objects and the gripper's joint couplings: near-rigid, as PhysX's iterated contacts are. At the default
#: (0.02 s time constant) a peg held in the air slides out of a closed grasp (median slip 27.9 mm over 0.5 s);
#: with these, UWLab's mid-air grasp states hold (3.8 mm). Genesis clamps the time constant to 2 substeps (4.2 ms).
GRASP_SOL_PARAMS = (0.002, 1.0, 0.95, 0.999, 0.001, 0.5, 2.0)

# Fixed scene: the table (``pat_vention``) and the ground plane under it. World frame = Isaac Lab env frame.
TABLE_POS = (0.4, 0.0, -0.881)
TABLE_QUAT = (0.70710678, 0.0, 0.0, -0.70710678)  # wxyz of UWLab's xyzw (0, 0, -0.707, 0.707)
GROUND_Z = -0.868

# Friction. UWLab randomizes each body's static friction at startup and PhysX averages a pair's two values;
# Genesis takes the pair's max. Values here put each pair the object touches near PhysX's mean average:
# object-receptive max(1.0, 0.4) vs (1.5 + 0.4) / 2, object-table 1.0 vs 0.98, object-finger 1.0 vs 1.1.
INSERTIVE_FRICTION = 1.0
RECEPTIVE_FRICTION = 0.4
TABLE_FRICTION = 0.45
ROBOT_FRICTION = 0.75

#: Mass of the insertive object, kg: the midpoint of UWLab's startup draw, U(0.02, 0.2). The free prop's MJCF
#: carries it as an ``<inertial>`` (centre of mass and inertia of its colliders), which is where Genesis reads it from.
INSERTIVE_MASS = 0.11

#: Scene entity names, as the reset buffers key them.
ROBOT, MOUNT, INSERTIVE, RECEPTIVE = "robot", "ur5_metal_support", "insertive_object", "receptive_object"
#: The composite rig's gripper entity (its arm is ``ROBOT``).
GRIPPER = "gripper"

#: The robot base and mount plate before a reset state places them (UWLab draws the base within +-1 cm of
#: this in x and z, +-2 cm in y); the plate sits 13 mm under the base.
ROBOT_BASE_POS = (0.0, -0.039, 0.0)
MOUNT_OFFSET_Z = -0.0129


#: UWLab's props converted to MJCF in Eden's asset hub (``objects/omnireset/<name>/<name>.xml``).
PROPS = ("peg", "peg_hole", "pat_vention", "ur5_metal_support")


def prop_file(name: str) -> str:
    """Prop ``name``'s MJCF in Eden's asset hub; an unknown name raises here, not as a missing download at build."""
    if name not in PROPS:
        raise ValueError(
            f"No OmniReset prop {name!r}; props: {list(PROPS)}. Fix: name one of them, or upload the converted MJCF to "
            f"objects/omnireset/{name}/{name}.xml on Kashu7100/eden_assets and add {name!r} to PROPS in "
            "eden/tasks/omnireset/constants.py."
        )
    return f"objects/omnireset/{name}/{name}.xml"


@dataclass(frozen=True)
class Pair:
    """An insertive/receptive object pair: its props, its reset datasets and its success test."""

    insertive: str
    receptive: str
    dataset_dir: str  # ``Datasets/OmniReset_isaaclab3/Resets/<dataset_dir>/resets_<family>.pt``
    #: Insertive pose in the receptive frame once assembled (``receptive metadata.yaml: assembled_offset``).
    assembled_pos: tuple[float, float, float]
    assembled_quat: tuple[float, float, float, float]  # wxyz
    success_position: float  # m
    success_orientation: float  # rad
    #: Insertive centre of mass in its frame, as PhysX derives it from the colliders. Pinned so that converting the
    #: reset states needs no USD.
    insertive_com: tuple[float, float, float]

    def __post_init__(self) -> None:
        for name in (self.insertive, self.receptive):  # an unknown prop fails where the pair is defined
            prop_file(name)

    def dataset_file(self, family: str) -> str:
        return f"Datasets/OmniReset_isaaclab3/Resets/{self.dataset_dir}/resets_{family}.pt"


PAIRS = {
    "peg": Pair(
        insertive="peg",
        receptive="peg_hole",
        dataset_dir="Peg__PegHole",
        assembled_pos=(0.0, 0.0, 0.014837),
        assembled_quat=(1.0, 0.0, 0.0, 0.0),
        success_position=0.0025,
        success_orientation=0.025,
        insertive_com=(0.0, 0.0, 0.0),
    ),
}


# Robot rigs. ``"uwlab"`` is UWLab's own UR5e + Robotiq 2F-85 (:class:`~eden.options.robots.OmniResetUR5e2f85`),
# converted from their USD with PhysX's inertials: the released experts observe and act on it, so the policy port and
# ``omnireset`` use it by default. ``"ur5e_2f85"`` is Eden's composite
# (:class:`~eden.options.robots.composite.ur_robotiq.UR5eRobotiq`), for training a policy of our own
# (``omnireset --robot ur5e_2f85``); ``config.robot_entities`` says how it differs from the stock composite.

RigName = Literal["uwlab", "ur5e_2f85"]

#: Gripper base offset from the UR5e flange (``ee_link`` frame, m) that puts the composite's pad midpoint on
#: UWLab's grasp point: measured over random arm poses (spread under 1 mm). The 8 mm along the approach axis is
#: UWLab's D415 mount plate less the two models' grasp-point definitions; the 2 mm across is the arm calibration.
COMPOSITE_MOUNT_OFFSET = (-0.00205, -0.00026, 0.00805)
#: UWLab's system-identified UR5e motor armature (robot ``metadata.yaml: sysid.armature``), kg m^2.
UR5E_SYSID_ARMATURE = (3.0026, 1.2213, 1.3881, 0.1773, 0.0825, 0.3825)
#: The composite gripper's substep must stay under 1 ms for its loop closures (``Robotiq2f85.max_substep_dt``).
_COMPOSITE_SUBSTEPS = 9


@dataclass(frozen=True)
class Rig:
    """Names and settings that change with the robot; the scene, terms and tasks read them from here."""

    name: RigName
    arm_entity: str
    arm_joints: tuple[str, ...]
    #: ``(entity, joints)`` the joint-angle observation reads, in order.
    joint_observation: tuple[tuple[str, tuple[str, ...]], ...]
    base_link: str  # the arm's root link: the frame of the end-effector pose and the controller
    ee_link: str  # the controlled link
    gripper_entity: str
    gripper_drivers: tuple[str, ...]
    gripper_closed: float
    #: Grasp point: a link of an entity and an offset in its frame.
    grasp_entity: str
    grasp_link: str
    grasp_offset: tuple[float, float, float]
    sim_substeps: int

    @property
    def arm_velocity_limits(self) -> dict[str, float]:
        return dict(zip(self.arm_joints, ARM_VELOCITY_LIMITS.values(), strict=True))


UWLAB_ARM_JOINTS = tuple(ISAAC_JOINT_ORDER[:6])
EDEN_ARM_JOINTS = tuple(name.removesuffix("_joint") for name in UWLAB_ARM_JOINTS)
_EDEN_GRIPPER_JOINTS = (
    "left_driver_joint",
    "left_spring_link_joint",
    "left_follower",
    "right_driver_joint",
    "right_spring_link_joint",
    "right_follower_joint",
)

RIGS: dict[str, Rig] = {
    "uwlab": Rig(
        name="uwlab",
        arm_entity=ROBOT,
        arm_joints=UWLAB_ARM_JOINTS,
        joint_observation=((ROBOT, ISAAC_JOINT_ORDER),),
        base_link="base_link",
        ee_link="wrist_3_link",
        gripper_entity=ROBOT,
        gripper_drivers=("finger_joint",),
        gripper_closed=GRIPPER_CLOSED,
        grasp_entity=ROBOT,
        grasp_link="robotiq_base_link",
        grasp_offset=(0.1345, 0.0, 0.0),
        sim_substeps=SIM_SUBSTEPS,
    ),
    "ur5e_2f85": Rig(
        name="ur5e_2f85",
        arm_entity=ROBOT,
        arm_joints=EDEN_ARM_JOINTS,
        joint_observation=((ROBOT, EDEN_ARM_JOINTS), (GRIPPER, _EDEN_GRIPPER_JOINTS)),
        base_link="world",
        ee_link="ee_link",
        gripper_entity=GRIPPER,
        gripper_drivers=("left_driver_joint", "right_driver_joint"),
        gripper_closed=0.8,
        grasp_entity=GRIPPER,
        grasp_link="base",
        grasp_offset=(0.0, 0.0, 0.1413),
        sim_substeps=_COMPOSITE_SUBSTEPS,
    ),
}
