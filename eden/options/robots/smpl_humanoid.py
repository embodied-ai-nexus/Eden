"""SMPL and SMPL-X humanoid configurations."""

from genesis.typing import Vec3FType
from eden.options.catalog import ROBOT_REGISTRY
from eden.options.entities import RobotOptions


#: The 24 bodies of the SMPL humanoid, in MJCF (and Genesis link) order; ``Pelvis`` is the free root.
SMPL_BODY_NAMES = [
    "Pelvis", "L_Hip", "L_Knee", "L_Ankle", "L_Toe", "R_Hip", "R_Knee", "R_Ankle", "R_Toe", "Torso", "Spine",
    "Chest", "Neck", "Head", "L_Thorax", "L_Shoulder", "L_Elbow", "L_Wrist", "L_Hand", "R_Thorax", "R_Shoulder",
    "R_Elbow", "R_Wrist", "R_Hand",
]  # fmt: skip
#: Its 69 hinge DOFs: three per non-root body, ``<body>_x``, ``_y``, ``_z``.
SMPL_DOF_NAMES = [f"{body}_{axis}" for body in SMPL_BODY_NAMES[1:] for axis in "xyz"]


@ROBOT_REGISTRY.register()
class SmplHumanoid(RobotOptions):
    """SMPL humanoid robot (24 bodies, 69 hinge DOFs)."""

    file: str = "humanoids/smpl_humanoid/smpl_humanoid.xml"

    dofs_name: list[str] = SMPL_DOF_NAMES
    default_root_pos: Vec3FType = (0.0, 0.0, 0.9)  # x,y,z [m]


#: PHC's SMPL joint ``stiffness`` per body; every hinge of a body shares it.
#: Isaac Gym reads it and the joint ``damping`` as the gains of a position drive, Genesis as a passive spring
#: and damper.
PHC_SMPL_BODY_STIFFNESS = {
    "L_Hip": 800.0, "L_Knee": 800.0, "L_Ankle": 800.0, "L_Toe": 500.0,
    "R_Hip": 800.0, "R_Knee": 800.0, "R_Ankle": 800.0, "R_Toe": 500.0,
    "Torso": 1000.0, "Spine": 1000.0, "Chest": 1000.0, "Neck": 500.0, "Head": 500.0,
    "L_Thorax": 500.0, "L_Shoulder": 500.0, "L_Elbow": 500.0, "L_Wrist": 300.0, "L_Hand": 300.0,
    "R_Thorax": 500.0, "R_Shoulder": 500.0, "R_Elbow": 500.0, "R_Wrist": 300.0, "R_Hand": 300.0,
}  # fmt: skip
#: Every PHC SMPL joint's ``damping`` is this fraction of its ``stiffness``.
PHC_DAMPING_PER_STIFFNESS = 0.1
#: PHC's nominal gains ring in Genesis (Isaac Gym's under-converged TGS solver makes them effectively softer).
#: Sweep on PHC's AMASS walking clips: nominal gains track 56 steps on average, these 140.
PHC_DRIVE_KP_SCALE = 0.4
PHC_DRIVE_KD_SCALE = 0.7
#: Isaac Gym effort limit of every PHC SMPL DOF (the source MJCF's motor ``gear``).
PHC_EFFORT_LIMIT = 500.0


def _phc_stiffness_times(scale: float) -> dict[str, float]:
    """``{dof: its body's PHC stiffness * scale}`` over the 69 hinge DOFs."""
    return {dof: PHC_SMPL_BODY_STIFFNESS[dof[:-2]] * scale for dof in SMPL_DOF_NAMES}


@ROBOT_REGISTRY.register()
class PhcSmplHumanoid(SmplHumanoid):
    """PHC's ``phc_3`` SMPL humanoid: neutral shape, upright start, box body, real weight, PD-driven.

    The :class:`SmplHumanoid` skeleton (same bodies, joints, geoms and DOF order) generated with the robot
    options PHC trained ``phc_3`` with (``real_weight_porpotion_boxes=True``, so the five box geoms carry their
    real-weight densities instead of 1000) and with the floor, light and motors stripped
    (``examples/projects/mimic/assistmimic/scripts/generate_smpl_xml.py``). Isaac Gym read each joint's
    ``stiffness`` / ``damping`` as a position drive and the motor ``gear`` as its effort limit; here they are
    ``default_dofs_kp`` / ``default_dofs_kd`` (calibrated for Genesis by ``PHC_DRIVE_K*_SCALE``) and
    ``default_dofs_force_limits``, and the passive spring and damper are zeroed.
    """

    file: str = "humanoids/smpl_humanoid/smpl_humanoid_phc.xml"
    default_root_pos: Vec3FType = (0.0, 0.0, 0.95)
    default_dofs_kp: dict[str, float] = _phc_stiffness_times(PHC_DRIVE_KP_SCALE)
    default_dofs_kd: dict[str, float] = _phc_stiffness_times(PHC_DAMPING_PER_STIFFNESS * PHC_DRIVE_KD_SCALE)
    default_dofs_stiffness: dict[str, float] = dict.fromkeys(SMPL_DOF_NAMES, 0.0)
    default_dofs_damping: dict[str, float] = dict.fromkeys(SMPL_DOF_NAMES, 0.0)
    default_dofs_force_limits: dict[str, float] = dict.fromkeys(SMPL_DOF_NAMES, PHC_EFFORT_LIMIT)


class SmplxHumanoid(RobotOptions):
    """Configuration for the SMPL-X humanoid model."""

    file: str = "humanoids/smplx_humanoid/smplx_humanoid.xml"

    up: tuple[int, int, int] = (0, 1, 0)
    front: tuple[int, int, int] = (0, 0, 1)
    default_root_pos: Vec3FType = (0.0, 0.0, 0.9)  # x,y,z [m]

    dofs_name: list[str] = [
        # "Pelvis",
        "L_Hip_x",
        "L_Hip_y",
        "L_Hip_z",
        "L_Knee_x",
        "L_Knee_y",
        "L_Knee_z",
        "L_Ankle_x",
        "L_Ankle_y",
        "L_Ankle_z",
        "L_Toe_x",
        "L_Toe_y",
        "L_Toe_z",
        "R_Hip_x",
        "R_Hip_y",
        "R_Hip_z",
        "R_Knee_x",
        "R_Knee_y",
        "R_Knee_z",
        "R_Ankle_x",
        "R_Ankle_y",
        "R_Ankle_z",
        "R_Toe_x",
        "R_Toe_y",
        "R_Toe_z",
        "Torso_x",
        "Torso_y",
        "Torso_z",
        "Spine_x",
        "Spine_y",
        "Spine_z",
        "Chest_x",
        "Chest_y",
        "Chest_z",
        "Neck_x",
        "Neck_y",
        "Neck_z",
        "Head_x",
        "Head_y",
        "Head_z",
        "L_Thorax_x",
        "L_Thorax_y",
        "L_Thorax_z",
        "L_Shoulder_x",
        "L_Shoulder_y",
        "L_Shoulder_z",
        "L_Elbow_x",
        "L_Elbow_y",
        "L_Elbow_z",
        "L_Wrist_x",
        "L_Wrist_y",
        "L_Wrist_z",
        "L_Index1_x",
        "L_Index1_y",
        "L_Index1_z",
        "L_Index2_x",
        "L_Index2_y",
        "L_Index2_z",
        "L_Index3_x",
        "L_Index3_y",
        "L_Index3_z",
        "L_Middle1_x",
        "L_Middle1_y",
        "L_Middle1_z",
        "L_Middle2_x",
        "L_Middle2_y",
        "L_Middle2_z",
        "L_Middle3_x",
        "L_Middle3_y",
        "L_Middle3_z",
        "L_Pinky1_x",
        "L_Pinky1_y",
        "L_Pinky1_z",
        "L_Pinky2_x",
        "L_Pinky2_y",
        "L_Pinky2_z",
        "L_Pinky3_x",
        "L_Pinky3_y",
        "L_Pinky3_z",
        "L_Ring1_x",
        "L_Ring1_y",
        "L_Ring1_z",
        "L_Ring2_x",
        "L_Ring2_y",
        "L_Ring2_z",
        "L_Ring3_x",
        "L_Ring3_y",
        "L_Ring3_z",
        "L_Thumb1_x",
        "L_Thumb1_y",
        "L_Thumb1_z",
        "L_Thumb2_x",
        "L_Thumb2_y",
        "L_Thumb2_z",
        "L_Thumb3_x",
        "L_Thumb3_y",
        "L_Thumb3_z",
        "R_Thorax_x",
        "R_Thorax_y",
        "R_Thorax_z",
        "R_Shoulder_x",
        "R_Shoulder_y",
        "R_Shoulder_z",
        "R_Elbow_x",
        "R_Elbow_y",
        "R_Elbow_z",
        "R_Wrist_x",
        "R_Wrist_y",
        "R_Wrist_z",
        "R_Index1_x",
        "R_Index1_y",
        "R_Index1_z",
        "R_Index2_x",
        "R_Index2_y",
        "R_Index2_z",
        "R_Index3_x",
        "R_Index3_y",
        "R_Index3_z",
        "R_Middle1_x",
        "R_Middle1_y",
        "R_Middle1_z",
        "R_Middle2_x",
        "R_Middle2_y",
        "R_Middle2_z",
        "R_Middle3_x",
        "R_Middle3_y",
        "R_Middle3_z",
        "R_Pinky1_x",
        "R_Pinky1_y",
        "R_Pinky1_z",
        "R_Pinky2_x",
        "R_Pinky2_y",
        "R_Pinky2_z",
        "R_Pinky3_x",
        "R_Pinky3_y",
        "R_Pinky3_z",
        "R_Ring1_x",
        "R_Ring1_y",
        "R_Ring1_z",
        "R_Ring2_x",
        "R_Ring2_y",
        "R_Ring2_z",
        "R_Ring3_x",
        "R_Ring3_y",
        "R_Ring3_z",
        "R_Thumb1_x",
        "R_Thumb1_y",
        "R_Thumb1_z",
        "R_Thumb2_x",
        "R_Thumb2_y",
        "R_Thumb2_z",
        "R_Thumb3_x",
        "R_Thumb3_y",
        "R_Thumb3_z",
    ]
