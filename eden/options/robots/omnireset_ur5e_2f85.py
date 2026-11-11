"""UWLab's OmniReset rig: a calibrated UR5e carrying a Robotiq 2F-85 with a D415 mount.

Converted once from UWLab's USD (``UW-Lab/uwlab-assets``), which Genesis cannot load: the gripper's parallel
linkage is a PhysX articulation with two parents per inner knuckle, coupled by ``physxMimicJoint`` constraints.
The MJCF rebuilds it as a tree. Each body is placed from its PhysX joint frames at q = 0, so every joint reads
PhysX's coordinate; the five mimic joints become joint equalities and the two loop-closing joints are dropped.
One authored gearing is corrected: ``right_inner_knuckle_joint`` says -1, but its hinge is flipped relative to its
mirror, which opens a 72 mm gap in the right finger's loop when closed; +1 closes it to 0.08 mm, and it matches
Isaac Lab's recorded states, which read that joint as ``-finger_joint``. Kinematics match UWLab's calibrated
analytic model (``uwlab_assets.robots.ur5e_robotiq_gripper.kinematics``) to 1e-7 m; joint values
are PhysX's, so a pose reads what Isaac Lab reports for it.

One asset rather than a ``ManipulatorOptions`` composite of :class:`UR5e` and :class:`Robotiq2f85`,
because neither part is the same robot: ``UR5e`` carries rounded nominal link lengths, putting its
flange 1.2 mm (max 2.3 mm) from this calibrated one across random poses — the size of OmniReset's
2.5 mm success tolerance — and ``Robotiq2f85`` is MuJoCo Menagerie's loop-closed 2F-85 (other joints,
pad-only colliders, sub-millisecond substeps, no D415 mount), not the mimic linkage whose six joint
angles the policy observes.

Inertials are PhysX's. The arm links' are authored in the USD; the gripper links have none, so the ones here are
what Isaac Sim 6.1 derives for UWLab's robot (measured), including 1 kg on each collider-less outer knuckle. With
them the arm's mass matrix matches PhysX's to 0.01 %, and UWLab's 120 Hz explicit controller is stable.

The dynamics are UWLab's *implicit-actuator* training/eval configuration
(``IMPLICIT_UR5E_ROBOTIQ_2F85``): the arm is torque-driven by an operational-space controller
(kp = kd = 0 here; efforts capped at 150/28 N·m), the gripper's ``finger_joint`` is a PD servo
(kp 17, kd 5, 60 N·m) and the five passive linkage joints follow it through joint equalities.
Every link floats (``disable_gravity`` in the USD), which gravity compensation reproduces, and no
DOF carries armature or friction — the USD authors none and the stage-1 policy never had the
sysid values the finetune stage randomizes.
"""

from __future__ import annotations

from typing import ClassVar

from eden.options.catalog import ROBOT_REGISTRY
from eden.options.entities import RobotOptions
from eden.options.materials import RigidMaterialOptions

ARM_DOFS = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)
GRIPPER_DRIVER = "finger_joint"


@ROBOT_REGISTRY.register()
class OmniResetUR5e2f85(RobotOptions):
    """UR5e + Robotiq 2F-85 exactly as UWLab's OmniReset policies were trained on it."""

    file: str = "arms/omnireset_ur5e_2f85/omnireset_ur5e_2f85.xml"
    is_fixed_base: bool = True
    # The linkage's passive joints are slaved to finger_joint by equalities; only the driver is controlled.
    dofs_name: list[str] = [*ARM_DOFS, GRIPPER_DRIVER]
    default_dofs_pos: dict[str, float] = {
        "shoulder_pan_joint": 0.0,
        "shoulder_lift_joint": -1.5708,
        "elbow_joint": 1.5708,
        "wrist_1_joint": -1.5708,
        "wrist_2_joint": -1.5708,
        "wrist_3_joint": -1.5708,
        GRIPPER_DRIVER: 0.0,
    }
    default_dofs_kp: dict[str, float] = {**dict.fromkeys(ARM_DOFS, 0.0), GRIPPER_DRIVER: 17.0}
    default_dofs_kd: dict[str, float] = {**dict.fromkeys(ARM_DOFS, 0.0), GRIPPER_DRIVER: 5.0}
    default_dofs_force_limits: dict[str, float] = {
        "shoulder_pan_joint": 150.0,
        "shoulder_lift_joint": 150.0,
        "elbow_joint": 150.0,
        "wrist_1_joint": 28.0,
        "wrist_2_joint": 28.0,
        "wrist_3_joint": 28.0,
        GRIPPER_DRIVER: 60.0,
    }
    default_armature: float = 0.0
    material: RigidMaterialOptions = RigidMaterialOptions(gravity_compensation=1.0)

    ee_links_name: ClassVar[list[str]] = ["wrist_3_link"]
