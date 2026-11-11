"""Unitree Go1 quadruped configuration."""

from __future__ import annotations
from typing import ClassVar

from genesis.typing import Vec3FType

from eden.options.catalog import ROBOT_REGISTRY
from eden.options.entities import RobotOptions
from eden.options.actuators import ActuatorSpecOptions, Unitree8010_6, Unitree8010_6Calf


__all__ = ["Go1", "GO1_ACTION_SCALE"]


_Go1_DOFS_SPEC: dict[str, type[ActuatorSpecOptions]] = {
    "*_hip_joint": Unitree8010_6,
    "*_thigh_joint": Unitree8010_6,
    "*_calf_joint": Unitree8010_6Calf,
}


# The four feet: what the robot stands on, and what Genesis must not merge away.
GO1_FEET_LINKS_NAME: tuple[str, ...] = ("FL_foot", "RL_foot", "FR_foot", "RR_foot")


@ROBOT_REGISTRY.register()
class Go1(RobotOptions):
    file: str = "quadrupeds/unitree_go1/go1.urdf"

    dofs_spec: ClassVar[dict[str, type[ActuatorSpecOptions]]] = _Go1_DOFS_SPEC

    dofs_name: list[str] = [
        "FL_hip_joint",
        "RL_hip_joint",
        "FR_hip_joint",
        "RR_hip_joint",
        "FL_thigh_joint",
        "RL_thigh_joint",
        "FR_thigh_joint",
        "RR_thigh_joint",
        "FL_calf_joint",
        "RL_calf_joint",
        "FR_calf_joint",
        "RR_calf_joint",
    ]

    default_root_pos: Vec3FType = (0.0, 0.0, 0.36)  # x,y,z [m]
    default_dofs_pos: dict[str, float] = {  # = target angles [rad] when action = 0.0
        "FL_hip_joint": 0.0,  # [rad]
        "RL_hip_joint": 0.0,  # [rad]
        "FR_hip_joint": -0.0,  # [rad]
        "RR_hip_joint": -0.0,  # [rad]
        "FL_thigh_joint": 0.8,  # [rad]
        "RL_thigh_joint": 1.0,  # [rad]
        "FR_thigh_joint": 0.8,  # [rad]
        "RR_thigh_joint": 1.0,  # [rad]
        "*_calf_joint": -1.5,  # [rad]
    }
    # Raibert helpers below are evaluated at class-definition time using the
    # default ``natural_freq=NATURAL_FREQ`` (10 Hz). If you tune NATURAL_FREQ
    # globally these dicts won't follow — rebuild them on the subclass.
    default_dofs_kp: dict[str, float] = {p: s.RAIBERT_HEURISTIC_KP() for p, s in _Go1_DOFS_SPEC.items()}
    default_dofs_kd: dict[str, float] = {p: s.RAIBERT_HEURISTIC_KD() for p, s in _Go1_DOFS_SPEC.items()}
    default_dofs_armature: dict[str, float] = {p: s.ARMATURE() for p, s in _Go1_DOFS_SPEC.items()}
    default_dofs_force_limits: dict[str, float] = {p: s.EFFORT_LIMIT for p, s in _Go1_DOFS_SPEC.items()}

    links_to_keep: list[str] = list(GO1_FEET_LINKS_NAME)
    feet_links_name: ClassVar[tuple[str, ...]] = GO1_FEET_LINKS_NAME


# Per-joint action scale = ``0.25 * EFFORT_LIMIT / kp``; same source-of-truth
# spec dict, same Raibert heuristic. Use this for ``ActionTerm.configure(scale=...)``.
GO1_ACTION_SCALE: dict[str, float] = {p: s.RAIBERT_HEURISTIC_ACTION_SCALE() for p, s in _Go1_DOFS_SPEC.items()}
