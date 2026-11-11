"""Unitree Go2 and Go2-W quadruped configurations."""

from __future__ import annotations
from typing import ClassVar

from genesis.typing import Vec3FType

from eden.options.catalog import ROBOT_REGISTRY
from eden.options.entities import RobotOptions
from eden.options.actuators import ActuatorSpecOptions, UnitreeGo2HV, UnitreeGo2HVCalf


__all__ = ["Go2", "GO2_ACTION_SCALE", "Go2W", "GO2W_ACTION_SCALE"]


_Go2_DOFS_SPEC: dict[str, type[ActuatorSpecOptions]] = {
    "*_hip_joint": UnitreeGo2HV,
    "*_thigh_joint": UnitreeGo2HV,
    "*_calf_joint": UnitreeGo2HVCalf,
}


# The four feet: what the robot stands on, and what Genesis must not merge away.
GO2_FEET_LINKS_NAME: tuple[str, ...] = ("FL_foot", "RL_foot", "FR_foot", "RR_foot")


@ROBOT_REGISTRY.register()
class Go2(RobotOptions):
    file: str = "quadrupeds/unitree_go2/go2.urdf"

    dofs_spec: ClassVar[dict[str, type[ActuatorSpecOptions]]] = _Go2_DOFS_SPEC

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
        "FL_calf_joint": -1.5,  # [rad]
        "RL_calf_joint": -1.5,  # [rad]
        "FR_calf_joint": -1.5,  # [rad]
        "RR_calf_joint": -1.5,  # [rad]
    }
    # Raibert helpers below are evaluated at class-definition time using the
    # default ``natural_freq=NATURAL_FREQ`` (10 Hz). If you tune NATURAL_FREQ
    # globally these dicts won't follow — rebuild them on the subclass.
    default_dofs_kp: dict[str, float] = {p: s.RAIBERT_HEURISTIC_KP() for p, s in _Go2_DOFS_SPEC.items()}
    default_dofs_kd: dict[str, float] = {p: s.RAIBERT_HEURISTIC_KD() for p, s in _Go2_DOFS_SPEC.items()}
    default_dofs_armature: dict[str, float] = {p: s.ARMATURE() for p, s in _Go2_DOFS_SPEC.items()}
    default_dofs_force_limits: dict[str, float] = {p: s.EFFORT_LIMIT for p, s in _Go2_DOFS_SPEC.items()}

    links_to_keep: list[str] = list(GO2_FEET_LINKS_NAME)
    feet_links_name: ClassVar[tuple[str, ...]] = GO2_FEET_LINKS_NAME


# Per-joint action scale = ``0.25 * EFFORT_LIMIT / kp``; same source-of-truth
# spec dict, same Raibert heuristic. Use this for ``ActionTerm.configure(scale=...)``.
GO2_ACTION_SCALE: dict[str, float] = {p: s.RAIBERT_HEURISTIC_ACTION_SCALE() for p, s in _Go2_DOFS_SPEC.items()}


_Go2W_DOFS_SPEC: dict[str, type[ActuatorSpecOptions]] = {
    **_Go2_DOFS_SPEC,
    "*_foot_joint": UnitreeGo2HV,
}


@ROBOT_REGISTRY.register()
class Go2W(Go2):
    file: str = "quadrupeds/unitree_go2w/go2w.urdf"

    # No feet, deliberately — the wheeled variant drives ``FL_foot``…``RR_foot`` through the
    # actuated ``*_foot_joint`` below, so those links are wheel hubs rather than the static
    # feet a leg-odometry estimator assumes. The contact point of a rolling wheel is
    # stationary while the hub translates, which is the opposite of what the deploy-time
    # velocity filter anchors on, and there is no Go2W to validate an estimate against. Left
    # empty so ``--estimate_lin_vel`` refuses rather than returning a quietly biased number;
    # ``--feet_links`` is there for anyone who does have the hardware to check it on.
    feet_links_name: ClassVar[tuple[str, ...]] = ()

    dofs_spec: ClassVar[dict[str, type[ActuatorSpecOptions]]] = _Go2W_DOFS_SPEC

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
        "FL_foot_joint",
        "RL_foot_joint",
        "FR_foot_joint",
        "RR_foot_joint",
    ]

    default_root_pos: Vec3FType = (0.0, 0.0, 0.46)  # x,y,z [m]
    default_dofs_pos: dict[str, float] = {  # = target angles [rad] when action = 0.0
        "FL_hip_joint": 0.0,  # [rad]
        "RL_hip_joint": 0.0,  # [rad]
        "FR_hip_joint": 0.0,  # [rad]
        "RR_hip_joint": 0.0,  # [rad]
        "FL_thigh_joint": 0.67,  # [rad]
        "RL_thigh_joint": 0.67,  # [rad]
        "FR_thigh_joint": 0.67,  # [rad]
        "RR_thigh_joint": 0.67,  # [rad]
        "FL_calf_joint": -1.3,  # [rad]
        "RL_calf_joint": -1.3,  # [rad]
        "FR_calf_joint": -1.3,  # [rad]
        "RR_calf_joint": -1.3,  # [rad]
        "FL_foot_joint": 0.0,
        "RL_foot_joint": 0.0,
        "FR_foot_joint": 0.0,
        "RR_foot_joint": 0.0,
    }
    default_dofs_kp: dict[str, float] = {p: s.RAIBERT_HEURISTIC_KP() for p, s in _Go2W_DOFS_SPEC.items()}
    default_dofs_kd: dict[str, float] = {p: s.RAIBERT_HEURISTIC_KD() for p, s in _Go2W_DOFS_SPEC.items()}
    default_dofs_armature: dict[str, float] = {p: s.ARMATURE() for p, s in _Go2W_DOFS_SPEC.items()}
    default_dofs_force_limits: dict[str, float] = {p: s.EFFORT_LIMIT for p, s in _Go2W_DOFS_SPEC.items()}


GO2W_ACTION_SCALE: dict[str, float] = {p: s.RAIBERT_HEURISTIC_ACTION_SCALE() for p, s in _Go2W_DOFS_SPEC.items()}
