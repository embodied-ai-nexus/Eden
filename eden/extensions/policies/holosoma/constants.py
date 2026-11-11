"""Frozen constants for the Holosoma G1 29-DOF locomotion policy port.

All arrays are in **Unitree G1 29-DOF declaration order** (``G1_DOFS29_NAME``),
which is identical to the joint order Holosoma trained against
(``holosoma_inference.config.config_values.robot.g1_29dof.dof_names``), so no
DOF remapping is needed between the two.

Only the default pose is pinned here: the released ONNX ``kp`` / ``kd`` metadata
is bit-for-bit identical to ``G1Dofs29``'s Raibert-heuristic ``default_dofs_kp``
/ ``default_dofs_kd`` (Holosoma trained with the same heuristic), so
``holosoma_g1_robot`` inherits the gains rather than re-specifying them.
"""

from __future__ import annotations

# fmt: off

# Holosoma g1_29dof default joint angles (rad), G1Dofs29 order. Differs from
# G1Dofs29's default pose (deeper crouch), so it is overridden in holosoma_g1_robot.
G1_LOCO_DEFAULT_DOF_POS: tuple[float, ...] = (
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,   # left leg
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,   # right leg
    0.0, 0.0, 0.0,                          # waist (yaw, roll, pitch)
    0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0,      # left arm
    0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0,     # right arm
)

# fmt: on

NUM_DOFS: int = 29

# Observation layout: Holosoma flattens ``actor_obs`` terms in **sorted (alphabetical)**
# order (``BasePolicy._initialize_history_state`` does ``sorted(term_names)``), NOT the
# order they are listed in the config. The 100-dim ``actor_obs`` input is therefore:
#
#   actions(29) | base_ang_vel(3) | command_ang_vel(1) | command_lin_vel(2) |
#   cos_phase(2) | dof_pos(29) | dof_vel(29) | projected_gravity(3) | sin_phase(2)
#
# Per-term observation scales (loco_g1_29dof.obs_scales).
OBS_SCALE_BASE_ANG_VEL: float = 0.25
OBS_SCALE_DOF_VEL: float = 0.05
# (dof_pos, actions, commands, projected_gravity, sin/cos_phase all use scale 1.0)
