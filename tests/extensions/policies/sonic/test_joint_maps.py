"""SONIC's joint permutations and per-joint constants against ``policy_parameters.hpp`` and IsaacLab's G1 order."""

from __future__ import annotations

import pytest

from eden.extensions.deployment.base import resolve_dofs_values
from eden.extensions.policies.sonic.joint_maps import (
    ACTION_SCALE_MJ,
    DEFAULT_ANGLES_MJ,
    IL_TO_MJ,
    LOWER_BODY_IL_INDICES,
    MJ_TO_IL,
    MODE_G1,
    MODE_SMPL,
    MODE_TELEOP,
    NUM_MODES,
    SONIC_DOFS_KD,
    SONIC_DOFS_KP,
)
from eden.options.robots.unitree_g1 import G1Dofs29

MJ_NAMES = G1Dofs29().dofs_name


def test_permutations_map_each_joint_name_onto_itself(isaaclab_joint_names):
    """A wrong entry silently swaps two joints in the observation or the action."""
    il_names = isaaclab_joint_names
    assert [il_names[IL_TO_MJ[i]] for i in range(29)] == MJ_NAMES
    assert [MJ_NAMES[MJ_TO_IL[k]] for k in range(29)] == il_names
    assert list(LOWER_BODY_IL_INDICES) == [il_names.index(name) for name in MJ_NAMES[:12]]


def test_default_angles_match_policy_parameters_hpp():
    """Same values as ProtoMotionsV3's ``_PD_ACTION_OFFSET``; a few-mrad mismatch tips the stance."""
    leg = (-0.312, 0.0, 0.0, 0.669, -0.363, 0.0)
    left_arm = (0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0)
    right_arm = (0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0)
    assert DEFAULT_ANGLES_MJ == pytest.approx((*leg, *leg, 0.0, 0.0, 0.0, *left_arm, *right_arm), abs=1e-12)


def test_encoder_modes_match_the_cpp_enum():
    assert (MODE_G1, MODE_TELEOP, MODE_SMPL, NUM_MODES) == (0, 1, 2, 4)


# The upstream header derives every number from a motor table rather than tabulating them;
# re-derived here the same way from the header's own constants.
_CPP_OMEGA = 10 * 2.0 * 3.1415926535  # NATURAL_FREQ, with the header's truncated pi
_CPP_ZETA = 2.0  # DAMPING_RATIO
_CPP_ARMATURE = {"5020": 0.003609725, "7520_14": 0.010177520, "7520_22": 0.025101925, "4010": 0.00425}
_CPP_EFFORT = {"5020": 25.0, "7520_14": 88.0, "7520_22": 139.0, "4010": 5.0}
#: Motor per DOF in MJCF order, from the ``kps[]`` comments; ``x2`` rows are the doubled 4-bar linkages.
_CPP_MOTOR_MJ = (
    *("7520_22", "7520_22", "7520_14", "7520_22", "5020x2", "5020x2") * 2,
    *("7520_14", "5020x2", "5020x2"),
    *("5020", "5020", "5020", "5020", "5020", "4010", "4010") * 2,
)
#: The header's truncated pi lands at ~4e-9.
CPP_RTOL = 1e-7


def cpp_gains() -> tuple[list[float], list[float], list[float]]:
    """``(kps, kds, action_scales)`` in MJCF order, per ``policy_parameters.hpp``."""
    kps, kds, scales = [], [], []
    for motor in _CPP_MOTOR_MJ:
        key = motor.removesuffix("x2")
        # Doubled rows double stiffness and damping but NOT the action scale, which the header
        # computes off the single motor. Reproduced, not corrected: the checkpoint was trained with it.
        mult = 2.0 if motor.endswith("x2") else 1.0
        stiffness = _CPP_ARMATURE[key] * _CPP_OMEGA**2
        kps.append(mult * stiffness)
        kds.append(mult * 2.0 * _CPP_ZETA * _CPP_ARMATURE[key] * _CPP_OMEGA)
        scales.append(0.25 * _CPP_EFFORT[key] / stiffness)
    return kps, kds, scales


def test_action_scale_matches_the_cpp_formula():
    assert ACTION_SCALE_MJ == pytest.approx(cpp_gains()[2], rel=CPP_RTOL)


def test_gain_maps_match_the_cpp_formula_on_every_dof():
    """Resolved the way ``DeploymentBase`` resolves them, since that is what reaches the wire."""
    kp, kd, _ = cpp_gains()
    assert resolve_dofs_values(SONIC_DOFS_KP, MJ_NAMES) == pytest.approx(kp, rel=CPP_RTOL)
    assert resolve_dofs_values(SONIC_DOFS_KD, MJ_NAMES) == pytest.approx(kd, rel=CPP_RTOL)


def test_gain_maps_differ_from_the_stock_g1_in_hip_pitch_only():
    stock = G1Dofs29()
    for sonic_map, stock_map in ((SONIC_DOFS_KP, stock.default_dofs_kp), (SONIC_DOFS_KD, stock.default_dofs_kd)):
        assert set(sonic_map) == set(stock_map)
        assert {k for k in sonic_map if sonic_map[k] != stock_map[k]} == {"*_hip_pitch_joint"}
    # Built from a stock instance: an in-place update would have retuned every other G1 policy.
    assert stock.default_dofs_kp["*_hip_pitch_joint"] == pytest.approx(40.179239, rel=1e-6)
    assert stock.default_dofs_kd["*_hip_pitch_joint"] == pytest.approx(2.557890, rel=1e-6)


def test_lower_body_il_indices_first_12():
    """Lower-body indices in IL space must be the IL slot for each MJ leg dof."""
    assert len(LOWER_BODY_IL_INDICES) == 12
    for i in range(12):
        assert LOWER_BODY_IL_INDICES[i] == IL_TO_MJ[i]
