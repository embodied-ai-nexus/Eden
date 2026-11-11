"""Actuator presets: the joint-side armature each derives, and the doubled Unitree linkage."""

from __future__ import annotations

import pytest

from eden.options.actuators import (
    DamiaoDM4310,
    DamiaoDM4340,
    LimxOliAnkleWaist,
    LimxOliHipKnee,
    LimxOliShoulderElbow,
    LimxOliWristHead,
    Unitree5020_16,
    Unitree5020_16_DOUBLE,
)


@pytest.mark.parametrize(
    "actuator, armature",
    [
        (DamiaoDM4310, 0.0018),  # 1.8e-5 rotor x gear 10^2
        (DamiaoDM4340, 0.032),  # 2e-5 rotor x gear 40^2
        # LimX specs state the identified joint-side value directly (gear ratio 1).
        (LimxOliHipKnee, 0.14125),
        (LimxOliAnkleWaist, 0.09488),
        (LimxOliShoulderElbow, 0.0886706),
        (LimxOliWristHead, 0.0153218),
        (Unitree5020_16, 3.609725e-3),  # two-stage planetary
    ],
)
def test_armature_is_the_rotor_inertia_reflected_to_the_joint(actuator, armature):
    assert actuator.ARMATURE() == pytest.approx(armature, rel=1e-9)


def test_double_5020_sums_two_motors_but_keeps_the_single_motor_speeds():
    """Two rotors on a 1:1 linkage: torque, friction and inertia add, speed limits do not."""
    single, double = Unitree5020_16, Unitree5020_16_DOUBLE
    for name in ("EFFORT_LIMIT", "DRIVING_TORQUE_LIMIT", "BRAKING_TORQUE_LIMIT", "STATIC_FRICTION", "DYNAMIC_FRICTION"):
        assert getattr(double, name) == 2 * getattr(single, name), name
    assert double.ARMATURE() == pytest.approx(2 * single.ARMATURE(), rel=1e-12)
    assert (double.FULL_TORQUE_SPEED, double.NO_LOAD_SPEED) == (single.FULL_TORQUE_SPEED, single.NO_LOAD_SPEED)
