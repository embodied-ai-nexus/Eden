"""Actuator presets: the joint-side armature each derives, the doubled Unitree linkage, and the Wuji MIT units."""

from __future__ import annotations

import pytest

from eden.options.actuators import (
    WujiHand2Actuator,
    DamiaoDM4310,
    DamiaoDM4340,
    LimxOliAnkleWaist,
    LimxOliHipKnee,
    LimxOliShoulderElbow,
    LimxOliWristHead,
    Unitree5020_16,
    Unitree5020_16_DOUBLE,
    WujiHand2Finger,
    WujiHand2Flex,
    WujiHand2Pinky,
    WujiHand2PinkyFlex,
    WujiHand2ThumbAbd,
    WujiHand2ThumbIp,
    WujiHand2ThumbMcp,
)

_WUJI_HAND2 = [
    WujiHand2Flex,
    WujiHand2ThumbAbd,
    WujiHand2ThumbMcp,
    WujiHand2ThumbIp,
    WujiHand2Finger,
    WujiHand2PinkyFlex,
    WujiHand2Pinky,
]


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


@pytest.mark.parametrize("actuator", _WUJI_HAND2, ids=lambda a: a.__name__)
def test_a_wuji_hand2_motor_is_limited_by_its_current_and_converts_mit_gains_both_ways(actuator):
    assert actuator.EFFORT_LIMIT == pytest.approx(1.5 * actuator.TORQUE_CONSTANT, rel=1e-12)
    assert actuator.MIT_GAIN(actuator.JOINT_GAIN(3.0)) == pytest.approx(3.0, rel=1e-12)


def test_the_wuji_hand2_thumb_flexor_matches_upstream():
    """wuji-mjlab pen spin: kt 0.4136 N*m/A, 1.9115 A per MIT unit, so MIT kp 3.0 is 2.3718 N*m/rad."""
    assert WujiHand2Flex.JOINT_GAIN(3.0) == pytest.approx(2.371753, rel=1e-6)
    assert WujiHand2Flex.EFFORT_LIMIT == pytest.approx(0.6204, rel=1e-12)


def test_a_wuji_spec_without_a_torque_constant_refuses_to_convert_and_names_the_motors():
    """The base class has none: converting through it would program 0 or inf on the firmware."""
    for convert in (WujiHand2Actuator.JOINT_GAIN, WujiHand2Actuator.MIT_GAIN):
        with pytest.raises(ValueError, match=r"WujiHand2Actuator\.TORQUE_CONSTANT is 0\.0.*WujiHand2Flex"):
            convert(3.0)
