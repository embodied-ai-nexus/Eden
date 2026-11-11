"""The Inspire RH56 hands: the stock RH56 and the RH56DFQ of Weave's G1 rig, one joint set with two couplings.

The built DFQ hands are exercised through the composite in ``composite/test_g1_inspire.py``.
"""

from __future__ import annotations

import pytest

from eden.options.catalog import ROBOT_REGISTRY
from eden.options.entities import RobotOptions
from eden.options.robots import InspireRH56_L, InspireRH56_R, InspireRH56DFQ_L, InspireRH56DFQ_R

_ALL = [InspireRH56_L, InspireRH56_R, InspireRH56DFQ_L, InspireRH56DFQ_R]
_DFQ = [InspireRH56DFQ_L, InspireRH56DFQ_R]


@pytest.mark.parametrize("cls", _ALL)
def test_every_inspire_hand_names_the_same_twelve_joints_and_five_fingertips(cls):
    assert cls().dofs_name == InspireRH56_L().dofs_name and len(cls().dofs_name) == 12
    assert cls.metadata.target_hand_links == [
        "thumb_distal",
        "index_intermediate",
        "middle_intermediate",
        "ring_intermediate",
        "pinky_intermediate",
    ]
    assert cls.metadata.side == ("left" if cls.__name__.endswith("_L") else "right")


@pytest.mark.parametrize("cls", _ALL)
def test_every_joint_is_either_one_of_six_drivers_or_a_follower_of_one(cls):
    """Declared once on the shared base, the driver subset reaches each left/right leaf (it once read all twelve)."""
    drivers, mimic = set(cls.actuated_dofs_name), cls.metadata.mimic_dofs
    assert len(drivers) == 6 and len(mimic) == 6
    assert drivers.isdisjoint(mimic) and drivers | set(mimic) == set(cls().dofs_name)
    assert {driver for driver, _, _ in mimic.values()} <= drivers


def test_the_two_variants_couple_at_their_own_assets_ratios():
    def ratios(cls):
        mimic = cls.metadata.mimic_dofs
        return (
            mimic["thumb_intermediate_joint"][1:],
            mimic["thumb_distal_joint"][1:],
            mimic["index_intermediate_joint"][1:],
        )

    assert ratios(InspireRH56DFQ_L) == ((1.6, 0.0), (2.4, 0.0), (1.0, 0.0))
    assert ratios(InspireRH56_L) == ((1.334, 0.0), (0.667, 0.0), (1.06399, -0.04545))


@pytest.mark.parametrize("cls", _DFQ)
def test_the_dfq_gains_are_weaves_finger_group_on_every_joint(cls):
    """Kp 5 / kd 0.5 / armature 0.01 / effort 1 on all twelve, followers included (they are PD-driven too)."""
    hand = cls()
    assert (hand.default_dofs_kp, hand.default_dofs_kd) == ({"*": 5.0}, {"*": 0.5})
    assert hand.default_dofs_armature == {"*": 0.01}
    assert hand.default_dofs_force_limits == {"*": 1.0}


@pytest.mark.parametrize("cls", _ALL)
def test_each_hand_is_registered_under_its_class_name(cls):
    key = {"InspireRH56_L": "inspire_rh56_l", "InspireRH56_R": "inspire_rh56_r"}.get(cls.__name__)
    assert ROBOT_REGISTRY.get(key or f"inspire_rh56_dfq_{cls.__name__[-1].lower()}") is cls


def test_a_subclass_inherits_a_declared_actuated_subset_but_not_one_filled_for_it():
    """The fill-from-``dofs_name`` default applies only where no ancestor declared the subset."""

    class _Declared(RobotOptions):
        dofs_name: list[str] = ["a", "b", "c"]
        actuated_dofs_name = ["a"]

    class _Leaf(_Declared):
        pass

    class _Filled(RobotOptions):
        dofs_name: list[str] = ["a", "b"]

    class _Wider(_Filled):
        dofs_name: list[str] = ["a", "b", "c"]

    assert _Leaf.actuated_dofs_name == ["a"]
    assert _Filled.actuated_dofs_name == ["a", "b"]
    assert _Wider.actuated_dofs_name == ["a", "b", "c"]
