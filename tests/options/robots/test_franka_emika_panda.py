"""The Franka arm assets are the robots their classes claim, fingerprinted against vendor data.

A hand-less Panda shipped as ``FrankaResearch3`` would pass silently: every name resolves
and every task builds, with masses up to 1.7x off and three joint limits an FR3 does not have. These
read the descriptors (no build); reference values are Franka's, via MuJoCo Menagerie.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest

from eden.options.robots import FrankaEmikaPanda, FrankaEmikaPandaNoHand, FrankaResearch3
from eden.utils.assets import get_asset_path

#: FR3 joint ranges [rad] (Menagerie ``franka_fr3``). A6 is the telling one: an FR3 cannot reach 0.
_FR3_RANGES = {
    "joint1": (-2.7437, 2.7437),
    "joint2": (-1.7837, 1.7837),
    "joint3": (-2.9007, 2.9007),
    "joint4": (-3.0421, -0.1518),
    "joint5": (-2.8065, 2.8065),
    "joint6": (0.5445, 4.5169),
    "joint7": (-3.0159, 3.0159),
}
#: Joint friction identified in torque control (Menagerie, 2025-04-25); the Panda model has none.
_FR3_ARMATURE = {**{f"joint{i}": 0.195 for i in range(1, 5)}, **{f"joint{i}": 0.074 for i in range(5, 8)}}
_FR3_FRICTIONLOSS = {"joint5": 0.763, "joint6": 0.44, "joint7": 0.248}
_POSE_DEFAULTS = (("pos", (0.0, 0.0, 0.0)), ("quat", (1.0, 0.0, 0.0, 0.0)))


def _root(options_type) -> ET.Element:
    return ET.parse(get_asset_path(options_type().file)).getroot()


def _joints(root: ET.Element) -> dict[str, ET.Element]:
    return {j.get("name"): j for j in root.iter("joint") if j.get("name")}


def _bodies(root: ET.Element) -> dict[str, ET.Element]:
    return {b.get("name"): b for b in root.iter("body") if b.get("name")}


def _floats(text: str | None, default: tuple[float, ...] = ()) -> tuple[float, ...]:
    """A pos/quat/range attribute, defaulted the way MuJoCo defaults an omitted one."""
    return default if text is None else tuple(float(v) for v in text.split())


def _link_mass(root, i) -> float:
    return float(_bodies(root)[f"link{i}"].find("inertial").get("mass"))


def test_the_fr3_asset_declares_the_fr3_joints_and_their_identified_friction():
    joints = _joints(_root(FrankaResearch3))
    assert set(joints) == set(_FR3_RANGES)
    for name, expected in _FR3_RANGES.items():
        assert _floats(joints[name].get("range")) == pytest.approx(expected), name
    for name, armature in _FR3_ARMATURE.items():
        declared = joints[name].get("armature")  # joints 1-4 inherit the class default, 5-7 override it
        assert float(declared) == pytest.approx(armature) if declared else armature == 0.195, name
    for name, friction in _FR3_FRICTIONLOSS.items():
        assert float(joints[name].get("frictionloss")) == pytest.approx(friction), name


def test_the_fr3_is_not_the_panda():
    """``arms/franka_research3/fr3.xml`` must not be Menagerie's ``panda_nohand.xml``."""
    fr3, panda = _root(FrankaResearch3), _root(FrankaEmikaPanda)
    assert fr3.get("model") == "fr3"
    fr3_joints, panda_joints = _joints(fr3), _joints(panda)
    differing = {n for n in _FR3_RANGES if fr3_joints[n].get("range") != panda_joints[n].get("range")}
    assert differing >= {"joint2", "joint4", "joint6", "joint7"}
    # Franka publishes different link masses for the two; link1 is the widest gap.
    assert (_link_mass(fr3, 1), _link_mass(panda, 1)) == (pytest.approx(2.92747, abs=1e-4), pytest.approx(4.970684))


def test_panda_no_hand_is_the_panda_minus_its_hand_not_a_second_arm():
    arm, panda = _root(FrankaEmikaPandaNoHand), _root(FrankaEmikaPanda)
    assert _bodies(arm).get("hand") is None and "attachment" in _bodies(arm)
    assert set(_joints(arm)) == {f"joint{i}" for i in range(1, 8)}
    assert arm.find("tendon") is None and arm.find("equality") is None  # both only coupled the fingers
    arm_bodies, panda_bodies = _bodies(arm), _bodies(panda)
    for i in range(8):
        name = f"link{i}"
        assert ET.tostring(arm_bodies[name].find("inertial")) == ET.tostring(panda_bodies[name].find("inertial"))
        for attribute, default in _POSE_DEFAULTS:
            assert _floats(arm_bodies[name].get(attribute), default) == pytest.approx(
                _floats(panda_bodies[name].get(attribute), default)
            ), f"{name}.{attribute}"
    assert _joints(arm)["joint6"].get("range") == _joints(panda)["joint6"].get("range")


@pytest.mark.parametrize("panda_type", [FrankaEmikaPanda, FrankaEmikaPandaNoHand])
def test_both_arms_share_one_kinematic_chain_and_mount_frame(panda_type):
    """What lets a rig swap one arm for the other without a per-arm correction."""
    fr3, panda = _bodies(_root(FrankaResearch3)), _bodies(_root(panda_type))
    for i in range(1, 8):
        for attribute, default in _POSE_DEFAULTS:
            assert _floats(fr3[f"link{i}"].get(attribute), default) == pytest.approx(
                _floats(panda[f"link{i}"].get(attribute), default)
            ), f"link{i}.{attribute}"
    assert _floats(fr3["attachment"].get("pos")) == pytest.approx((0.0, 0.0, 0.107))
    # 135 deg about z: the Panda's hand frame plus the 180 deg its standalone MJCF root carries.
    assert _floats(fr3["attachment"].get("quat")) == pytest.approx((0.3826834, 0.0, 0.0, 0.9238795))
    mount = panda["hand"] if "hand" in panda else panda["attachment"]
    assert _floats(mount.get("pos")) == pytest.approx((0.0, 0.0, 0.107))


@pytest.mark.parametrize("panda_type", [FrankaEmikaPanda, FrankaEmikaPandaNoHand])
def test_the_panda_loads_no_mesh_out_of_the_fr3_directory(panda_type):
    files = [mesh.get("file") for mesh in _root(panda_type).iter("mesh")]
    assert files and not [f for f in files if "franka_research3" in f]
