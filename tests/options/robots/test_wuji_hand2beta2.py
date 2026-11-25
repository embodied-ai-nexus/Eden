"""Wuji Hand2 beta2 (both sides, bare and mounted) built: actuator calibration, axis convention, and the mount."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from eden.options.robots import (
    WujiHand2Beta2_L,
    WujiHand2Beta2_R,
    WujiHand2Beta2WithMount_L,
    WujiHand2Beta2WithMount_R,
)
from tests.options.robots.conftest import build_scene, dofs_in_declared_order, links_pos, set_dofs

pytestmark = pytest.mark.requires_genesis

_TIPS = ["thumb_tip", "index_finger_tip", "middle_finger_tip", "ring_finger_tip", "pinky_tip"]
_HANDS = {
    "r": (WujiHand2Beta2_R, "r"),
    "l": (WujiHand2Beta2_L, "l"),
    "r_mounted": (WujiHand2Beta2WithMount_R, "r"),
    "l_mounted": (WujiHand2Beta2WithMount_L, "l"),
}


@pytest.fixture(scope="module")
def _env():
    return build_scene(
        **{
            name: cls(is_fixed_base=True, default_root_pos=(0.5 * i, 0.0, 0.3))
            for i, (name, (cls, _)) in enumerate(_HANDS.items())
        }
    )


@pytest.fixture
def env(_env):
    yield _env
    _env.reset()


@pytest.mark.parametrize("name", _HANDS)
def test_every_declared_joint_and_tip_link_resolves(env, name):
    hand, prefix = env.entities[name], _HANDS[name][1]
    declared = _HANDS[name][0].model_fields["dofs_name"].default
    assert hand.num_dofs == 20
    assert dofs_in_declared_order(hand, declared) == declared
    tips = [f"{prefix}_{t}" for t in _TIPS]
    assert set(hand.find_named_links_idx_local(tips, preserve_order=True)[0]) == set(tips)


# wuji-mjlab v2026.10.22 ``pen_spin/config/wuji_hand2/hand2_constants.py``, in ``dofs_name`` order.
_TORQUE_CONSTANT = (
    [0.4136, 0.3545, 0.2216, 0.1329] + [0.4136, 0.1673, 0.1673, 0.1673] * 3 + [0.4461, 0.1115, 0.1115, 0.1115]
)
_MIT_KD = [0.075, 0.07, 0.05, 0.05] * 5
_JOINT_GAIN = [kt * 1.9114709702965849 for kt in _TORQUE_CONSTANT]
# Upstream's per-joint values. Armature and friction loss are averaged over the joints of each of Wuji's seven
# motor models: thumb CMC flexion with index / middle / ring MCP flexion; index / middle / ring abduction, PIP and
# DIP; little-finger abduction, PIP and DIP; and thumb CMC abduction, thumb MCP, thumb IP and little-finger MCP
# flexion, one joint each. Damping is per joint, as upstream identified it.
_MOTOR_JOINTS = [[0, 4, 8, 12], [5, 6, 7, 9, 10, 11, 13, 14, 15], [17, 18, 19], [1], [2], [3], [16]]
_PER_JOINT = {
    "armature": [
        1.457162e-4, 1.366389e-4, 1.273553e-4, 8.768902e-5, 1.130474e-4, 5.127161e-5, 4.843783e-5,
        7.906193e-5, 2.011048e-4, 7.106655e-5, 6.054746e-5, 5.659272e-5, 1.393897e-4, 3.544108e-5,
        5.233974e-5, 6.047948e-5, 9.706259e-5, 3.826054e-5, 4.664456e-5, 4.743778e-5,
    ],
    "damping": [
        0.0026915050, 0.0042812198, 0.0014135711, 0.0005131806, 0.0010510923, 0.0, 0.0010302809,
        0.0000663526, 0.0021587672, 0.0008842859, 0.0012610488, 0.0001128493, 0.0014506542, 0.0,
        0.0007149635, 0.0, 0.0, 0.0, 0.0016246495, 0.0,
    ],
    "frictionloss": [
        0.0367355430, 0.0342266392, 0.0187921182, 0.0122545518, 0.0377084683, 0.0177932607, 0.0297927367,
        0.0161822467, 0.0355644840, 0.0240616629, 0.0127755677, 0.0109169800, 0.0528257763, 0.0266927477,
        0.0148677454, 0.0215947297, 0.0338251954, 0.0261331247, 0.0142008972, 0.0129471168,
    ],
}  # fmt: skip


def _motor_mean(values: list[float]) -> list[float]:
    out = [0.0] * len(values)
    for joints in _MOTOR_JOINTS:
        mean = sum(values[j] for j in joints) / len(joints)
        for j in joints:
            out[j] = mean
    return out


_EXPECTED = {
    "kp": [3.0 * g for g in _JOINT_GAIN],
    "kd": [kd * g for kd, g in zip(_MIT_KD, _JOINT_GAIN)],
    "force_limits": [1.5 * kt for kt in _TORQUE_CONSTANT],
    "armature": _motor_mean(_PER_JOINT["armature"]),
    "frictionloss": _motor_mean(_PER_JOINT["frictionloss"]),
    "damping": _PER_JOINT["damping"],
}


@pytest.mark.parametrize("name", _HANDS)
def test_the_calibration_reaches_the_solver_joint_by_joint(env, name):
    """Armature and friction loss come from the joint's motor spec, damping from the robot's per-joint table."""
    hand = env.entities[name]
    assert not any(getattr(_HANDS[name][0](), f"default_dofs_{f}") for f in ("armature", "frictionloss"))
    lower, upper = hand.get_dofs_force_range()
    for field, got in (
        ("kp", hand.get_dofs_kp()),
        ("kd", hand.get_dofs_kd()),
        ("force_limits", upper),
        ("force_limits", -lower),
        ("armature", hand.get_dofs_armature()),
        ("damping", hand.get_dofs_damping()),
        ("frictionloss", hand.get_dofs_frictionloss()),
    ):
        got = got.reshape(-1, hand.num_dofs)[0].cpu().double()
        torch.testing.assert_close(got, torch.tensor(_EXPECTED[field]).double(), atol=1e-7, rtol=1e-4, msg=field)
    for field in ("armature", "damping", "frictionloss"):
        reported = getattr(hand, f"default_dofs_{field}").reshape(-1, hand.num_dofs)[0].cpu().double()
        simulated = getattr(hand, f"get_dofs_{field}")().reshape(-1, hand.num_dofs)[0].cpu().double()
        torch.testing.assert_close(reported, simulated, msg=f"default_dofs_{field} reports what the solver runs")


def _unit(vec):
    return vec / np.linalg.norm(vec)


@pytest.mark.parametrize("name", _HANDS)
def test_up_and_front_load_the_hand_fingers_up_palm_forward(env, name):
    """``up``/``front`` map the URDF finger axis (-Z) to world +Z and the palm normal (+/-Y) to world +X."""
    hand, prefix = env.entities[name], _HANDS[name][1]
    frames = [f"{prefix}_wrist"] + [f"{prefix}_{t}" for t in _TIPS]
    set_dofs(hand, {})
    rest = links_pos(hand, frames)
    assert _unit(rest[1:].mean(axis=0) - rest[0])[2] > 0.9

    set_dofs(
        hand, {f"{prefix}_{f}_{j}": 0.5 for f in ("index_finger", "middle_finger") for j in ("mcp_flex", "pip", "dip")}
    )
    assert _unit((links_pos(hand, frames)[1:] - rest[1:]).mean(axis=0))[0] > 0.5


@pytest.mark.parametrize("prefix", ["r", "l"])
def test_the_mounted_variant_is_the_same_hand_on_a_new_root_link(env, prefix):
    """The flange becomes the base link (what the root pose getters report); the hand does not move on the wrist."""
    bare, mounted = env.entities[prefix], env.entities[f"{prefix}_mounted"]
    assert (bare.base_link_name, mounted.base_link_name) == (f"{prefix}_wrist", f"{prefix}_mount")
    frames = [f"{prefix}_wrist"] + [f"{prefix}_{t}" for t in _TIPS]
    bare_pos, mounted_pos = links_pos(bare, frames), links_pos(mounted, frames)
    np.testing.assert_allclose(mounted_pos[1:] - mounted_pos[0], bare_pos[1:] - bare_pos[0], atol=1e-4)
