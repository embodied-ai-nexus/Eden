"""Wuji Hand2 beta2 (both sides, bare and mounted) built: actuator calibration, axis convention, and the mount.

The URDF carries no armature, damping or friction loss, so every one of those, and the gains, comes from the
class's identified calibration (wuji-mjlab's pen-spin Hand 2 table). Without it every joint would take Genesis'
0.1 morph armature, 500 to 2800 times the identified rotor inertia.
"""

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


def _table(options, field: str) -> torch.Tensor:
    """``default_dofs_<field>`` looked up by joint name, in ``dofs_name`` order."""
    table = getattr(options, f"default_dofs_{field}")
    return torch.tensor([table[f".*_{dof.split('_', 1)[1]}"] for dof in options.dofs_name])


@pytest.mark.parametrize("name", _HANDS)
def test_the_calibration_reaches_the_solver_joint_by_joint(env, name):
    """Full-width ``get_dofs_*`` reads are in ``dofs_name`` order, the order ``_table`` builds."""
    hand, cls = env.entities[name], _HANDS[name][0]
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
        torch.testing.assert_close(got, _table(cls(), field).double(), atol=1e-7, rtol=1e-5, msg=field)


@pytest.mark.parametrize(
    "dof, field, value",
    [
        # MIT kp 3.0 * 1.9115 A/unit * kt 0.4136 N*m/A
        ("thumb_cmc_flex", "kp", 2.371753),
        # MIT kd 0.05 on a distal joint * 1.9115 * kt 0.1115
        ("pinky_dip", "kd", 0.01065645),
        # 1.5 A * kt 0.4461
        ("pinky_mcp_flex", "force_limits", 0.66915),
        ("middle_finger_mcp_flex", "armature", 2.011048e-4),
        ("thumb_cmc_abd", "damping", 0.0042812198),
        ("index_finger_mcp_abd", "damping", 0.0),
        ("ring_finger_mcp_flex", "frictionloss", 0.0528257763),
    ],
)
def test_the_table_matches_the_upstream_pen_spin_calibration(dof, field, value):
    """Spot values from wuji-mjlab v2026.10.22 ``pen_spin/config/wuji_hand2/hand2_constants.py``."""
    table = getattr(WujiHand2Beta2_R(), f"default_dofs_{field}")
    assert table[f".*_{dof}"] == pytest.approx(value, rel=1e-5)


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
