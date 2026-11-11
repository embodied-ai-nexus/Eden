"""Wuji Hand2 beta2 (both sides, bare and mounted) built: gains, armature, axis convention, and the mount.

The URDF carries no armature or damping, so without the class overrides every joint would take Genesis'
0.1 morph default, ~200x the manufacturer's 5e-4 rotor inertia, and the fingers would barely move.
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


@pytest.mark.parametrize("name", _HANDS)
def test_gains_effort_and_armature_reach_the_solver(env, name):
    hand = env.entities[name]
    for got, value in (
        (hand.get_dofs_kp(), 50.0),
        (hand.get_dofs_kd(), 5.0),
        (hand.get_dofs_force_range()[1], 5.0),
        (-hand.get_dofs_force_range()[0], 5.0),
        (hand.get_dofs_armature(), 5.0e-4),
    ):
        assert torch.allclose(got, torch.full_like(got, value), atol=1e-6)


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
