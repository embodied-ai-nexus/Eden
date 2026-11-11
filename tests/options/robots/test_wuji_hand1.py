"""Wuji Hand 1 (both sides, both collision flavours) built: the manufacturer actuator table and the axis convention.

The URDF carries no armature or damping and the hand runs at kp ~0.2-0.7 N m/rad, so both a missing
override (Genesis' 0.1 armature, 200-500x too high) and a blanket gain would build fine and not grasp.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from eden.options.robots import WujiHand1_L, WujiHand1_R, WujiHand1Simplified_L, WujiHand1Simplified_R
from tests.options.robots.conftest import build_scene, dofs_in_declared_order, links_pos, set_dofs

pytestmark = pytest.mark.requires_genesis

# Fingertip frames in finger order; finger1 is the thumb.
_TIPS = [f"finger{f}_tip_link" for f in range(1, 6)]
_HANDS = {
    "right": (WujiHand1_R, "right"),
    "left": (WujiHand1_L, "left"),
    "right_simplified": (WujiHand1Simplified_R, "right"),
    "left_simplified": (WujiHand1Simplified_L, "left"),
}

# The manufacturer system-identification table (upstream ``hand/body-with-soft/params.csv``), in
# declared DOF order (finger1..5 x joint1..4), transcribed independently of the config's regex keys.
_KP = [
    0.408447, 0.68586, 0.239111, 0.207361, 0.373522, 0.455924, 0.243684, 0.18027, 0.368709, 0.416425,
    0.222186, 0.194276, 0.357182, 0.429773, 0.249302, 0.228503, 0.365533, 0.413931, 0.227294, 0.196472,
]  # fmt: skip
_KD = [
    0.020882, 0.0306109, 0.0101815, 0.00909699, 0.0188227, 0.0197982, 0.010477, 0.00824021, 0.0184862,
    0.0180329, 0.0095922, 0.00915299, 0.0183766, 0.018677, 0.0105951, 0.0099176, 0.018617, 0.0187322,
    0.00951005, 0.0090173,
]  # fmt: skip
_EFFORT = [
    0.4452, 0.4259, 0.1888, 0.1468, 0.6188, 0.1822, 0.2251, 0.217, 0.6494, 0.1827, 0.2078, 0.2018, 0.6389,
    0.1832, 0.2249, 0.2044, 0.6441, 0.1798, 0.2384, 0.1866,
]  # fmt: skip
_ARMATURE = [
    0.0005, 0.0005, 0.0002, 0.0002, 0.0005, 0.0002, 0.0002, 0.0002, 0.0005, 0.0002, 0.0002, 0.0002, 0.0005,
    0.0002, 0.0002, 0.0002, 0.0005, 0.0002, 0.0002, 0.0002,
]  # fmt: skip


@pytest.fixture(scope="module")
def _env():
    """All four hands, fixed-base: every use welds or fixes this hand, so its DOFs are 0..19."""
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


def _declared(name) -> list[str]:
    return _HANDS[name][0].model_fields["dofs_name"].default


def _per_declared_dof(hand, name, values) -> torch.Tensor:
    """``values`` (declared DOF order) laid out in the getters' ``dofs_name`` order."""
    by_name = dict(zip(_declared(name), values))
    return torch.tensor([by_name[dof] for dof in hand.dofs_name])


@pytest.mark.parametrize("name", _HANDS)
def test_every_declared_joint_and_kept_link_resolves(env, name):
    hand, prefix = env.entities[name], _HANDS[name][1]
    assert hand.num_dofs == 20
    assert dofs_in_declared_order(hand, _declared(name)) == _declared(name)
    # The empty fingertip frames and the thumb's silicone pad survive fixed-link merging (links_to_keep).
    kept = [f"{prefix}_{tip}" for tip in _TIPS] + [f"{prefix}_finger1_link2_softbody"]
    assert set(hand.find_named_links_idx_local(kept, preserve_order=True)[0]) == set(kept)


@pytest.mark.parametrize("name", _HANDS)
def test_the_per_joint_actuator_table_reaches_the_solver(env, name):
    """A whole-vector comparison, so a mis-scoped regex leaving some DOFs on the sim default fails.

    Under Genesis' default implicit integrator ``post_build`` does not fold ``kd * substep_dt`` into the
    armature, so the configured value arrives unchanged.
    """
    hand = env.entities[name]
    for got, table in (
        (hand.get_dofs_kp(), _KP),
        (hand.get_dofs_kd(), _KD),
        (hand.get_dofs_force_range()[1], _EFFORT),
        (-hand.get_dofs_force_range()[0], _EFFORT),
        (hand.get_dofs_armature(), _ARMATURE),
    ):
        torch.testing.assert_close(got.reshape(-1).cpu(), _per_declared_dof(hand, name, table), atol=1e-6, rtol=0)


def _unit(vec):
    return vec / np.linalg.norm(vec)


@pytest.mark.parametrize("name", _HANDS)
def test_the_urdf_frame_already_is_edens_fingers_up_palm_forward(env, name):
    """Why neither side declares an ``up``/``front`` override: the left URDF mirrors about XZ, keeping +X.

    Measured from the fully extended pose, since the manufacturer keyframe is a relaxed grasp whose
    fingertips sit as far along +X as along +Z.
    """
    hand, prefix = env.entities[name], _HANDS[name][1]
    frames = [f"{prefix}_palm_link"] + [f"{prefix}_{tip}" for tip in _TIPS]
    lower = hand.get_dofs_limit()[0][0]
    set_dofs(hand, {}, base=lower)
    extended = links_pos(hand, frames)
    finger_axis = _unit(extended[1:].mean(axis=0) - extended[0])
    assert finger_axis[2] > 0.85 and abs(finger_axis[2]) == pytest.approx(np.abs(finger_axis).max())

    # Flexing the four non-thumb fingers (the thumb is opposed) drives their tips along the palm normal.
    set_dofs(hand, {f"{prefix}_finger{f}_joint{j}": 0.6 for f in range(2, 6) for j in (1, 3, 4)}, base=lower)
    assert _unit((links_pos(hand, frames)[2:] - extended[2:]).mean(axis=0))[0] > 0.9


@pytest.mark.parametrize("side", ["right", "left"])
def test_the_simplified_flavour_is_the_same_hand_with_cheaper_collision(env, side):
    """Same links, joints and inertials, so a policy trained on one is kinematically valid on the other."""
    full, simplified = env.entities[side], env.entities[f"{side}_simplified"]
    assert simplified.base_link_name == full.base_link_name == f"{side}_palm_link"
    frames = [f"{side}_palm_link"] + [f"{side}_{tip}" for tip in _TIPS]
    full_pos, simplified_pos = links_pos(full, frames), links_pos(simplified, frames)
    np.testing.assert_allclose(simplified_pos[1:] - simplified_pos[0], full_pos[1:] - full_pos[0], atol=1e-4)
