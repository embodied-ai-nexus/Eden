"""Wuji Hand 2 hand2_2 (both sides) built: wuji-sim's joint dynamics, the axis convention, and the mount root."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from eden.options.robots import WujiHand2_2_L, WujiHand2_2_R, WujiHand2Beta2WithMount_R
from tests.options.robots.conftest import build_scene, dofs_in_declared_order, links_pos, set_dofs

pytestmark = pytest.mark.requires_genesis

_TIPS = ["thumb_tip", "index_finger_tip", "middle_finger_tip", "ring_finger_tip", "pinky_tip"]
_HANDS = {"r": WujiHand2_2_R, "l": WujiHand2_2_L}


@pytest.fixture(scope="module")
def _env():
    return build_scene(
        r=WujiHand2_2_R(is_fixed_base=True, default_root_pos=(0.0, 0.0, 0.3)),
        l=WujiHand2_2_L(is_fixed_base=True, default_root_pos=(0.5, 0.0, 0.3)),
        beta2_r=WujiHand2Beta2WithMount_R(is_fixed_base=True, default_root_pos=(1.0, 0.0, 0.3)),
    )


@pytest.fixture
def env(_env):
    yield _env
    _env.reset()


@pytest.mark.parametrize("prefix", _HANDS)
def test_every_declared_joint_and_tip_link_resolves_on_the_mount_root(env, prefix):
    hand = env.entities[prefix]
    declared = _HANDS[prefix].model_fields["dofs_name"].default
    assert hand.num_dofs == 20
    assert dofs_in_declared_order(hand, declared) == declared
    assert declared == [name.replace("r_", f"{prefix}_", 1) for name in WujiHand2Beta2WithMount_R().dofs_name]
    assert hand.base_link_name == f"{prefix}_mount"
    kept = [f"{prefix}_wrist", f"{prefix}_wrist_imu_frame"] + [f"{prefix}_{t}" for t in _TIPS]
    assert set(hand.find_named_links_idx_local(kept, preserve_order=True)[0]) == set(kept)


# wuji-sim v2026.11.23 ``wuji_sim/model_overrides.toml``, in ``dofs_name`` order: ``joint_armature`` and
# ``joint_torque_limit`` ("1.5 A current limit x torque constant", given to 1e-3 N*m).
_WUJI_SIM = {
    "armature": [
        1.457162066834878e-4, 1.3663892981144644e-4, 1.2735528418833324e-4, 8.768902139971928e-5,
        1.1304740105532539e-4, 5.1271605241470905e-5, 4.843782959990254e-5, 7.906192627320932e-5,
        2.0110478286346846e-4, 7.106655434321573e-5, 6.05474605842666e-5, 5.659271669130923e-5,
        1.3938973043403448e-4, 3.544107855351169e-5, 5.233973575843903e-5, 6.0479482928204684e-5,
        9.70625901007757e-5, 3.8260542056944675e-5, 4.664456024117704e-5, 4.74377771614785e-5,
    ],
    "force_limits": [
        0.620, 0.532, 0.332, 0.200, 0.620, 0.251, 0.251, 0.251, 0.620, 0.251, 0.251, 0.251,
        0.620, 0.251, 0.251, 0.251, 0.669, 0.167, 0.167, 0.167,
    ],
}  # fmt: skip
# wuji-sim's pen-spin example (``examples/_shared/pen_spin.py``): MIT kp 3.0 and kd by joint position, through
# 1.91147 A per MIT unit and each joint's torque constant.
_TORQUE_CONSTANT = (
    [0.4136, 0.3545, 0.2216, 0.1329] + [0.4136, 0.1673, 0.1673, 0.1673] * 3 + [0.4461, 0.1115, 0.1115, 0.1115]
)
_JOINT_GAIN = [kt * 1.9114709702965849 for kt in _TORQUE_CONSTANT]
_MIT_KD = [0.075, 0.07, 0.05, 0.05] * 5


@pytest.mark.parametrize("prefix", _HANDS)
def test_the_solver_runs_wuji_sims_joint_dynamics(env, prefix):
    """Per-joint armature, wuji-sim's torque limits, zero friction loss and damping, and the pen-spin gains."""
    hand = env.entities[prefix]
    lower, upper = hand.get_dofs_force_range()
    expected = {
        "kp": [3.0 * g for g in _JOINT_GAIN],
        "kd": [kd * g for kd, g in zip(_MIT_KD, _JOINT_GAIN)],
        "armature": _WUJI_SIM["armature"],
        "damping": [0.0] * 20,
        "frictionloss": [0.0] * 20,
    }
    for field, got in (
        ("kp", hand.get_dofs_kp()),
        ("kd", hand.get_dofs_kd()),
        ("armature", hand.get_dofs_armature()),
        ("damping", hand.get_dofs_damping()),
        ("frictionloss", hand.get_dofs_frictionloss()),
    ):
        got = got.reshape(-1, hand.num_dofs)[0].cpu().double()
        torch.testing.assert_close(got, torch.tensor(expected[field]).double(), atol=1e-9, rtol=1e-5, msg=field)
    for bound in (upper, -lower):
        got = bound.reshape(-1, hand.num_dofs)[0].cpu().double()
        torch.testing.assert_close(got, 1.5 * torch.tensor(_TORQUE_CONSTANT).double())
        # Within wuji-sim's rounding, except thumb IP: wuji-sim lists 0.200 N*m, and 1.5 A times its 0.1329 N*m/A
        # (the torque constant its pen-spin example also uses) is 0.19935.
        torch.testing.assert_close(got, torch.tensor(_WUJI_SIM["force_limits"]).double(), atol=7e-4, rtol=0)


@pytest.mark.parametrize("prefix", _HANDS)
def test_up_and_front_load_the_hand_fingers_up_palm_forward(env, prefix):
    """``up``/``front`` map the URDF finger axis (-Z) to world +Z and the palm normal (+/-Y) to world +X."""
    hand = env.entities[prefix]
    frames = [f"{prefix}_wrist"] + [f"{prefix}_{t}" for t in _TIPS]
    set_dofs(hand, {})
    rest = links_pos(hand, frames)
    finger_axis = rest[1:].mean(axis=0) - rest[0]
    assert finger_axis[2] / np.linalg.norm(finger_axis) > 0.9

    set_dofs(
        hand, {f"{prefix}_{f}_{j}": 0.5 for f in ("index_finger", "middle_finger") for j in ("mcp_flex", "pip", "dip")}
    )
    curl = (links_pos(hand, frames)[1:] - rest[1:]).mean(axis=0)
    assert curl[0] / np.linalg.norm(curl) > 0.5


def test_the_fingertips_are_not_where_beta2_puts_them(env):
    """hand2_2 is a new kinematic model: at the zero pose its fingertips sit more than 1 mm off beta2's."""
    frames = ["r_wrist"] + [f"r_{t}" for t in _TIPS]
    for name in ("r", "beta2_r"):
        set_dofs(env.entities[name], {})
    new, old = links_pos(env.entities["r"], frames), links_pos(env.entities["beta2_r"], frames)
    offsets = np.linalg.norm((new[1:] - new[0]) - (old[1:] - old[0]), axis=1)
    assert offsets.max() > 1e-3
