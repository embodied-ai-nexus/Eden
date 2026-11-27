"""Wuji Hand 2 hand2_2 built, covered and skeleton, both sides: wuji-sim's joint dynamics, the frames, the mount
root, and a fist that does not collide with itself.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from eden.options.robots import (
    WujiHand2_2_L,
    WujiHand2_2_R,
    WujiHand2_2Skeleton_L,
    WujiHand2_2Skeleton_R,
    WujiHand2Beta2WithMount_R,
)
from tests.options.robots.conftest import build_scene, dofs_in_declared_order, links_pos, set_dofs

pytestmark = pytest.mark.requires_genesis

_TIPS = ["thumb_tip", "index_finger_tip", "middle_finger_tip", "ring_finger_tip", "pinky_tip"]
# Entity name -> (options class, joint prefix).
_HANDS = {
    "r": (WujiHand2_2_R, "r"),
    "l": (WujiHand2_2_L, "l"),
    "skeleton_r": (WujiHand2_2Skeleton_R, "r"),
    "skeleton_l": (WujiHand2_2Skeleton_L, "l"),
}


@pytest.fixture(scope="module")
def _env():
    hands = {
        name: cls(is_fixed_base=True, default_root_pos=(0.5 * i, 0.0, 0.3))
        for i, (name, (cls, _)) in enumerate(_HANDS.items())
    }
    return build_scene(**hands, beta2_r=WujiHand2Beta2WithMount_R(is_fixed_base=True, default_root_pos=(2.0, 0.0, 0.3)))


def _frames(prefix: str) -> list[str]:
    return [f"{prefix}_wrist", f"{prefix}_wrist_imu_frame"] + [f"{prefix}_{t}" for t in _TIPS]


@pytest.fixture
def env(_env):
    yield _env
    _env.reset()


@pytest.mark.parametrize("name", _HANDS)
def test_every_declared_joint_and_frame_resolves_on_the_mount_root(env, name):
    """The fingertip and IMU frames are sites in upstream's MJCF; Eden's MJCF wraps each in a body Genesis keeps."""
    hand, (cls, prefix) = env.entities[name], _HANDS[name]
    declared = cls.model_fields["dofs_name"].default
    assert hand.num_dofs == 20
    assert dofs_in_declared_order(hand, declared) == declared
    assert declared == [n.replace("r_", f"{prefix}_", 1) for n in WujiHand2Beta2WithMount_R().dofs_name]
    assert hand.base_link_name == f"{prefix}_mount"
    kept = _frames(prefix)
    assert set(hand.find_named_links_idx_local(kept, preserve_order=True)[0]) == set(kept)


@pytest.mark.parametrize(("covered", "skeleton", "prefix"), [("r", "skeleton_r", "r"), ("l", "skeleton_l", "l")])
def test_the_cover_adds_contact_geometry_and_mass_on_the_same_frames(env, covered, skeleton, prefix):
    """wuji-description v2026.11.26: 21 cover pieces, 95.8 g in all, so 783.6 g against the bare hand's 687.8 g."""
    hands = env.entities[covered], env.entities[skeleton]
    for hand in hands:
        set_dofs(hand, {})
    rel = [links_pos(h, _frames(prefix)) - links_pos(h, _frames(prefix))[:1] for h in hands]
    np.testing.assert_allclose(rel[0], rel[1], atol=1e-6)
    masses = [sum(float(link.get_mass()) for link in h.links) for h in hands]
    np.testing.assert_allclose(masses, [0.7836, 0.6878], atol=1e-6)
    assert [len(h.geoms) for h in hands] == [48, 27]


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


@pytest.mark.parametrize("name", _HANDS)
def test_the_solver_runs_wuji_sims_joint_dynamics(env, name):
    """Per-joint armature, wuji-sim's torque limits, zero friction loss and damping, and the pen-spin gains, in place
    of the MJCF's own armature and position-actuator gains.
    """
    hand = env.entities[name]
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


@pytest.mark.parametrize("name", _HANDS)
def test_up_and_front_load_the_hand_fingers_up_palm_forward(env, name):
    """``up``/``front`` map the model's finger axis (-Z) to world +Z and the palm normal (+/-Y) to world +X."""
    hand, prefix = env.entities[name], _HANDS[name][1]
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
    for name in ("skeleton_r", "beta2_r"):
        set_dofs(env.entities[name], {})
    new, old = links_pos(env.entities["skeleton_r"], frames), links_pos(env.entities["beta2_r"], frames)
    offsets = np.linalg.norm((new[1:] - new[0]) - (old[1:] - old[0]), axis=1)
    assert offsets.max() > 1e-3


# wuji-sim v2026.11.23 ``wuji_sim/scene.py`` ``_FIST``, whose straight path from the open hand upstream keeps free of
# self-contact.
_FIST = {
    "thumb_cmc_flex": 0.625, "thumb_cmc_abd": -1.0, "thumb_mcp": 1.085, "thumb_ip": 1.085,
    "index_finger_mcp_flex": 1.3, "index_finger_mcp_abd": -0.15, "index_finger_pip": 1.9, "index_finger_dip": 1.1,
    "middle_finger_mcp_flex": 1.3, "middle_finger_mcp_abd": 0.0, "middle_finger_pip": 1.9, "middle_finger_dip": 1.1,
    "ring_finger_mcp_flex": 1.3, "ring_finger_mcp_abd": 0.075, "ring_finger_pip": 1.9, "ring_finger_dip": 1.1,
    "pinky_mcp_flex": 1.3, "pinky_mcp_abd": 0.15, "pinky_pip": 1.9, "pinky_dip": 1.1,
}  # fmt: skip


@pytest.mark.parametrize("name", _HANDS)
def test_closing_into_wuji_sims_fist_never_touches_the_hand_itself(env, name):
    """The MJCF's contact excludes reach Genesis: loaded from the URDF, which cannot carry them, the covered hand
    touches itself within the first fifth of the way, and the bare hand near the fist.
    """
    hand, prefix = env.entities[name], _HANDS[name][1]
    for alpha in np.linspace(0.0, 1.0, 11):
        set_dofs(hand, {f"{prefix}_{joint}": alpha * q for joint, q in _FIST.items()})
        assert len(hand.detect_collision()) == 0, f"self-contact at {alpha:.1f} of the way to the fist"
