"""``KukaSharpa``: the offset weld the rig declares, and where the built solver puts the hand.

It is the one rig with no dedicated attach body, so the whole arm -> hand transform rides on
``attach_offset`` / ``attach_quat``; drop them and the hand collapses onto the flange, missing every
grasp by ~9.5 cm and 75 deg. Reference values come from SimToolReal's merged 29-DOF URDF.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from genesis.utils import geom as gu

from eden.options.robots import COMPOSITE_ROBOT_REGISTRY, KukaSharpa, SharpaWave_L
from eden.options.robots.composite.kuka_sharpa import SIMTOOLREAL_HAND_DOFS_POS_LIMITS
from tests.options.robots.conftest import build_scene, link_pose, links_pos

_BASE_POS = (0.0, 0.8, 0.0)
_TIPS = ["left_index_DP", "left_middle_DP", "left_ring_DP", "left_thumb_DP", "left_pinky_DP"]
_YAW = math.radians(-75.0)  # 15 - 90 deg through the merged URDF's two intermediate frames
_GRIP = 1.5


# ── options ────────────────────────────────────────────────────────────────────────────────────────


def test_the_rig_welds_the_hand_at_the_flange_through_the_mount_offset():
    rig = KukaSharpa(base_pos=_BASE_POS)
    entities, attachments = rig.expand()
    assert {n: type(e).__name__ for n, e in entities.items()} == {"robot": "Iiwa14", "hand": "SharpaWave_L"}
    assert tuple(entities["robot"].default_root_pos) == _BASE_POS
    assert (len(rig.arm_dofs), len(rig.hand_dofs)) == (7, 22)
    assert rig.grasp_links == _TIPS
    assert COMPOSITE_ROBOT_REGISTRY.get("kuka_sharpa") is KukaSharpa

    child, link, pos, quat = attachments["hand"]
    assert (child, link, pos) == ("robot", "link7", (0.0, 0.0, 0.095))
    assert quat == pytest.approx((math.cos(_YAW / 2), 0.0, 0.0, math.sin(_YAW / 2)), abs=1e-12)


def test_the_hand_gains_are_rekeyed_onto_the_stock_joint_names():
    """SimToolReal's URDF renames five proximal joints; unrekeyed, those would silently keep the sim's own gains."""
    rig = KukaSharpa()
    hand_dofs = set(rig.hand_dofs)
    for field in ("default_dofs_kp", "default_dofs_kd", "default_dofs_armature"):
        assert set(getattr(rig.hand, field)) == hand_dofs, field
    # The ported PD gains assume a gravity-free arm and hand.
    assert rig.arm.material.gravity_compensation == rig.hand.material.gravity_compensation == 1.0


# ── built ──────────────────────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def _env():
    """The stock rig, plus a second one whose hand restricts its limits and grips harder at the tips."""
    tuned_hand = SharpaWave_L(
        is_fixed_base=False,
        default_dofs_pos_limits=dict(SIMTOOLREAL_HAND_DOFS_POS_LIMITS),
        default_geoms_friction={link: _GRIP for link in _TIPS},
    )
    return build_scene(
        robot=KukaSharpa(base_pos=_BASE_POS),
        tuned=KukaSharpa(arm_name="tuned", hand_name="tuned_hand", base_pos=(2.0, 0.8, 0.0), hand=tuned_hand),
    )


@pytest.fixture
def env(_env):
    yield _env
    _env.reset()


def test_the_weld_leaves_the_hand_with_its_finger_dofs_only(env):
    arm, hand = env.entities["robot"], env.entities["hand"]
    assert (arm.num_dofs, hand.num_dofs, hand.n_dofs) == (7, 22, 22)


def test_the_hand_root_sits_at_the_mount_transform_off_the_flange_in_any_arm_pose(env):
    arm, hand = env.entities["robot"], env.entities["hand"]
    lo, hi = (bound[0].cpu().numpy() for bound in arm.get_dofs_limit())
    twisted = lo + np.random.default_rng(0).uniform(0.2, 0.8, lo.shape) * (hi - lo)
    mount = np.array([math.cos(_YAW / 2), 0.0, 0.0, math.sin(_YAW / 2)])

    for q in (arm.default_dofs_pos[0].cpu().numpy(), twisted):
        arm.set_qpos(torch.as_tensor(q, dtype=torch.float32, device=env.device)[None])
        flange_pos, flange_quat = link_pose(arm, "link7")
        hand_pos, hand_quat = link_pose(hand, "left_hand_C_MC")
        np.testing.assert_allclose(
            hand_pos, flange_pos + gu.transform_by_quat(np.array([0.0, 0.0, 0.095]), flange_quat), atol=1e-5
        )
        np.testing.assert_allclose(
            gu.quat_to_R(hand_quat), gu.quat_to_R(gu.transform_quat_by_quat(mount, flange_quat)), atol=1e-4
        )


def test_the_arm_carries_the_mjcf_s_missing_damping_and_the_hand_its_rotor_inertia(env):
    arm, hand = env.entities["robot"], env.entities["hand"]
    assert torch.allclose(arm.get_dofs_damping(), torch.tensor(0.5))
    assert torch.allclose(arm.get_dofs_force_range()[1], torch.tensor(300.0))
    # SimToolReal's ~3e-3 finger rotor inertia, not Genesis' 0.1 morph default.
    assert (hand.get_dofs_armature() < 0.01).all()


def test_the_fingertips_survive_the_fixed_link_merge_as_five_distinct_points(env):
    tips = links_pos(env.entities["hand"], KukaSharpa().grasp_links)
    spread = np.linalg.norm(tips[:, None] - tips[None], axis=-1)
    assert spread[np.triu_indices(5, k=1)].min() > 0.005 and spread.max() < 0.30


def test_restricted_hand_limits_move_edens_view_but_not_the_joint_stops(env):
    """Genesis reads hard limits from the asset and has no runtime setter, so only the soft range moves.

    Also the one range-per-DOF ``default_dofs_*`` entry, whose tuple path used to raise.
    """
    hand = env.entities["tuned_hand"]
    solver_hi = hand.get_dofs_limit()[1][0]
    for name, (lo, hi) in SIMTOOLREAL_HAND_DOFS_POS_LIMITS.items():
        i = hand.dofs_name.index(name)
        assert hand.default_dofs_pos_limits[0, i].tolist() == pytest.approx([lo, hi])
        assert solver_hi[i].item() > hi + 1e-3
        mid, half = (lo + hi) / 2, (hi - lo) / 2
        assert hand.soft_dofs_pos_limits[0, i].tolist() == pytest.approx([mid - 0.9 * half, mid + 0.9 * half])


def test_default_geoms_friction_raises_the_matched_links_only_and_keeps_every_link_colliding(env):
    """Unlike ``collision_friction``, which also switches collision off outside ``collision_link_patterns``."""
    hand = env.entities["tuned_hand"]
    tips = [float(g.get_friction()) for g in hand.geoms if g.link.name in _TIPS]
    others = [float(g.get_friction()) for g in hand.geoms if g.link.name not in _TIPS]
    assert tips and others
    assert all(f == pytest.approx(_GRIP) for f in tips)
    assert all(f != pytest.approx(_GRIP) for f in others)


def test_a_friction_pattern_matching_no_link_raises_listing_the_links():
    hand = SharpaWave_L(is_fixed_base=False, default_geoms_friction={"nope_*": 1.5})
    with pytest.raises(ValueError, match="patterns are matched"):
        build_scene(robot=KukaSharpa(base_pos=_BASE_POS, hand=hand))
