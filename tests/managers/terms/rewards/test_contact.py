"""The contact rewards in ``eden/managers/terms/rewards/contact.py``.

``MotionContactChamferReward``'s tensor kernel is tested directly. The demo-contact rewards run on a mimic env whose boxes rest on
fixed partners, with the demonstrated contacts written into the motion reference.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from eden.managers.terms.rewards.contact import (
    EMPTY_DISTANCE_SENTINEL,
    MotionContactDistanceReward,
    MotionContactMatchReward,
    _compute_contact_chamfer_reward,
)
from tests.managers.terms.conftest import write_stack_urdf
from tests.support.motion import build_mimic_env, manager_options, set_reference, term_options, write_clip

STD = 1.0 / math.sqrt(30.0)  # DexMachina's contact_beta = 30
SIGMOID_B = 5.0
SIGMOID_OFFSET = 1.0 - 1.0 / (1.0 + math.exp(-SIGMOID_B))  # keeps the sigmoid kernel's peak at 1
IDENTITY_QUAT = torch.tensor([[1.0, 0.0, 0.0, 0.0]])


def _reward(policy_pos, demo_pos, policy_valid=None, demo_valid=None, **overrides):
    """The kernel over ``(B, n_a, n_b, 3)`` clouds, all points valid unless masks are given."""
    params = {
        "policy_pos": policy_pos,
        "policy_valid": torch.ones(policy_pos.shape[:-1], dtype=torch.bool) if policy_valid is None else policy_valid,
        "policy_anchor_pos": None,
        "policy_anchor_quat": None,
        "demo_pos": demo_pos,
        "demo_valid": torch.ones(demo_pos.shape[:-1], dtype=torch.bool) if demo_valid is None else demo_valid,
        "demo_anchor_pos": None,
        "demo_anchor_quat": None,
        "kernel": "exp",
        "std": STD,
        "sigmoid_a": 100.0,
        "sigmoid_b": SIGMOID_B,
        "sigmoid_offset": SIGMOID_OFFSET,
        "zero_contact_mode": "zero",
        "empty_distance": EMPTY_DISTANCE_SENTINEL,
    }
    return _compute_contact_chamfer_reward(**{**params, **overrides})


def _point(x, y=0.0, z=0.0):
    return torch.tensor([[[[x, y, z]]]])  # (1, 1, 1, 3)


@pytest.mark.parametrize(
    ("distance", "expected"),
    [(0.0, 1.0), (0.05, math.exp(-30.0 * 0.05)), (1.0, 0.0)],  # 0.05 m reproduces DexMachina's beta = 30
)
def test_exp_kernel_is_exp_of_minus_beta_times_the_chamfer_distance(distance, expected):
    assert _reward(_point(0.0), _point(distance)).item() == pytest.approx(expected, rel=1e-5, abs=1e-6)


@pytest.mark.parametrize("num_envs", [1, 7])
def test_sigmoid_kernel_peaks_at_one_and_falls_with_distance(num_envs):
    """The slope is negated, so a larger contact error never earns more."""
    rewards = torch.stack(
        [
            _reward(torch.zeros(num_envs, 1, 1, 3), _point(d).expand(num_envs, 1, 1, 3), kernel="sigmoid")
            for d in torch.linspace(0.0, 0.2, 12).tolist()
        ]
    )
    torch.testing.assert_close(rewards[0], torch.ones(num_envs))
    assert (rewards[1:] - rewards[:-1] <= 1e-6).all(), rewards


@pytest.mark.parametrize(("mode", "expected"), [("zero", 0.0), ("one", 1.0)])
def test_both_clouds_empty_follow_the_zero_contact_mode(mode, expected):
    """DexMachina's ``mask_zero_contact`` True / False: no contact prescribed and none made."""
    empty = torch.zeros(1, 1, 1, dtype=torch.bool)
    assert _reward(_point(0.0), _point(0.0), empty, empty, zero_contact_mode=mode).item() == expected


def test_one_empty_cloud_scores_through_the_sentinel_distance():
    """Contact the demo does not have is a near-zero reward, not the both-empty mask's value."""
    out = _reward(_point(0.1), _point(0.0), demo_valid=torch.zeros(1, 1, 1, dtype=torch.bool))
    assert 0.0 <= out.item() < 1e-6


def test_the_zero_contact_mask_is_per_env():
    pos = torch.tensor([[[[0.1, 0.0, 0.0]]], [[[0.0, 0.0, 0.0]]]])  # (2, 1, 1, 3)
    valid = torch.tensor([[[True]], [[False]]])
    torch.testing.assert_close(_reward(pos, pos.clone(), valid, valid), torch.tensor([1.0, 0.0]))


def test_each_cloud_is_compared_in_its_own_anchors_frame():
    """Independent world offsets of the two anchors leave the reward unchanged (DexMachina's two-anchor form).

    One shared anchor would carry the offset between the policy and demo objects into the distance.
    """
    local_policy, local_demo = _point(0.1, 0.2, 0.3), _point(0.15, 0.25, 0.35)  # a non-saturating error

    def anchored(policy_anchor, demo_anchor):
        return _reward(
            local_policy + policy_anchor.view(1, 1, 1, 3),
            local_demo + demo_anchor.view(1, 1, 1, 3),
            policy_anchor_pos=policy_anchor,
            policy_anchor_quat=IDENTITY_QUAT,
            demo_anchor_pos=demo_anchor,
            demo_anchor_quat=IDENTITY_QUAT,
        )

    in_world = _reward(local_policy, local_demo)
    torch.testing.assert_close(anchored(torch.zeros(1, 3), torch.zeros(1, 3)), in_world)
    torch.testing.assert_close(anchored(torch.tensor([[10.0, -5.0, 2.0]]), torch.tensor([[-7.0, 3.0, 1.0]])), in_world)
    assert 0.01 < in_world.item() < 0.9  # not saturated, so a frame error would show


def test_the_same_contact_pattern_on_objects_apart_scores_the_maximum():
    """The demo object 0.5 m from the policy's: identical local contacts, so kernel(0) = 1."""
    local = _point(0.05)
    demo_anchor = torch.tensor([[0.5, 0.0, 0.0]])
    out = _reward(
        local,
        local + demo_anchor.view(1, 1, 1, 3),
        policy_anchor_pos=torch.zeros(1, 3),
        policy_anchor_quat=IDENTITY_QUAT,
        demo_anchor_pos=demo_anchor,
        demo_anchor_quat=IDENTITY_QUAT,
    )
    assert out.item() == pytest.approx(1.0, abs=1e-5)


# ---------------------------------------------------------------------------
# MotionContactMatchReward / MotionContactDistanceReward
# ---------------------------------------------------------------------------

BOX = ["box_baselink"]
SEQUENCES = {
    "robot": ([], BOX),
    "hand1": ([], BOX),
    "object": ([], ["bottom", "top"]),
    "cube": ([], BOX),
}
FRAMES = 20


@pytest.fixture(scope="module")
def mimic_env(tmp_path_factory):
    """Box ``robot`` rests on the top of a fixed two-link stack (``object``), box ``hand1`` on a fixed ``cube``."""
    from eden.options.entities import BoxOptions, EntityOptions

    root = tmp_path_factory.mktemp("contact")
    contact = {
        "robot.contact_points_w": np.zeros((FRAMES, 1, 2, 3), np.float32),
        "robot.contact_points_valid": np.zeros((FRAMES, 1, 2), bool),
        "hand1.contact_points_w": np.zeros((FRAMES, 1, 1, 3), np.float32),
        "hand1.contact_points_valid": np.zeros((FRAMES, 1, 1), bool),
    }
    write_clip(root / "clip_00.npz", FRAMES, sequences=SEQUENCES, overrides=contact)
    scene = {
        "robot": BoxOptions(size=(0.1, 0.1, 0.1), default_root_pos=(0.0, 0.0, 0.3)),
        "hand1": BoxOptions(size=(0.1, 0.1, 0.1), default_root_pos=(2.0, 0.0, 0.3)),
        "object": EntityOptions(file=write_stack_urdf(root / "stack.urdf"), links_to_keep=["top"], is_fixed_base=True),
        "cube": BoxOptions(size=(0.2, 0.2, 0.2), default_root_pos=(2.0, 0.0, 0.1), is_fixed_base=True),
    }
    env = build_mimic_env(
        root,
        scene=scene,
        num_envs=2,
        motion_options=manager_options(terms={"default": term_options(root, sequences=SEQUENCES)}),
    )
    for _ in range(30):  # let both boxes settle onto their partners
        env.step()
    return env


def _term(mimic_env, cls, **params):
    term = cls(mimic_env, cls.configure(hand_links_name=BOX, **params))
    term.build()
    return term


def _demo(mimic_env, valid, points=None, frame_valid=True):
    """Write the robot's demo contact (partner columns ordered top, bottom) and put the object where it is."""
    obj = mimic_env.entities["object"]
    fields = {
        "contact_points_valid": torch.tensor(valid, dtype=torch.bool).view(1, 1, 1, 2),
        "validity": torch.tensor(frame_valid),
    }
    if points is not None:
        fields["contact_points_w"] = points[:, None, :, :, :]
    set_reference(mimic_env, "robot", **fields)
    set_reference(mimic_env, "object", base_pos_w=obj.get_pos()[:, None], base_quat_w=obj.get_quat()[:, None])


def _sim_contacts(mimic_env, order=("top", "bottom")):
    robot, obj = mimic_env.entities["robot"], mimic_env.entities["object"]
    return robot.get_grouped_contacts(
        obj, links_b_idx_local=obj.find_named_links_idx_local(list(order), preserve_order=True)[1]
    )


@pytest.mark.requires_genesis
def test_the_box_touches_only_the_top_of_the_stack(mimic_env):
    assert _sim_contacts(mimic_env)["valid"][:, 0].tolist() == [[True, False]] * 2


@pytest.mark.requires_genesis
@pytest.mark.parametrize(
    "demo_valid, frame_valid, expected",
    [
        ([True, False], True, 1.0),
        # Demo contact with the bottom plus sim contact with the top is no match (a phantom one before).
        ([False, True], True, 0.0),
        ([True, False], False, 0.0),  # past the clip end
    ],
    ids=["same-partner", "other-partner", "past-clip-end"],
)
def test_match_reward_credits_demo_contacts_the_sim_makes_on_the_same_partner(
    mimic_env, demo_valid, frame_valid, expected
):
    _demo(mimic_env, demo_valid, frame_valid=frame_valid)

    reward = _term(mimic_env, MotionContactMatchReward, object_links_name=["top", "bottom"]).compute()

    torch.testing.assert_close(reward.cpu(), torch.full((2,), expected))


@pytest.mark.requires_genesis
@pytest.mark.parametrize("offset, expected", [(0.0, 1.0), (0.09, math.exp(-9.0))])
def test_distance_reward_scores_where_on_the_object_the_link_touches(mimic_env, offset, expected):
    sim_points = _sim_contacts(mimic_env)["position"]  # (N, 1, 2, 3), world frame
    _demo(mimic_env, [True, False], points=sim_points + torch.tensor([offset, 0.0, 0.0], device=sim_points.device))

    reward = _term(mimic_env, MotionContactDistanceReward, object_links_name=["top", "bottom"], std=0.03).compute()

    torch.testing.assert_close(reward.cpu(), torch.full((2,), expected), atol=1e-5, rtol=1e-4)


@pytest.mark.requires_genesis
def test_distance_reward_without_a_joint_contact_is_zero(mimic_env):
    _demo(mimic_env, [False, False], points=torch.zeros(2, 1, 2, 3))

    reward = _term(mimic_env, MotionContactDistanceReward, object_links_name=["top", "bottom"], std=0.03).compute()

    assert reward.tolist() == [0.0, 0.0]


@pytest.mark.requires_genesis
@pytest.mark.parametrize(
    "params, match",
    [
        ({}, "2 partner columns, so 'object_links_name' is required"),
        ({"object_links_name": ["top"]}, r"object_links_name \(1\) does not match the demo contact reference's 2"),
    ],
)
def test_partner_columns_must_be_pinned_to_object_links(mimic_env, params, match):
    with pytest.raises(ValueError, match=match):
        _term(mimic_env, MotionContactMatchReward, **params)


@pytest.mark.requires_genesis
def test_a_single_partner_reference_needs_no_object_links(mimic_env):
    set_reference(mimic_env, "hand1", contact_points_valid=torch.ones(1, dtype=torch.bool), validity=torch.tensor(True))

    term = MotionContactMatchReward(
        mimic_env,
        MotionContactMatchReward.configure(hand_entity_name="hand1", object_entity_name="cube", hand_links_name=BOX),
    )
    term.build()

    torch.testing.assert_close(term.compute().cpu(), torch.ones(2))
