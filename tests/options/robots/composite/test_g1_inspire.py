"""``G1Inspire``: the welds it emits, and the built rig's mass, meshes, rotor inertia and couplings.

The rig is Weave's merged ``g1_29dof_rev_1_0_with_inspire_hand_DFQ.urdf`` cut at its two fixed hand joints. Its
kinematic parity with that file is checked against the motions Weave recorded on it, in
``tests/datasets/test_weave_dataset.py::test_the_real_reference_links_are_the_rigs_own_forward_kinematics``: every
one of the 54 bodies at the recorded joints, which a mount frame off by a rotation would fail.
"""

from __future__ import annotations

import math

import pytest
import torch

from eden.options.robots import COMPOSITE_ROBOT_REGISTRY, G1Inspire
from eden.utils.geom import quat_from_euler_xyz
from tests.options.robots.conftest import build_scene

_ENTITIES = ("robot", "left_hand", "right_hand")
#: Weave's ``*_base_link_joint`` origins: the hand base link in the ``*_wrist_yaw_link`` frame.
_WEAVE_MOUNT_RPY = {"left": (0.0, 0.0, math.pi / 2), "right": (math.pi, 0.0, -math.pi / 2)}


# ── options ────────────────────────────────────────────────────────────────────────────────────────


def test_the_rig_expands_into_the_bare_wrist_g1_and_two_inspire_rh56dfq_hands():
    rig = G1Inspire()
    entities, attachments = rig.expand()
    assert {n: type(e).__name__ for n, e in entities.items()} == {
        "robot": "G1Dofs29NoHands",
        "left_hand": "InspireRH56DFQ_L",
        "right_hand": "InspireRH56DFQ_R",
    }
    assert entities["robot"].file == "humanoids/unitree_g1/g1_29dof_rev_1_0_bare_wrist.urdf"
    assert COMPOSITE_ROBOT_REGISTRY.get("g1_inspire") is G1Inspire
    assert set(attachments) == {"left_hand", "right_hand"}
    assert not entities["left_hand"].is_fixed_base and not entities["right_hand"].is_fixed_base


@pytest.mark.parametrize("side", ["left", "right"])
def test_each_hand_is_welded_to_its_wrist_at_weaves_fixed_hand_joint(side):
    parent, link, offset, quat = G1Inspire().attachments()[f"{side}_hand"]
    assert (parent, link) == ("robot", f"{side}_wrist_yaw_link")
    assert offset == pytest.approx((0.0415, 0.0, 0.0))
    weave = quat_from_euler_xyz(*(torch.tensor([a], dtype=torch.float64) for a in _WEAVE_MOUNT_RPY[side]))[0]
    assert float(abs((torch.tensor(quat, dtype=torch.float64) * weave).sum())) == pytest.approx(1.0, abs=1e-12)


def test_the_actuated_dofs_are_the_six_drivers_and_the_rest_follow_them():
    rig = G1Inspire()
    for actuated, mimic, dofs in (
        (rig.left_hand_actuated_dofs, rig.left_hand_mimic_dofs, rig.left_hand_dofs),
        (rig.right_hand_actuated_dofs, rig.right_hand_mimic_dofs, rig.right_hand_dofs),
    ):
        assert actuated == [
            f"{n}_joint"
            for n in (
                "thumb_proximal_yaw",
                "thumb_proximal_pitch",
                "index_proximal",
                "middle_proximal",
                "ring_proximal",
                "pinky_proximal",
            )
        ]
        assert set(actuated) | set(mimic) == set(dofs)
        assert mimic["thumb_intermediate_joint"] == ("thumb_proximal_pitch_joint", 1.6, 0.0)
        assert mimic["thumb_distal_joint"] == ("thumb_proximal_pitch_joint", 2.4, 0.0)
        for finger in ("index", "middle", "ring", "pinky"):
            assert mimic[f"{finger}_intermediate_joint"] == (f"{finger}_proximal_joint", 1.0, 0.0)
    assert rig.left_grasp_links == [
        "thumb_distal",
        "index_intermediate",
        "middle_intermediate",
        "ring_intermediate",
        "pinky_intermediate",
    ]


# ── built ──────────────────────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def env():
    """The rig with a pinned pelvis (kinematic checks, and a hand stepped in place)."""
    rig = G1Inspire()
    return build_scene(
        sim_substeps=1, robot=rig.model_copy(update={"robot": rig.robot.model_copy(update={"is_fixed_base": True})})
    )


def test_the_rig_has_29_body_dofs_and_12_per_hand(env):
    assert [env.entities[n].num_dofs for n in _ENTITIES] == [29, 12, 12]


def test_the_rig_weighs_what_weaves_merged_urdf_does(env):
    """Bare wrists plus two 0.19 kg hands: 33.3847 kg, the split conserving every link's mass."""
    rig = sum(float(env.entities[n].get_links_inertial_mass().sum()) for n in _ENTITIES)
    assert rig == pytest.approx(33.3847, abs=1e-3)
    wrist = env.entities["robot"].get_link("left_wrist_yaw_link")
    assert float(wrist.get_mass()) == pytest.approx(0.08457647, abs=1e-6)  # not the no-hand MJCF's 0.2546


def _aabb(verts: list[torch.Tensor]) -> torch.Tensor:
    v = torch.cat([x.reshape(-1, 3) for x in verts])
    return torch.cat([v.min(0).values, v.max(0).values])


@pytest.mark.parametrize("hand", ["left_hand", "right_hand"])
def test_every_hand_link_draws_its_mesh_where_it_collides(env, hand):
    """A link's visual and collision elements name the same STL, so their bounding boxes must coincide.

    Genesis once drew the thumb, ring and pinky visuals 15-45 mm off their own collision geometry (fingers seen
    floating beside the palm) because the split had written the hand's links in traversal order rather than the
    source file's; physics was unaffected, which is why only this check catches it.
    """
    entity = env.entities[hand]
    entity.set_dofs_pos(torch.full((1, entity.num_dofs), 0.4, device=entity.device))
    env.scene.visualizer.update(force=True)
    checked = 0
    for link in entity._entity.links:
        if not link.vgeoms or not link.geoms:
            continue
        visual = _aabb([g.get_vverts() for g in link.vgeoms])
        collision = _aabb([g.get_verts() for g in link.geoms])
        # 8 mm: a finger intermediate folds its massless fixed tip link's collision into its own.
        assert float((visual - collision).abs().max()) < 8e-3, link.name
        checked += 1
    assert checked == 13


def test_the_hands_carry_weaves_rotor_inertia_not_the_loader_fill(env):
    """URDF has no armature element; the explicit 0.01 must beat the 0.1 default fill."""
    for hand in ("left_hand", "right_hand"):
        armature = env.entities[hand].get_dofs_armature()
        assert torch.allclose(armature, torch.full_like(armature, 0.01))


def test_the_hand_urdfs_carry_no_mimic_equalities(env):
    """The split stripped ``<mimic>``, so the followers are free for the software mimic to drive."""
    for hand in ("left_hand", "right_hand"):
        assert len(env.entities[hand]._entity.equalities) == 0


def test_a_follower_driven_at_its_mimic_target_tracks_its_driver(env):
    """What the software mimic commands: follower target = driver target x ratio, both PD-tracked."""
    hand = env.entities["right_hand"]
    mimic = G1Inspire().right_hand_mimic_dofs
    driver_targets = {"thumb_proximal_pitch_joint": 0.3, "index_proximal_joint": 0.8}
    targets = {n: 0.0 for n in hand.dofs_name}
    targets.update(driver_targets)
    for follower, (driver, mult, offset) in mimic.items():
        targets[follower] = targets[driver] * mult + offset
    target = torch.tensor([[targets[n] for n in hand.dofs_name]], device=hand.device)
    # Start from the default posture: a hand left inside the torso makes stepping a deep-penetration solve
    # (inter-entity collisions are on), not a mimic test.
    for name in _ENTITIES:
        entity = env.entities[name]
        entity.set_dofs_pos(entity.default_dofs_pos, zero_velocity=True)
    hand.set_dofs_pos(torch.zeros_like(target))
    hand.control_dofs_pos(target)
    try:
        for _ in range(300):
            env.scene.step()
        reached = dict(zip(hand.dofs_name, hand.get_dofs_pos()[0].tolist(), strict=True))
        assert reached["index_intermediate_joint"] == pytest.approx(reached["index_proximal_joint"], abs=0.05)
        assert reached["thumb_distal_joint"] == pytest.approx(2.4 * reached["thumb_proximal_pitch_joint"], abs=0.1)
        assert reached["index_proximal_joint"] == pytest.approx(0.8, abs=0.1)
    finally:
        hand.control_dofs_pos(torch.zeros_like(target))
