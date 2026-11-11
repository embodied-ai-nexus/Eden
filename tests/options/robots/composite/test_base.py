"""The composite bases: expansion into entities and welds, scene placement, validation, and round trips.

Arm rigs are exercised through ``Droid`` and ``FrankaXHand``; humanoid ones through a G1 + Wuji rig
built straight from ``HumanoidOptions``, so the base is shown to carry a whole rig on its own.
"""

from __future__ import annotations

import copy
import math

import numpy as np
import pytest
from genesis.utils import geom as gu

import eden.options.robots as robots
import eden.options.robots.composite as composite
from eden.options import SceneOptions
from eden.options.entities import CompositeEntityOptions
from eden.options.robots import (
    COMPOSITE_ROBOT_REGISTRY,
    DexManipulatorOptions,
    Droid,
    FrankaResearch3,
    FrankaXHand,
    G1Dofs29NoHands,
    HumanoidOptions,
    KukaSharpa,
    OliNoHands,
    OliWuji,
    PandaFrankaHand,
    Robotiq2f85,
    UR5e,
    WujiHand2Beta2WithMount_L,
    WujiHand2Beta2WithMount_R,
    XArm7XHand1_L,
)
from eden.options.robots.composite.base import _ArmCompositeOptions, has_frame_alignment, weld_quat
from eden.utils.configs import EdenConfig
from eden.utils.geom import align_up_and_front

_YAW90 = (math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5))


def _g1_wuji(**overrides) -> HumanoidOptions:
    """G1 (hands stripped) + mounted Wuji hands; ``overrides`` replace the defaults rather than add to them."""
    fields = dict(
        robot=G1Dofs29NoHands(),
        left_hand=WujiHand2Beta2WithMount_L(is_fixed_base=False),
        right_hand=WujiHand2Beta2WithMount_R(is_fixed_base=False),
        left_ee_link="left_wrist_yaw_link",
        right_ee_link="right_wrist_yaw_link",
        left_grasp_links=list(WujiHand2Beta2WithMount_L.metadata.target_hand_links),
        right_grasp_links=list(WujiHand2Beta2WithMount_R.metadata.target_hand_links),
    )
    return HumanoidOptions(**{**fields, **overrides})


def _round_trip(tmp_path, **entities):
    path = str(tmp_path / "scene.json")
    EdenConfig(scene_options=SceneOptions(**entities)).save_as_json(path, lock_config=False)
    return EdenConfig.load_from_file(path).scene_options


def _mount_R(hand, quat) -> np.ndarray:
    """The rotation a weld must carry for ``quat`` to be the hand's base-link orientation on the flange."""
    return gu.quat_to_R(np.asarray(quat)) @ align_up_and_front(hand.up, hand.front).T


# ── arm rigs ───────────────────────────────────────────────────────────────────────────────────────


def test_expand_yields_fresh_copies_and_leaves_the_rig_untouched():
    droid = Droid(base_pos=(0.0, 0.0, 0.4))
    first, attachments = droid.expand()
    assert attachments == {"gripper": ("robot", "attachment")}
    assert tuple(first["robot"].default_root_pos) == (0.0, 0.0, 0.4)
    second, _ = droid.expand()
    first["robot"].default_root_pos = (9.0, 9.0, 9.0)
    assert tuple(second["robot"].default_root_pos) == (0.0, 0.0, 0.4)
    assert tuple(droid.arm.default_root_pos) == (0.0, 0.0, 0.4)


def test_base_pos_none_leaves_a_caller_set_arm_placement_alone():
    droid = Droid(arm=FrankaResearch3(default_root_pos=(1.0, 0.0, 0.0)))
    assert tuple(droid.expand()[0]["robot"].default_root_pos) == (1.0, 0.0, 0.0)


@pytest.mark.parametrize(
    "make, gripper_pose, weld",
    [
        # No mount stated: the end-effector's own default root pose is the mount (the 2-tuple).
        (lambda: Droid(), None, ("robot", "attachment")),
        (
            lambda: Droid(attach_offset=(0.0, 0.0, 0.05), attach_quat=(0.0, 0.0, 0.0, 1.0)),
            None,
            ("robot", "attachment", (0.0, 0.0, 0.05), (0.0, 0.0, 0.0, 1.0)),
        ),
        # One component set: the other falls back to the end-effector's own default, never zero/identity.
        (
            lambda: Droid(attach_offset=(0.0, 0.0, 0.05)),
            dict(default_root_quat=(0.0, 1.0, 0.0, 0.0)),
            ("robot", "attachment", (0.0, 0.0, 0.05), (0.0, 1.0, 0.0, 0.0)),
        ),
        (
            lambda: Droid(attach_quat=(0.0, 0.0, 0.0, 1.0)),
            dict(default_root_pos=(0.0, 0.0, 0.02)),
            ("robot", "attachment", (0.0, 0.0, 0.02), (0.0, 0.0, 0.0, 1.0)),
        ),
    ],
    ids=["no-mount", "full-mount", "offset-only", "quat-only"],
)
def test_the_weld_takes_the_stated_mount_component_by_component(make, gripper_pose, weld):
    droid = make()
    for field, value in (gripper_pose or {}).items():
        setattr(droid.gripper, field, value)
    assert droid.expand()[1] == {"gripper": weld}


def test_a_dex_hand_takes_the_same_weld_rules():
    unmounted = {"attach_offset": None, "attach_quat": None}
    assert FrankaXHand(**unmounted).expand()[1] == {"hand": ("robot", "attachment")}
    assert FrankaXHand(**{**unmounted, "attach_offset": (0.0, 0.0, 0.05)}).expand()[1] == {
        "hand": ("robot", "attachment", (0.0, 0.0, 0.05), (1.0, 0.0, 0.0, 0.0))
    }


@pytest.mark.parametrize("cls", [KukaSharpa, FrankaXHand, XArm7XHand1_L])
def test_a_rig_redeclaring_the_mount_fields_keeps_their_validation(cls):
    """A bare tuple annotation would drop the quaternion normalizer and the NaN guard, skewing the mount."""
    offset, quat = (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)
    assert cls(attach_offset=offset, attach_quat=(1.0, 1.0, 0.0, 0.0)).attach_quat == pytest.approx(
        (2**-0.5, 2**-0.5, 0.0, 0.0)
    )
    with pytest.raises(ValueError, match="nan|finite"):
        cls(attach_offset=(0.0, float("nan"), 0.0), attach_quat=quat)


@pytest.mark.parametrize(
    "make",
    [
        lambda: FrankaXHand(attach_offset=(0.0, 0.0, 0.05)),
        lambda: FrankaXHand(attach_quat=(1.0, 0.0, 0.0, 0.0)),
        lambda: KukaSharpa(attach_offset=(0.0, 0.0, 0.05)),
        lambda: XArm7XHand1_L(attach_quat=(1.0, 0.0, 0.0, 0.0)),
    ],
    ids=["franka-xhand-offset", "franka-xhand-quat", "kuka-sharpa-offset", "xarm-xhand-quat"],
)
def test_half_a_mount_over_a_rigs_own_mount_is_refused(make):
    """The weld would keep the rig's default for the other half: a new offset on the old mount's rotation."""
    with pytest.raises(ValueError, match=r"sets half of a mount.*Set both"):
        make()


def test_half_a_mount_is_fine_where_the_other_half_has_no_rig_default():
    """There the unset half is the end-effector's own pose, as on any rig without a mount of its own."""
    assert XArm7XHand1_L(attach_offset=(0.0, 0.0, 0.05)).attach_offset == (0.0, 0.0, 0.05)
    assert XArm7XHand1_L(attach_offset=None, attach_quat=(1.0, 0.0, 0.0, 0.0)).attach_offset is None
    assert FrankaXHand(attach_offset=(0.0, 0.0, 0.05), attach_quat=None).expand()[1]["hand"][2] == (0.0, 0.0, 0.05)


def test_an_aligned_hand_on_an_arm_rig_gets_the_cancellation_with_nothing_set():
    """The next arm + Wuji/MANO rig cannot repeat OliWuji's bug: an aligned hand never takes the 2-tuple."""
    rig = DexManipulatorOptions(
        arm=FrankaResearch3(), ee_link="attachment", hand=WujiHand2Beta2WithMount_R(is_fixed_base=False)
    )
    _, _, pos, quat = rig.attachments()["hand"]
    assert tuple(pos) == tuple(rig.hand.default_root_pos)
    np.testing.assert_allclose(
        gu.quat_to_R(np.asarray(quat)), _mount_R(rig.hand, rig.hand.default_root_quat), atol=1e-9
    )


# ── scene placement ────────────────────────────────────────────────────────────────────────────────


def test_a_scene_keeps_the_composite_as_one_child():
    """Not flattened at construction, which is what lets a saved config record and reload the rig itself."""
    scene = SceneOptions(robot=Droid(base_pos=(0.0, 0.0, 0.4)))
    assert isinstance(scene.__pydantic_extra__["robot"], Droid)
    assert scene.attachments_dict == {}  # the welds are the composite's, not the scene's


def test_the_placement_key_must_name_the_primary_body():
    with pytest.raises(ValueError, match="primary entity"):
        SceneOptions(foo=Droid())
    with pytest.raises(ValueError, match="primary entity"):
        SceneOptions(left_hand=OliWuji())


def test_two_renamed_composites_share_a_scene():
    left = Droid(arm_name="left", gripper_name="left_gripper", base_pos=(0.0, 0.3, 0.0))
    right = Droid(arm_name="right", gripper_name="right_gripper", base_pos=(0.0, -0.3, 0.0))
    SceneOptions(left=left, right=right)
    assert left.expand()[1] == {"left_gripper": ("left", "attachment")}
    assert right.expand()[1] == {"right_gripper": ("right", "attachment")}


@pytest.mark.parametrize(
    "entities",
    [
        dict(left=Droid(arm_name="left", gripper_name="hand"), right=Droid(arm_name="right", gripper_name="hand")),
        dict(robot=Droid(), gripper=Robotiq2f85()),
    ],
    ids=["two-composites", "composite-and-entity"],
)
def test_a_sub_entity_name_collision_fails_at_the_scene(entities):
    with pytest.raises(ValueError, match="collides"):
        SceneOptions(**entities)


@pytest.mark.parametrize("name, cls", list(COMPOSITE_ROBOT_REGISTRY))
def test_every_registered_rig_round_trips_through_a_saved_config(tmp_path, name, cls):
    rig = cls(base_pos=(0.0, 0.5, 0.4))
    loaded = getattr(_round_trip(tmp_path, **{rig.primary_name: rig}), rig.primary_name)
    assert type(loaded) is cls
    assert tuple(loaded.base_pos) == (0.0, 0.5, 0.4)
    assert {n: type(e) for n, e in loaded.expand()[0].items()} == {n: type(e) for n, e in rig.expand()[0].items()}
    assert loaded.expand()[1] == rig.expand()[1]


def test_a_stated_mount_round_trips(tmp_path):
    droid = Droid(attach_offset=(0.0, 0.0, 0.05), attach_quat=(0.0, 0.0, 0.0, 1.0))
    assert _round_trip(tmp_path, robot=droid).robot.expand()[1] == droid.expand()[1]


def test_a_saved_unmounted_rig_stays_unmounted_where_the_rig_now_defaults_a_mount(tmp_path):
    """A config saved while ``FrankaXHand`` welded the hand flush stores ``attach_offset: null``; it reloads flush."""
    flush = FrankaXHand(attach_offset=None, attach_quat=None)
    assert _round_trip(tmp_path, robot=flush).robot.expand()[1] == {"hand": ("robot", "attachment")}


# ── humanoid rigs ──────────────────────────────────────────────────────────────────────────────────


def test_the_humanoid_base_carries_a_whole_rig_with_one_cancelled_weld_per_wrist():
    rig = _g1_wuji()
    assert rig.entity_names == ["robot", "left_hand", "right_hand"]
    assert rig.robot_dofs == list(rig.robot.dofs_name) and len(rig.left_hand_dofs) == 20
    for side in ("left", "right"):
        parent, link, offset, quat = rig.attachments()[f"{side}_hand"]
        assert (parent, link, offset) == ("robot", f"{side}_wrist_yaw_link", (0.0, 0.0, 0.0))
        hand = getattr(rig, f"{side}_hand")
        # Genesis applies the weld before the child's up/front offset, so the two compose to mount_quat.
        np.testing.assert_allclose(
            gu.quat_to_R(np.asarray(quat)) @ align_up_and_front(hand.up, hand.front), np.eye(3), atol=1e-9
        )


def test_a_humanoid_can_wear_a_gripper_per_wrist():
    """An Eden-default gripper has no alignment, so its weld is the plain mount; open/close stay on the gripper."""
    rig = _g1_wuji(left_hand=Robotiq2f85(is_fixed_base=False), right_hand=Robotiq2f85(is_fixed_base=False))
    assert not has_frame_alignment(rig.left_hand)
    for side in ("left", "right"):
        np.testing.assert_allclose(rig.attachments()[f"{side}_hand"][3], rig.mount_quat, atol=1e-12)
    assert set(rig.left_hand.open_dofs_pos) == {"left_driver_joint", "right_driver_joint"}


def test_base_pos_places_the_pelvis_only_when_set():
    assert tuple(_g1_wuji().robot.default_root_pos) == tuple(G1Dofs29NoHands().default_root_pos)
    assert tuple(_g1_wuji(base_pos=(1.0, 2.0, 0.8)).robot.default_root_pos) == (1.0, 2.0, 0.8)


def test_renamed_humanoid_entities_carry_their_welds_along():
    rig = _g1_wuji(robot_name="alice", left_hand_name="alice_left", right_hand_name="alice_right")
    assert rig.entity_names == ["alice", "alice_left", "alice_right"]
    assert {spec[0] for spec in rig.attachments().values()} == {"alice"}
    SceneOptions(alice=rig)
    with pytest.raises(ValueError, match="primary entity"):
        SceneOptions(robot=rig)


def test_the_shared_mount_transform_can_be_overridden_per_side_and_component():
    """Needed where the wrist frames are not mirror images; unset overrides keep the shared value."""
    rig = _g1_wuji(mount_offset=(0.0, 0.0, -0.02), left_mount_quat=_YAW90, right_mount_offset=(0.0, 0.01, 0.0))
    _, _, left_offset, left_quat = rig.attachments()["left_hand"]
    _, _, right_offset, right_quat = rig.attachments()["right_hand"]
    assert (left_offset, right_offset) == ((0.0, 0.0, -0.02), (0.0, 0.01, 0.0))
    np.testing.assert_allclose(gu.quat_to_R(np.asarray(left_quat)), _mount_R(rig.left_hand, _YAW90), atol=1e-9)
    np.testing.assert_allclose(
        gu.quat_to_R(np.asarray(right_quat)), _mount_R(rig.right_hand, rig.mount_quat), atol=1e-9
    )


def test_a_non_default_humanoid_round_trips(tmp_path):
    """Non-default names and mount are where ``model_construct`` and the name guard interact on load."""
    rig = _g1_wuji(
        robot_name="alice", left_hand_name="alice_left", right_hand_name="alice_right", base_pos=(0.0, 0.0, 0.78)
    )
    loaded = _round_trip(tmp_path, alice=rig).alice
    assert type(loaded) is HumanoidOptions
    assert loaded.entity_names == ["alice", "alice_left", "alice_right"]
    assert loaded.expand()[1] == rig.expand()[1]


@pytest.mark.parametrize("override", [None, ("left_sole", "right_sole"), ()], ids=["inherit", "override", "none"])
def test_the_feet_override_reloads_as_the_intent_not_the_resolved_value(tmp_path, override):
    """Freezing the resolved value at save time would stop the rig tracking its humanoid's feet."""
    rig = OliWuji(feet_links_name=override)
    loaded = _round_trip(tmp_path, robot=rig).robot
    assert loaded.feet_links_name == override and loaded.feet_links == rig.feet_links


def test_the_feet_are_resolved_on_read_and_follow_a_swapped_humanoid():
    rig = _g1_wuji()
    assert rig.feet_links == tuple(G1Dofs29NoHands.feet_links_name)
    rig.robot = OliNoHands()
    assert rig.feet_links == tuple(OliNoHands.feet_links_name)


def test_a_locked_rig_copies_as_a_snapshot_without_unlocking_the_source():
    """``base_pos`` set after construction does not move the body; a copy must neither re-apply nor alias it."""
    rig = OliWuji(base_pos=(0.0, 0.0, 0.903))
    rig.base_pos = (0.0, 0.0, 1.0)
    rig.lock()
    clone = copy.deepcopy(rig)
    assert clone.robot is not rig.robot
    assert (tuple(clone.base_pos), tuple(clone.robot.default_root_pos)) == ((0.0, 0.0, 1.0), (0.0, 0.0, 0.903))

    config = EdenConfig(scene_options=SceneOptions(robot=rig))
    config.lock()
    copied = config.clone(unlock=True)
    copied.scene_options.robot.base_pos = (1.0, 2.0, 3.0)
    assert tuple(config.scene_options.robot.base_pos) == (0.0, 0.0, 1.0)
    with pytest.raises(RuntimeError, match="locked"):
        config.scene_options.robot.base_pos = (1.0, 2.0, 3.0)


def test_base_pos_is_reapplied_on_load_but_not_on_copy(tmp_path):
    """Deliberately asymmetric (see ``_apply_base_pos``): deepcopy snapshots, validated load re-runs the knob."""
    rig = OliWuji(base_pos=(0.0, 0.0, 1.0))
    rig.robot.default_root_pos = (0.0, 0.0, 0.5)  # drift the body behind the knob
    assert tuple(copy.deepcopy(rig).robot.default_root_pos) == (0.0, 0.0, 0.5)
    assert tuple(_round_trip(tmp_path, robot=rig).robot.robot.default_root_pos) == (0.0, 0.0, 1.0)


# ── validation ─────────────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "make",
    [
        lambda: _g1_wuji(left_hand_name="hand", right_hand_name="hand"),
        lambda: _g1_wuji(left_hand_name="robot"),
        lambda: _g1_wuji(right_hand_name="robot"),
        lambda: Droid(gripper_name="robot"),
        lambda: FrankaXHand(hand_name="robot"),
    ],
)
def test_colliding_entity_names_fail_at_construction(make):
    """Two bodies under one name would silently collapse the rig; arm rigs used to fail only at first use."""
    with pytest.raises(ValueError, match="same name"):
        make()


def test_colliding_entity_names_set_after_construction_are_still_caught():
    """Options do not validate assignment, so the guard lives in ``entity_options()`` as well."""
    rig = OliWuji()
    rig.left_hand_name = "right_hand"
    for read in (rig.entity_options, lambda: rig.entity_names, rig.expand, lambda: SceneOptions(robot=rig)):
        with pytest.raises(ValueError, match="same name"):
            read()


@pytest.mark.parametrize(
    "make",
    [
        lambda: Droid(gripper=Robotiq2f85(is_fixed_base=True)),
        lambda: _g1_wuji(left_hand=WujiHand2Beta2WithMount_L(is_fixed_base=True)),
        lambda: _g1_wuji(right_hand=WujiHand2Beta2WithMount_R(is_fixed_base=True)),
    ],
    ids=["manipulator", "humanoid-left", "humanoid-right"],
)
def test_a_welded_end_effector_may_not_also_be_world_pinned(make):
    """It would nail the end-effector to the origin while the rig moves away; the scene builds regardless."""
    with pytest.raises(ValueError, match="is_fixed_base"):
        make()


@pytest.mark.parametrize("make", [lambda: Droid(base_poss=(0.0, 0.0, 0.4)), lambda: _g1_wuji(base_poss=(0, 0, 0.8))])
def test_a_misspelled_field_is_rejected(make):
    with pytest.raises(ValueError):
        make()


def test_the_private_arm_base_is_neither_constructible_nor_exported():
    """It declares no ``entity_parts()``; the underscore and the missing export say the same thing."""
    for package in (robots, composite):
        assert "_ArmCompositeOptions" not in package.__all__ and not hasattr(package, "ArmCompositeOptions")
    with pytest.raises(NotImplementedError, match="entity_parts"):
        _ArmCompositeOptions(arm=FrankaResearch3(), ee_link="attachment")


def test_overriding_entity_options_instead_of_entity_parts_fails_at_class_definition():
    """The old extension point would bypass the duplicate-name validation."""
    with pytest.raises(TypeError, match=r"must implement entity_parts\(\), not entity_options\(\)"):

        class LegacyComposite(CompositeEntityOptions):
            def entity_options(self):
                return {}


# ── weld_quat ──────────────────────────────────────────────────────────────────────────────────────


def test_weld_quat_is_the_identity_map_for_an_eden_default_asset():
    gripper = Robotiq2f85()
    assert not has_frame_alignment(gripper)
    for quat in ((1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), (math.sqrt(0.5), 0.0, math.sqrt(0.5), 0.0)):
        np.testing.assert_allclose(weld_quat(gripper, quat), quat, atol=1e-12)


def test_weld_quat_inverts_an_asset_alignment():
    hand = WujiHand2Beta2WithMount_R()
    assert has_frame_alignment(hand)
    R = gu.quat_to_R(np.asarray(weld_quat(hand, (1.0, 0.0, 0.0, 0.0))))
    np.testing.assert_allclose(R @ align_up_and_front(hand.up, hand.front), np.eye(3), atol=1e-12)


def test_weld_quat_tolerates_a_scaled_quat_and_names_a_zero_one():
    """Public, so it cannot assume ``UnitVec4FType``; zero would reach Genesis as a bare ZeroDivisionError."""
    hand = WujiHand2Beta2WithMount_R()
    np.testing.assert_allclose(weld_quat(hand, (3.0, 0.0, 0.0, 0.0)), weld_quat(hand, (1.0, 0.0, 0.0, 0.0)), atol=1e-12)
    np.testing.assert_allclose(weld_quat(hand, (0.0, 0.6, 0.0, 0.0)), weld_quat(hand, (0.0, 1.0, 0.0, 0.0)), atol=1e-12)
    with pytest.raises(ValueError, match="non-zero quaternion"):
        weld_quat(hand, (0.0, 0.0, 0.0, 0.0))


@pytest.mark.parametrize("name, cls", list(COMPOSITE_ROBOT_REGISTRY))
def test_every_registered_rig_welds_its_end_effector_at_the_intended_frame(name, cls):
    """Routing every weld through ``weld_quat`` moved nothing: composing weld and alignment gives the mount."""
    rig = cls()
    welded = {n: sub for n, sub in rig.entity_options().items() if n in rig.attachments()}
    for child, spec in rig.attachments().items():
        if isinstance(rig, HumanoidOptions):
            assert len(spec) == 4, f"{name}/{child}: a humanoid always states its mount"
            # A side's own mount wins over the shared one (G1Inspire's wrists are not mirror images).
            side_quat = rig.left_mount_quat if child == rig.left_hand_name else rig.right_mount_quat
            want = rig.mount_quat if side_quat is None else side_quat
        else:
            # An unaligned end-effector takes the 2-tuple unless the rig states a mount.
            assert (len(spec) == 4) == (rig.attach_offset is not None or rig.attach_quat is not None), spec
            if len(spec) == 2:
                continue
            want = rig.attach_quat if rig.attach_quat is not None else welded[child].default_root_quat
        align = align_up_and_front(welded[child].up, welded[child].front)
        np.testing.assert_allclose(gu.quat_to_R(np.asarray(spec[3])) @ align, gu.quat_to_R(np.asarray(want)), atol=1e-9)


# ── manipulator tool point and grip width ──────────────────────────────────────────────────────────


def test_the_tool_point_and_stroke_come_from_the_gripper_and_follow_a_stated_mount():
    panda, droid = PandaFrankaHand(), Droid()
    assert (panda.tcp_offset, panda.stroke) == (pytest.approx((0.0, 0.0, 0.1034)), 0.08)
    assert (droid.tcp_offset, droid.stroke) == (pytest.approx((0.0, 0.0, 0.1413)), 0.085)
    # 90 deg about x sends the gripper's +z to -y.
    mounted = Droid(attach_offset=(0.0, 0.0, 0.05), attach_quat=(math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0))
    assert mounted.tcp_offset == pytest.approx((0.0, -0.1413, 0.05), abs=1e-6)


def test_a_gripper_that_declares_no_tool_point_raises_naming_it():
    rig = Droid(gripper=UR5e(is_fixed_base=False))  # an arm standing in for a gripper
    with pytest.raises(ValueError, match="UR5e declares no tcp_offset"):
        rig.tcp_offset
    with pytest.raises(ValueError, match="UR5e declares no stroke"):
        rig.stroke


def test_hold_width_interpolates_open_to_close_and_clamps_past_the_stroke():
    rig = PandaFrankaHand()
    assert rig.hold_width(rig.stroke, clearance=0.0) == rig.gripper_open
    assert rig.hold_width(0.0, clearance=0.0) == rig.gripper_close
    assert rig.hold_width(0.04, clearance=0.0) == {
        "finger_joint1": pytest.approx(0.02),
        "finger_joint2": pytest.approx(0.02),
    }
    assert rig.hold_width(1.0) == rig.gripper_open  # clamps, never extrapolates
    assert 0.0 < Droid().hold_width(0.04, clearance=0.0)["left_driver_joint"] < 0.8


def test_preset_hold_writes_the_pose_and_refuses_a_width_it_cannot_close_on():
    """``hold_width`` may clamp a query; a clamped preset would start the episode holding nothing."""
    rig = PandaFrankaHand()
    rig.preset_hold(0.04)
    assert rig.gripper.default_dofs_pos["finger_joint1"] == pytest.approx(0.022)  # 0.04 / 2 + clearance

    rig = PandaFrankaHand()
    for width, match in [
        (1.0, r"cannot hold a 1\.0 m object"),
        (rig.stroke, r"stroke is 0\.08 m"),  # exactly the stroke still needs 2 x clearance on top
        (-0.01, r"needs the width of an object to hold, got -0\.01 m"),
        (0, r"needs the width of an object to hold, got 0 m"),  # an unset width constant, not "shut"
    ]:
        with pytest.raises(ValueError, match=match):
            rig.preset_hold(width)
    assert rig.gripper.default_dofs_pos == PandaFrankaHand().gripper.default_dofs_pos, "a refusal wrote a pose"

    rig.preset_hold(rig.stroke - 2.0 * 0.002)  # the widest width that fits, and clearance moves it
    assert rig.gripper.default_dofs_pos["finger_joint1"] == pytest.approx(rig.gripper_open["finger_joint1"])
    PandaFrankaHand().preset_hold(rig.stroke, clearance=0.0)
