"""Tests for ``eden.managers.terms.terminations.tracking``: motion-tracking terminations.

The terms compare a built G1 and Panda with a reference written into the env's motion buffer
relative to the entities' real state.
"""

from __future__ import annotations

import math

import pytest
import torch

from eden.managers.termination_manager import TerminationManager, TerminationTermFuncWrapper
from eden.managers.terms.terminations.tracking import (
    BadMotionLinkPosZOnly,
    MotionBasePoseTrackingBreak,
    _adaptive_pos_threshold,
    bad_anchor_ori,
    bad_anchor_pos_z_only,
    bad_base_ori,
    motion_dofs_pos_tracking_error_above,
    motion_ends,
)
from eden.options.managers.terminations import TerminationManagerOptions
from eden.utils.geom import quat_from_euler_xyz, quat_mul
from tests.support.motion import build_tracking_env, set_reference, write_tracking_library

LINKS = ["pelvis", "torso_link"]  # the first two reference links


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    return build_tracking_env(write_tracking_library(tmp_path_factory.mktemp("tracking")))


def _place_robot(env, height: float) -> torch.Tensor:
    """Stand the G1 with its base at ``height``; returns the pelvis and torso heights, ``(N, 2)``."""
    robot = env.entities["robot"]
    robot.set_pos(torch.tensor([[0.0, 0.0, height]] * env.num_envs, device=env.device))
    return robot.get_links_pos(ls_idx_local=robot.find_named_links_idx_local(LINKS, preserve_order=True)[1])[..., 2]


def _reference_heights(env, pelvis_z, torso_z) -> None:
    links = env.motion_manager.data["robot"].links_pos_w.clone()
    links[:, :, 0, 2] = torch.as_tensor(pelvis_z, device=links.device).reshape(-1, 1)
    links[:, :, 1, 2] = torch.as_tensor(torso_z, device=links.device).reshape(-1, 1)
    set_reference(env, "robot", links_pos_w=links)


def test_motion_ends_fires_when_the_current_frame_is_past_the_clip(env):
    validity = torch.ones(4, 9, dtype=torch.bool)
    validity[[1, 3], 0] = False
    set_reference(env, "robot", validity=validity)

    assert motion_ends(env, entity_name="robot").tolist() == [False, True, False, True]


@pytest.mark.parametrize(
    "anchor_z, threshold, down, expected",
    [
        ([0.6, 0.8, 1.5], 0.15, 0.75, [0.15] * 3),  # upright: the configured gate
        ([0.0], 0.15, 0.75, [0.2625]),  # at the floor: threshold * (1 + down)
        ([0.25], 0.15, 0.75, [0.15 * 1.375]),  # halfway down: half the ramp
        ([-0.1, -1.0], 0.1, 0.5, [0.15, 0.15]),  # below the floor saturates
    ],
)
def test_adaptive_threshold_ramps_up_as_the_reference_anchor_drops(anchor_z, threshold, down, expected):
    out = _adaptive_pos_threshold(torch.tensor(anchor_z), threshold, True, down, 0.5)

    torch.testing.assert_close(out, torch.tensor(expected))


def test_adaptive_threshold_off_is_the_static_scalar_and_a_zero_ramp_height_raises():
    assert _adaptive_pos_threshold(torch.tensor([0.0, 1.0]), 0.15, False, 0.75, 0.5) == 0.15
    with pytest.raises(ValueError, match="root_height_threshold"):
        _adaptive_pos_threshold(torch.tensor([0.5]), 0.1, True, 0.5, 0.0)


@pytest.mark.parametrize("adaptive", [False, True], ids=["static", "adaptive-upright"])
def test_anchor_height_error_trips_the_gate_on_an_upright_reference(env, adaptive):
    pelvis = _place_robot(env, 0.8)[:, 0]
    _reference_heights(env, pelvis + 0.2, pelvis + 0.2)

    tripped = bad_anchor_pos_z_only(
        env,
        entity_name="robot",
        anchor_link_name="pelvis",
        threshold=0.15,
        threshold_adaptive=adaptive,
        down_threshold=0.75,
        root_height_threshold=0.5,
    )

    assert tripped.all()


def test_a_crawling_reference_opens_the_anchor_gate(env):
    pelvis = _place_robot(env, 0.2)[:, 0]
    assert ((pelvis > 0.15) & (pelvis < 0.2625)).all()  # between the static and the floor gate
    _reference_heights(env, 0.0, 0.0)

    kwargs = dict(entity_name="robot", anchor_link_name="pelvis", threshold=0.15, down_threshold=0.75)
    assert bad_anchor_pos_z_only(env, threshold_adaptive=False, **kwargs).all()
    assert not bad_anchor_pos_z_only(env, threshold_adaptive=True, root_height_threshold=0.5, **kwargs).any()


def test_anchor_terminations_in_one_termination_pass_trip_what_each_trips_alone(env, install_manager):
    """In a pass, the terms on one anchor link share one read of its state; each keeps its own anchor."""
    heights = _place_robot(env, 0.8)
    _reference_heights(
        env, heights[:, 0] + torch.tensor([0.2, 0.0, 0.2, 0.0]), heights[:, 1] + torch.tensor([0.0, 0.0, 0.2, 0.2])
    )
    quat = torch.zeros_like(env.motion_manager.data["robot"].links_quat_w)
    quat[..., 0] = 1.0
    quat[[0, 3], :, 1] = torch.tensor(
        [math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0]
    )  # the reference torso lies on its side
    set_reference(env, "robot", links_quat_w=quat)

    def term(func, anchor, threshold):
        params = {"entity_name": "robot", "anchor_link_name": anchor, "threshold": threshold}
        return TerminationTermFuncWrapper.configure(func=func, params=params)

    manager = install_manager(
        env,
        "termination_manager",
        TerminationManager(
            env,
            TerminationManagerOptions(
                pelvis_z=term(bad_anchor_pos_z_only, "pelvis", 0.15),
                torso_z=term(bad_anchor_pos_z_only, "torso_link", 0.15),
                torso_ori=term(bad_anchor_ori, "torso_link", 0.5),
            ),
        ),
    )

    manager.compute()

    expected = {
        "pelvis_z": [True, False, True, False],
        "torso_z": [False, False, True, True],
        "torso_ori": [True, False, False, True],
    }
    for name, dones in expected.items():
        assert manager.get_term_dones(name).tolist() == dones, name
        assert manager.get_term(name).compute().tolist() == dones, f"{name} alone"


@pytest.mark.parametrize("tilt_deg, expected", [(40.0, False), (90.0, True)])
def test_anchor_ori_compares_the_unit_gravity_direction(env, tilt_deg, expected):
    """A tilt gap of ``1 - cos(tilt)`` against BeyondMimic's 0.8: 40 deg passes, 90 deg trips.

    With gravity in m/s^2 the gap scales by 9.81 and 40 deg (0.23 -> 2.3) would trip too.
    """
    _place_robot(env, 0.8)
    robot = env.entities["robot"]
    torso_quat = robot.get_links_quat(
        ls_idx_local=robot.find_named_links_idx_local(["torso_link"], preserve_order=True)[1]
    )[:, 0]
    tilt = _quat(0.0, math.radians(tilt_deg), 0.0).to(env.device).expand_as(torso_quat)
    quat = env.motion_manager.data["robot"].links_quat_w.clone()
    quat[:, :, 1] = quat_mul(tilt, torso_quat)[:, None]  # the reference torso: the robot's, tilted in the world
    set_reference(env, "robot", links_quat_w=quat)

    dones = bad_anchor_ori(env, entity_name="robot", anchor_link_name="torso_link", threshold=0.8)
    assert dones.tolist() == [expected] * env.num_envs


def _link_gate(env, adaptive: bool):
    options = BadMotionLinkPosZOnly.configure(
        entity_name="robot",
        anchor_link_name="pelvis",
        links_name=LINKS,
        threshold=0.2,
        threshold_adaptive=adaptive,
        down_threshold=0.75,
        root_height_threshold=0.5,
    )
    term = BadMotionLinkPosZOnly(env, options)
    term.build()
    return term


@pytest.mark.parametrize(
    "height, reference_pelvis, adaptive, expected",
    [
        (0.8, "robot", False, True),  # the torso is 0.25 off: past a 0.2 gate
        (0.8, "robot", True, True),  # an upright anchor keeps the gate at 0.2
        (0.1, "floor", True, False),  # a crawling anchor widens it to 0.35
    ],
    ids=["static", "adaptive-upright", "adaptive-crawl"],
)
def test_any_link_past_the_height_gate_terminates(env, height, reference_pelvis, adaptive, expected):
    pelvis, torso = _place_robot(env, height).unbind(-1)
    _reference_heights(env, pelvis + 0.05 if reference_pelvis == "robot" else 0.0, torso + 0.25)
    if reference_pelvis == "floor":
        assert (pelvis < 0.35).all()

    assert _link_gate(env, adaptive).compute().tolist() == [expected] * 4


@pytest.mark.parametrize(
    "entity, names",
    [
        ("robot", ["left_hip_pitch_joint", "left_hip_roll_joint", "left_knee_joint", "waist_yaw_joint"]),
        ("arm", ["joint1", "joint2", "joint3", "joint4"]),
    ],
    ids=["floating-base", "fixed-base"],
)
def test_dofs_error_reads_each_joint_by_name(env, entity, names):
    ent = env.entities[entity]
    ent.set_dofs_pos(
        torch.tensor([[0.0, 0.1, 0.2, 0.3]] * 4, device=env.device),
        dofs_idx_local=ent.find_named_dofs_idx_local(names, preserve_order=True)[1],
    )
    dofs = env.motion_manager.data[entity].dofs_pos.clone()
    motion_names = env.motion_manager.get_target_dofs_name(entity)
    dofs[:, :, [motion_names.index(n) for n in names]] = 0.5
    set_reference(env, entity, dofs_pos=dofs)

    # max |q - 0.5| is 0.5 (the first joint); mean is 0.35.
    assert motion_dofs_pos_tracking_error_above(env, entity_name=entity, threshold=0.45, dofs_name=names).all()
    assert not motion_dofs_pos_tracking_error_above(
        env, entity_name=entity, threshold=0.4, dofs_name=names, reduction="mean"
    ).any()


def test_dofs_error_on_an_entity_without_dofs_says_to_use_a_link_term(env):
    with pytest.raises(ValueError, match="'object' has no DOFs in the motion reference"):
        motion_dofs_pos_tracking_error_above(env, entity_name="object", threshold=0.1)


def _quat(roll: float, pitch: float, yaw: float) -> torch.Tensor:
    return quat_from_euler_xyz(*(torch.tensor([a]) for a in (roll, pitch, yaw)))[0]


@pytest.mark.parametrize(
    "sim_rpy, ref_rpy, expected",
    [
        ((0.0, 0.0, 0.0), (0.0, 0.0, 2.5), False),  # yaw alone never tilts the gravity vector
        ((0.0, 0.0, 0.0), (0.2, 0.0, 0.0), False),  # |cos 0.2 - 1| = 0.02 < 0.1
        ((0.0, 0.0, 0.0), (0.0, 0.6, 0.0), True),  # |cos 0.6 - 1| = 0.175 > 0.1
        ((0.0, 0.6, 1.0), (0.0, 0.6, -2.0), False),  # the same tilt under a different heading
    ],
)
def test_base_ori_fires_on_the_tilt_gap_and_ignores_heading(env, sim_rpy, ref_rpy, expected):
    obj = env.entities["object"]
    obj.set_quat(_quat(*sim_rpy).repeat(env.num_envs, 1).to(env.device))
    set_reference(env, "object", base_quat_w=_quat(*ref_rpy))

    assert bad_base_ori(env, entity_name="object", threshold=0.1).tolist() == [expected] * env.num_envs


def test_base_ori_needs_a_free_base_in_the_reference(env):
    with pytest.raises(ValueError, match="is_free_floating=False"):
        bad_base_ori(env, entity_name="arm", threshold=0.1)


def _pose_break_gate(env, **params) -> MotionBasePoseTrackingBreak:
    """A built gate on the object, its reference 0.2 m off the object's position and at its orientation."""
    gate = MotionBasePoseTrackingBreak(env, MotionBasePoseTrackingBreak.configure(entity_name="object", **params))
    gate.build()
    obj = env.entities["object"]
    offset = torch.tensor([0.2, 0.0, 0.0], device=env.device)
    # Every frame of the window: (num_envs, 1, ...) broadcasts over the frame axis.
    set_reference(env, "object", base_pos_w=(obj.get_pos() + offset)[:, None], base_quat_w=obj.get_quat()[:, None])
    return gate


def test_the_pose_break_gate_moves_with_set_thresholds_on_the_device(env):
    gate = _pose_break_gate(env, pos_threshold=0.1, rot_threshold=1.0)
    assert gate.compute().all()  # 0.2 m > 0.1 m

    gate.set_thresholds(torch.tensor(0.3, device=env.device), 1.0)

    assert not gate.compute().any()
    assert [float(value) for value in gate.thresholds] == pytest.approx([0.3, 1.0])
    assert (gate.pos_threshold, gate.rot_threshold) == (0.1, 1.0)  # the configured start, untouched


def test_assigning_a_pose_break_threshold_after_build_raises_naming_set_thresholds(env):
    """The gate compares against its device thresholds: an assigned field would be silently ignored."""
    gate = _pose_break_gate(env, pos_threshold=0.1, rot_threshold=1.0)

    for name in ("pos_threshold", "rot_threshold"):
        with pytest.raises(RuntimeError, match=rf"{name} was assigned .*after build.*set_thresholds\("):
            setattr(gate, name, 0.5)
    with pytest.raises(RuntimeError, match=r"pos_threshold was assigned \(a Tensor\)"):  # a tensor by its type
        gate.pos_threshold = torch.tensor(0.5, device=env.device)
    assert (gate.pos_threshold, gate.rot_threshold) == (0.1, 1.0)
    assert gate.compute().all()  # the gate never moved
