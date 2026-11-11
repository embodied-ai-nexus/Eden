"""Tests for ``eden.extensions.retargeting.otretarget.stance.FloorStance`` on a G1 that demonstrates to itself."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("proxsuite")  # the QP solver; in the ``ot-retargeter`` extra
pytest.importorskip("threadpoolctl")

from eden.extensions.retargeting.otretarget import OTRetargeter
from eden.extensions.retargeting.otretarget.retargeter import Targets
from tests.extensions.retargeting.otretarget.conftest import FEET, identity_options, lowest_foot_point


def _narrowed(human, factor):
    """The demonstration squeezed sideways (y) by ``factor``: a demonstrator with narrower hips and stance."""
    scale = np.array([1.0, factor, 1.0])
    return replace(
        human,
        vertices=(human.vertices * scale).astype(np.float32),
        joint_pos=human.joint_pos * scale,
        rest_vertices=human.rest_vertices * scale,
        rest_joint_pos=human.rest_joint_pos * scale,
    )


def test_footprint_offsets_vanish_for_a_demonstrator_of_the_robots_own_proportions(self_retarget):
    ret, _, _, _ = self_retarget
    labels = np.array(ret._probe_labels)
    for foot in FEET:
        assert np.abs(ret.addon("floor_stance").offset[labels == foot].mean(axis=0)).max() < 0.005


@pytest.mark.parametrize(("offset", "width"), [(True, 1.0), (False, 0.6)])
def test_feet_land_at_the_robots_own_stance_width(env, kin, self_retarget, offset, width):
    # A demonstrator 40% narrower: with footprint offsets the robot keeps its own stance width (the floor is
    # translation invariant); without them its feet are pulled in onto the narrower footprints.
    _, human, qs, _ = self_retarget
    ret = OTRetargeter(env, OTRetargeter.configure(**identity_options(kin, {"footprint_offset": offset})))
    result = ret.retarget_motion(_narrowed(human, 0.6), progress=False)
    feet = [kin.link_index(n) for n in FEET]

    def stance(q):
        kin.set_configuration(q)
        pos, _ = kin.link_poses()
        return pos[feet[0], 1] - pos[feet[1], 1]

    np.testing.assert_allclose(stance(result.qpos[-1]), width * stance(qs[-1]), rtol=0.15)


def _turning(human):
    """A narrowed demonstrator whose pelvis turns 90 deg over feet that stay where they are (a spin on the spot)."""
    narrow = _narrowed(human, 0.6)
    pelvis = narrow.joint_index("pelvis")
    joint_rot = narrow.joint_rot.copy()
    for t, yaw in enumerate(np.linspace(0.0, 0.5 * np.pi, narrow.num_frames)):
        c, s = np.cos(yaw), np.sin(yaw)
        joint_rot[t, pelvis] = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]) @ joint_rot[t, pelvis]
    return replace(narrow, joint_rot=joint_rot)


def test_a_planted_foot_keeps_its_footprint_while_the_pelvis_turns(env, kin, self_retarget):
    # The footprint offset of a probe in floor contact is the one it landed with; re-aimed at the pelvis heading every
    # frame, it swept the stance foot of a spin round a circle as wide as the offset (skate).
    turning = _turning(self_retarget[1])

    def floor_witness(offset):
        ret = OTRetargeter(
            env, OTRetargeter.configure(**identity_options(kin, {"footprint_offset": offset, "stick": False}))
        )
        ret._setup(turning, [])
        floor = [
            ret._frame_inputs(turning, t, None, None, None, forward=True).robot_t[0] for t in range(turning.num_frames)
        ]
        return np.stack([f.witness for f in floor]), np.stack([f.distance for f in floor]), ret

    with_offset, distance, ret = floor_witness(True)
    applied = with_offset - floor_witness(False)[0]  # (T, probes, 3) offset added to each floor witness
    planted = (distance < ret.addon("floor_stance").footprint_contact).all(axis=0)
    assert planted.sum() > 20
    assert np.abs(applied[0, planted]).max() > 0.01  # the narrowed stance does shift the footprints
    np.testing.assert_allclose(applied[-1, planted], applied[0, planted], atol=1e-9)


def test_a_frame_read_out_of_order_does_not_advance_the_footprint_state(env, kin, self_retarget):
    # A stage that re-solves an earlier frame (the hand-grasp re-approach goes back up to 45 frames) reads that frame's
    # targets; the held footprints of the forward pass must not move to it, or the next frame compares its contact
    # against a frame long past.
    turning = _turning(self_retarget[1])
    ret = OTRetargeter(env, OTRetargeter.configure(**identity_options(kin, {"stick": False})))
    ret._setup(turning, [])
    stance = ret.addon("floor_stance")
    for t in range(turning.num_frames):
        ret._frame_inputs(turning, t, None, None, None, forward=True)
    held = [a.copy() for a in stance._footprint]
    ret._frame_inputs(turning, 0, None, None, None)
    for a, b in zip(held, stance._footprint):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize(("support", "floats"), [(0.01, False), (0.0, True)])
def test_floor_support_keeps_a_demonstrated_stance_on_the_floor(env, kin, self_retarget, support, floats):
    # Strong position style targets 6 cm above the demonstration pull the whole robot up while its soles stay
    # demonstrated on the floor. The contact residuals alone give way (the feet float); the support rows hold
    # each sole within 1 cm of the floor.
    _, human, _, _ = self_retarget
    lifted = replace(human, joint_pos=human.joint_pos + np.array([0.0, 0.0, 0.06]))
    opts = identity_options(kin, {"support_height": support}, style_pos_weight=30.0)
    result = OTRetargeter(env, OTRetargeter.configure(**opts)).retarget_motion(lifted, progress=False)
    lowest = [lowest_foot_point(kin, q) for q in result.qpos]
    assert (np.max(lowest) > 0.02) == floats, lowest


@pytest.mark.parametrize(("stick", "slides"), [(True, False), (False, True)])
def test_planted_feet_do_not_slide_under_a_sideways_pull(env, kin, self_retarget, stick, slides):
    # Strong position style targets drift 8 cm sideways (y) over the squat while the demonstrated feet keep their y
    # (the squat moves them only fore-aft, which the planted feet follow). The floor witnesses alone give way and the
    # soles slide sideways after the body; the stance rows hold them.
    _, human, _, _ = self_retarget
    drift = np.linspace(0.0, 0.08, human.num_frames)[:, None, None] * np.array([0.0, 1.0, 0.0])
    pulled = replace(human, joint_pos=human.joint_pos + drift)
    opts = identity_options(kin, {"stick": stick}, style_pos_weight=30.0)
    result = OTRetargeter(env, OTRetargeter.configure(**opts)).retarget_motion(pulled, progress=False)
    feet = [kin.link_index(n) for n in FEET]
    xy = []
    for q in result.qpos:
        kin.set_configuration(q)
        pos, _ = kin.link_poses()
        xy.append(pos[feet, 1])
    slide = np.abs(np.array(xy)[-1] - np.array(xy)[1]).max()
    assert (slide > 0.01) == slides, slide
    if not slides:
        assert slide < 0.005, slide


def test_a_lifted_foot_hands_its_offset_back_linearly_over_stick_release(env, kin, self_retarget):
    # After lift-off the floor witnesses return from the latched offset to the bare footprints over ``stick_release``
    # of human foot travel. The fade is a function of the travel so far; applied to an offset stored already faded,
    # it compounded (0.8, 0.48, 0.19, ... of the offset) and released the foot early.
    _, human, qs, _ = self_retarget
    ret = OTRetargeter(env, OTRetargeter.configure(**identity_options(kin, {"stick": True, "stick_release": 0.1})))
    ret._setup(human, [])
    stance = ret.addon("floor_stance")
    n = len(ret.src)
    stride = np.arange(human.num_frames)[:, None, None] * np.array([0.02, 0.0, 0.0])  # 2 cm per frame, off the floor
    swinging = replace(human, vertices=(human.vertices[:1].astype(np.float64) + stride).astype(np.float32))

    def off_the_floor():
        return Targets(np.full(n, np.inf), np.zeros((n, 3)), np.zeros((n, 3)))

    stance._anchors(swinging, 0, off_the_floor(), qs[0])  # the previous frame of the first one read below
    offset = np.array([0.08, -0.04])
    stance._carry = (np.tile(offset, (n, 1)), np.zeros(n), np.ones(n, dtype=bool))  # lifted off with this offset
    for t in range(1, human.num_frames):  # released after 5 frames (10 cm), held at the footprints after
        targets = off_the_floor()
        stance._anchors(swinging, t, targets, qs[0])
        np.testing.assert_allclose(targets.witness[:, :2], np.tile(max(0.0, 1.0 - 0.2 * t) * offset, (n, 1)), atol=1e-6)


@pytest.mark.parametrize(
    ("override", "stance", "match"),
    [
        (dict(ground=False), {}, "no floor channel"),
        ({}, {"stick_anchors": 0}, "stick_anchors"),
        ({}, {"support_slack": -0.01}, "support_slack"),
    ],
)
def test_floor_stance_misconfiguration_is_refused_at_construction(env, kin, override, stance, match):
    with pytest.raises(ValueError, match=match):
        OTRetargeter(env, OTRetargeter.configure(**identity_options(kin, stance, **override)))
