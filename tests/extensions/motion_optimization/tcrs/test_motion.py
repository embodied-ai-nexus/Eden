"""``MotionClip``: foot-contact detection on the synthetic G1 walk."""

from __future__ import annotations

import pytest

from eden.extensions.motion_optimization.tcrs.demo import make_demo_clip
from eden.extensions.motion_optimization.tcrs.motion import MotionClip
from eden.options.extensions.motion_optimization import TCRSRobotOptions


def test_contacts_alternate_on_a_walk_and_the_pelvis_height_is_read_off_the_stance():
    fps, dofs_pos, links_pos, links_quat = make_demo_clip()
    clip = MotionClip(
        robot_spec=TCRSRobotOptions(),
        fps=fps,
        dofs_pos=dofs_pos,
        links_pos_w=links_pos,
        links_quat_w=links_quat,
        contact_z_threshold=0.06,
        contact_speed_threshold=0.5,
    )
    for mask in (clip.left_contact, clip.right_contact):
        assert 0.2 < mask.mean() < 0.8
    assert clip.pelvis_height_above_foot == pytest.approx(0.73, abs=0.1)
