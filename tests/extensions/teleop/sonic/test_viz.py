"""``eden.extensions.teleop.sonic.viz``: the operator visualizer, off-screen."""

from __future__ import annotations

import numpy as np
import pytest


def test_visualizer_offscreen_smoke():
    pytest.importorskip("pyvista")
    from eden.extensions.teleop.sonic.viz import Vr3PtPoseVisualizer

    try:
        viz = Vr3PtPoseVisualizer(with_g1_robot=True, enable_waist_tracking=True, enable_smpl_vis=True, off_screen=True)
    except (FileNotFoundError, RuntimeError) as e:
        pytest.skip(f"viz unavailable in this environment: {e}")

    pose = np.zeros((3, 7))
    pose[:, 3] = 1.0
    pose[0, :3] = [0.3, 0.15, 0.1]
    pose[1, :3] = [0.3, -0.15, 0.1]
    pose[2, :3] = [0.0, 0.0, 0.4]
    joints = np.random.default_rng(0).normal(scale=0.3, size=(24, 3))
    viz.on_frame(pose, joints)
    viz.on_frame(pose, None)
    viz.close()
