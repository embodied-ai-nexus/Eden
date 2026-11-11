"""Synthetic demo clip for the TCRS examples and tests (numpy-only).

Lives in the package (not the example scripts) so the recipes and tests can
share one generator instead of importlib/sys.path hacks on ``examples/``.
"""

from __future__ import annotations

import numpy as np


def make_demo_clip(num_frames: int = 100, fps: float = 50.0):
    """Synthetic straight-line G1 walk along +x (Eden dofs/links order).

    Feet alternate 1.4 Hz strides with a 6 cm ankle clearance on flat ground;
    hips/knees are roughly interpolated so leg-geometry helpers see sane
    capsule lengths. Returns ``(fps, dofs_pos, links_pos_w, links_quat_w)``.
    """
    from eden.options.robots.unitree_g1 import G1_DOFS29_NAME, G1_DOFS29_TRACKED_LINKS_NAME

    dofs, links = list(G1_DOFS29_NAME), list(G1_DOFS29_TRACKED_LINKS_NAME)
    t = np.arange(num_frames) / fps
    phase = 2 * np.pi * 1.4 * t
    pelvis = np.stack([0.45 * t, np.zeros(num_frames), 0.79 + 0.01 * np.sin(2 * phase)], axis=1)

    def foot(sign, off):
        p = phase + off
        swing = np.clip(np.sin(p), 0.0, 1.0)  # > 0 during the swing half of the cycle
        z = 0.06 + 0.08 * swing * np.sin(np.clip(p % (2 * np.pi), 0, np.pi))
        z = np.where(np.sin(p) <= 0, 0.06, z)
        return np.stack([0.45 * t + 0.10 * np.sin(p), np.full(num_frames, sign * 0.12), z], axis=1)

    lfoot, rfoot = foot(+1, 0.0), foot(-1, np.pi)
    links_pos = np.repeat(pelvis[:, None, :], len(links), axis=1)
    links_pos[:, links.index("left_ankle_roll_link")] = lfoot
    links_pos[:, links.index("right_ankle_roll_link")] = rfoot
    for side, f in (("left", lfoot), ("right", rfoot)):
        hip = pelvis + np.array([0.0, (1 if side == "left" else -1) * 0.06, -0.17])
        links_pos[:, links.index(f"{side}_hip_pitch_link")] = hip
        links_pos[:, links.index(f"{side}_knee_link")] = 0.5 * (hip + f) + np.array([0.02, 0.0, 0.0])
    links_quat = np.zeros((num_frames, len(links), 4))
    links_quat[..., 0] = 1.0
    dofs_pos = np.zeros((num_frames, len(dofs)))
    for side, off in (("left", 0.0), ("right", np.pi)):
        dofs_pos[:, dofs.index(f"{side}_knee_joint")] = 0.4 + 0.1 * np.sin(phase + off)
        dofs_pos[:, dofs.index(f"{side}_hip_pitch_joint")] = -0.2 - 0.15 * np.sin(phase + off)
        dofs_pos[:, dofs.index(f"{side}_ankle_pitch_joint")] = -0.2
    return fps, dofs_pos, links_pos, links_quat
