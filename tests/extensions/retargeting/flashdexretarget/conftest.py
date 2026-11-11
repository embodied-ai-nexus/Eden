"""Shared fixtures: clips made from the robot hand's own kinematics (the synthetic MANO model and box meshes come
from ``tests/datasets/conftest.py``).
"""

from __future__ import annotations

import mujoco
import numpy as np

from eden.extensions.retargeting.flashdexretarget import clips as C
from eden.extensions.retargeting.flashdexretarget.ik import TIPS
from eden.extensions.retargeting.flashdexretarget.scene import robot_xml
from tests.datasets.conftest import mano_dir, mano_template, pose_track, write_box  # noqa: F401 -- fixtures, helpers


def self_fk_clip(T: int = 24, fps: float = 60.0, box_pos=(0.6, 0.6, 0.05)) -> tuple[dict, np.ndarray]:
    """A right-hand clip whose MANO wrist and fingertips are the XHand's own palm and fingertip sites along a
    known smooth qpos trajectory, so the IK can reach every target exactly; the object is a box far from the hand.

    Returns the clip and the (T, 18) qpos that produced it.
    """
    m = mujoco.MjModel.from_xml_path(str(robot_xml("xhand", ("right",))))
    d = mujoco.MjData(m)
    lo, hi = m.jnt_range[:, 0], m.jnt_range[:, 1]
    s = np.linspace(0.0, 1.0, T)[:, None]
    q0 = np.zeros(m.nq)
    fingers = np.array(["forearm" not in m.joint(j).name for j in range(m.njnt)])
    q = np.repeat(q0[None], T, 0)
    q[:, 0] += 0.05 * s[:, 0]  # the wrist slides 5 cm
    mid = np.clip(0.0, lo, hi) + 0.3 * (np.clip(hi, -2, 2) - np.clip(0.0, lo, hi))
    q[:, fingers] = np.clip(0.0, lo, hi)[fingers] + s * (mid - np.clip(0.0, lo, hi))[fingers]
    palm = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "right_palm")
    tips = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f"right_{t}") for t in TIPS]
    wrist_pos, wrist_rot, tip_pos = [], [], []
    for t in range(T):
        d.qpos[:] = q[t]
        mujoco.mj_kinematics(m, d)
        wrist_pos.append(d.site_xpos[palm].copy())
        wrist_rot.append(d.site_xmat[palm].reshape(3, 3).copy())
        tip_pos.append(d.site_xpos[tips].copy())
    wrist_pos, tip_pos = np.asarray(wrist_pos), np.asarray(tip_pos)
    joints = np.repeat(wrist_pos[:, None], len(C.MANO_JOINT_NAMES), 1)
    for k, finger in enumerate(C.FINGERS):
        joints[:, C.MANO_JOINT_NAMES.index(f"{finger}_tip")] = tip_pos[:, k]
    clip = {
        "fps": np.float32(fps),
        "mano_right_wrist_pos": wrist_pos.astype(np.float32),
        "mano_right_wrist_rot": np.asarray(wrist_rot, dtype=np.float32),
        "mano_right_joints": joints.astype(np.float32),
        "obj_right_pose": pose_track(np.tile(box_pos, (T, 1))).astype(np.float32),
        "obj_right_id": np.array("box"),
    }
    return clip, q
