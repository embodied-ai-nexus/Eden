"""``JacobianIK`` against the real G1 MJCF."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from eden.extensions.motion_optimization.tcrs.ik import JacobianIK
from eden.options.extensions.motion_optimization import TCRSRobotOptions


@pytest.mark.slow
def test_converges_at_once_on_foot_targets_the_seed_already_satisfies(g1_mjcf):
    """FK a bent-knee pose, then ask IK for those exact foot poses from the same seed."""
    spec = TCRSRobotOptions()
    model = mujoco.MjModel.from_xml_path(str(g1_mjcf))
    data = mujoco.MjData(model)
    ik = JacobianIK.from_config(model, spec)

    qpos = np.zeros(model.nq)
    qpos[0:3] = [0.0, 0.0, 0.70]
    qpos[3] = 1.0  # identity root quat (wxyz)
    for side in ("left", "right"):
        for joint, value in (("hip_pitch", -0.4), ("knee", 0.8), ("ankle_pitch", -0.4)):
            qpos[model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_{joint}_joint")]] = (
                value
            )
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    feet = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, link) for link in (spec.left_foot_link, spec.right_foot_link)
    ]

    result = ik.solve_with_root(
        data,
        target_root_pos=qpos[0:3],
        target_root_quat=qpos[3:7],
        target_lf_pos=data.xpos[feet[0]].copy(),
        target_lf_quat=data.xquat[feet[0]].copy(),
        target_rf_pos=data.xpos[feet[1]].copy(),
        target_rf_quat=data.xquat[feet[1]].copy(),
        fixed_upper_body_qpos=qpos[7:],
        initial_leg_qpos=ik.extract_leg_qpos(qpos),
    )
    assert result.converged
    assert result.iterations <= 2
    assert result.left_pos_err < 1e-3 and result.right_pos_err < 1e-3  # sub-mm
