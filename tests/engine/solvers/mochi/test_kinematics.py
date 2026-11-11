"""Kinematics helpers of the MochiSolver's Newton iterate."""

import genesis as gs
import numpy as np
import quadrants as qd
from genesis.utils import array_class

from eden.engine.solvers.mochi.kinematics import func_joint_q_end

# A free root, then a hinge, a slide and a ball.
_EVERY_JOINT_TYPE_MJCF = """
<mujoco>
  <worldbody>
    <body name="root" pos="0 0 1">
      <freejoint/>
      <geom type="box" size="0.1 0.1 0.1"/>
      <body name="hinge" pos="0 0 0.3">
        <joint name="hinge" type="hinge" axis="0 1 0"/>
        <geom type="box" size="0.05 0.05 0.1"/>
        <body name="slide" pos="0 0 0.3">
          <joint name="slide" type="slide" axis="0 0 1"/>
          <geom type="box" size="0.05 0.05 0.1"/>
          <body name="ball" pos="0 0 0.3">
            <joint name="ball" type="ball"/>
            <geom type="box" size="0.05 0.05 0.1"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


@qd.kernel
def _kernel_joints_q_end(dyn_info: array_class.DynInfo, q_end: qd.types.ndarray()):
    for i_j in range(q_end.shape[0]):
        q_end[i_j] = func_joint_q_end(dyn_info.joints.q_start[i_j], dyn_info.joints.type[i_j])


def test_the_coordinates_of_a_joint_end_where_genesis_lays_them_out(tmp_path):
    path = tmp_path / "every_joint_type.xml"
    path.write_text(_EVERY_JOINT_TYPE_MJCF)
    scene = gs.Scene(show_viewer=False)
    scene.add_entity(gs.morphs.MJCF(file=str(path)))
    # A fixed base is welded to the world by a fixed joint
    scene.add_entity(gs.morphs.Box(size=(0.1, 0.1, 0.1), pos=(1.0, 0.0, 0.05), fixed=True))
    scene.build()
    joints = scene.rigid_solver.joints
    assert {joint.type for joint in joints} == set(gs.JOINT_TYPE)

    q_end = np.zeros(len(joints), dtype=gs.np_int)
    _kernel_joints_q_end(scene.rigid_solver.dyn_info, q_end)
    assert q_end.tolist() == [joint.q_end for joint in joints]
