"""Operator visualization: calibrated VR 3-point pose, G1 mesh, SMPL body (PyVista).

Port of the reference VR3Pt visualizer's rendered content, in one window:

* the three calibrated keypoints (left wrist light green, right wrist light
  blue, head orange) as balls with RGB frame triads,
* the synthesized head chain (origin -> torso offset -> neck along its local Z),
* a semi-transparent G1 mesh at the origin for spatial reference (geometry
  from the same MJCF asset the calibration FK loads); with waist tracking
  the three waist joints follow the calibrated neck orientation,
* optionally the 24 SMPL joints (lower body blue, upper red) with bones,
  anchored beside the robot — a pre-engage check that the avatar mirrors
  the operator.

Requires ``pyvista`` (the ``sonic-viz`` extra). The window renders inline on
the manager thread, exactly like the reference (visualization costs stream
latency; run without ``--vis-vr3pt`` for the tightest loop).
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as sRot

from eden.extensions.teleop.sonic.three_point import NECK_LINK_LENGTH, TORSO_LINK_OFFSET_Z
from eden.utils.distribution import install_hint

VR_BALL_COLORS = ("lightgreen", "lightblue", "orange")  # left wrist, right wrist, head
SMPL_LOWER_BODY = frozenset({0, 1, 2, 4, 5, 7, 8, 10, 11})
SMPL_PARENTS_VIS = (-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21)
_SMPL_ANCHOR = np.array([-0.3, 0.0, 0.0])  # skeleton stands beside the robot


def _pose_matrix(pos: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
    mat = np.eye(4)
    mat[:3, :3] = sRot.from_quat(quat_wxyz, scalar_first=True).as_matrix()
    mat[:3, 3] = pos
    return mat


class Vr3PtPoseVisualizer:
    """One interactive window updated once per processed device frame."""

    def __init__(
        self,
        with_g1_robot: bool = True,
        enable_waist_tracking: bool = False,
        enable_smpl_vis: bool = False,
        axis_length: float = 0.08,
        ball_radius: float = 0.015,
        robot_opacity: float = 0.4,
        off_screen: bool = False,
    ) -> None:
        try:
            import pyvista as pv
        except ImportError as exc:
            raise ImportError(f"--vis-vr3pt needs pyvista: {install_hint('sonic-viz')}") from exc
        self._pv = pv
        self._waist_tracking = enable_waist_tracking
        self._smpl_vis = enable_smpl_vis

        self.plotter = pv.Plotter(window_size=(1100, 850), off_screen=off_screen)
        self.plotter.add_axes()
        self.plotter.add_mesh(pv.Plane(i_size=2.0, j_size=2.0), color="lightgray", opacity=0.3)
        for axis, color in zip(np.eye(3), ("red", "green", "blue")):
            self.plotter.add_mesh(pv.Line((0, 0, 0), tuple(0.2 * axis)), color=color, line_width=4)

        # VR keypoints: ball + RGB triad each, positioned via the actor matrix.
        self._kp_actors = []
        for color in VR_BALL_COLORS:
            ball = self.plotter.add_mesh(pv.Sphere(radius=ball_radius), color=color)
            triad = []
            for axis, axis_color in zip(np.eye(3), ("red", "green", "blue")):
                actor = self.plotter.add_mesh(
                    pv.Line((0, 0, 0), tuple(axis_length * axis)), color=axis_color, line_width=3
                )
                triad.append(actor)
            self._kp_actors.append((ball, triad))

        # Synthesized head chain: origin -> torso offset -> neck.
        self._chain = pv.PolyData(np.zeros((3, 3)), lines=np.array([2, 0, 1, 2, 1, 2]), force_float=False)
        self.plotter.add_mesh(self._chain, color="orange", line_width=2)

        self._robot = _G1MeshView(self.plotter, robot_opacity) if with_g1_robot else None

        self._smpl_points = None
        if enable_smpl_vis:
            self._smpl_spheres = []
            for i in range(24):
                color = "steelblue" if i in SMPL_LOWER_BODY else "indianred"
                self._smpl_spheres.append(self.plotter.add_mesh(pv.Sphere(radius=0.02), color=color))
            lines = []
            for i in range(1, 24):
                lines.extend([2, SMPL_PARENTS_VIS[i], i])
            self._smpl_bones = pv.PolyData(np.zeros((24, 3)), lines=np.array(lines), force_float=False)
            self.plotter.add_mesh(self._smpl_bones, color="gray", line_width=2)

        self.plotter.show(interactive_update=True, auto_close=False)

    def on_frame(self, vr_3pt_pose: np.ndarray, smpl_joints_local: np.ndarray | None) -> None:
        """Update the scene for one calibrated ``(3, 7)`` pose (+ optional SMPL joints)."""
        for row, (ball, triad) in enumerate(self._kp_actors):
            mat = _pose_matrix(vr_3pt_pose[row, :3], vr_3pt_pose[row, 3:])
            ball.user_matrix = mat
            for actor in triad:
                actor.user_matrix = mat

        neck_z = sRot.from_quat(vr_3pt_pose[2, 3:], scalar_first=True).apply([0.0, 0.0, 1.0])
        torso = np.array([0.0, 0.0, TORSO_LINK_OFFSET_Z])
        self._chain.points = np.stack([np.zeros(3), torso, torso + NECK_LINK_LENGTH * neck_z])

        if self._robot is not None and self._waist_tracking:
            # Waist follows the calibrated neck orientation. as_euler("ZYX")
            # returns [z, y, x] = [yaw, pitch, roll]; the waist joints are
            # yaw (q12), roll (q13), pitch (q14).
            yaw, pitch, roll = sRot.from_quat(vr_3pt_pose[2, 3:], scalar_first=True).as_euler("ZYX")
            q = np.zeros(29)
            q[12], q[13], q[14] = yaw, roll, pitch
            self._robot.set_pose(q)

        if self._smpl_vis and smpl_joints_local is not None:
            points = np.asarray(smpl_joints_local) + _SMPL_ANCHOR
            for i, sphere in enumerate(self._smpl_spheres):
                mat = np.eye(4)
                mat[:3, 3] = points[i]
                sphere.user_matrix = mat
            self._smpl_bones.points = points

        self.plotter.update()

    def close(self) -> None:
        self.plotter.close()


class _G1MeshView:
    """Semi-transparent G1 at the origin, geometry from the calibration MJCF."""

    def __init__(self, plotter, opacity: float) -> None:
        import mujoco

        from eden.options.robots.unitree_g1 import G1Dofs29Dex3

        self._mujoco = mujoco
        self._model = mujoco.MjModel.from_xml_path(G1Dofs29Dex3().fetch_asset())
        self._data = mujoco.MjData(self._model)

        import pyvista

        self._actors = []  # (actor, body_id, T_geom_local)
        for geom_id in range(self._model.ngeom):
            if self._model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_MESH:
                continue
            mesh_id = int(self._model.geom_dataid[geom_id])
            vert_start = int(self._model.mesh_vertadr[mesh_id])
            vert_num = int(self._model.mesh_vertnum[mesh_id])
            face_start = int(self._model.mesh_faceadr[mesh_id])
            face_num = int(self._model.mesh_facenum[mesh_id])
            vertices = self._model.mesh_vert[vert_start : vert_start + vert_num]
            faces = self._model.mesh_face[face_start : face_start + face_num]
            cells = np.hstack([np.full((face_num, 1), 3), faces]).ravel()
            poly = pyvista.PolyData(vertices.copy(), cells)
            actor = plotter.add_mesh(poly, color="silver", opacity=opacity)
            t_local = np.eye(4)
            t_local[:3, :3] = sRot.from_quat(self._model.geom_quat[geom_id], scalar_first=True).as_matrix()
            t_local[:3, 3] = self._model.geom_pos[geom_id]
            self._actors.append((actor, int(self._model.geom_bodyid[geom_id]), t_local))

        from eden.options.robots.unitree_g1 import G1_DOFS29_NAME

        self._joint_qpos_adr = []
        for name in G1_DOFS29_NAME:
            joint_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, name)
            self._joint_qpos_adr.append(int(self._model.jnt_qposadr[joint_id]))

        self.set_pose(np.zeros(29))

    def set_pose(self, q_mj29: np.ndarray) -> None:
        self._data.qpos[:] = 0.0
        self._data.qpos[3] = 1.0  # root at the origin, identity orientation
        for adr, value in zip(self._joint_qpos_adr, np.asarray(q_mj29, dtype=np.float64)):
            self._data.qpos[adr] = value
        self._mujoco.mj_kinematics(self._model, self._data)

        for actor, body_id, t_local in self._actors:
            t_body = np.eye(4)
            t_body[:3, :3] = np.asarray(self._data.xmat[body_id]).reshape(3, 3)
            t_body[:3, 3] = self._data.xpos[body_id]
            actor.user_matrix = t_body @ t_local
