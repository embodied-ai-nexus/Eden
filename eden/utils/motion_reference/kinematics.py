"""Batched, differentiable forward kinematics of a floating-base robot built from its MJCF."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeAlias

import mujoco
import numpy as np
import torch

#: A vector or a wxyz quaternion as a tuple of its per-component ``(..., L)`` tensors. The pass keeps this form, which
#: ``torch.compile`` fuses into few kernels, so ``_cross``, ``_rotate`` and ``_mul`` below are written per component
#: instead of calling ``eden.utils.geom`` on stacked ``(..., L, 3)`` / ``(..., L, 4)`` tensors.
#: ``tests/utils/motion_reference/test_kinematics.py`` pins ``_mul`` and ``_rotate`` to ``quat_mul`` and
#: ``transform_by_quat``.
_Components: TypeAlias = tuple[torch.Tensor, ...]


class ForwardKinematics(torch.nn.Module):
    """World-frame poses and twists of selected links from the root pose, root twist and joint state.

    Each link follows MuJoCo's kinematics: the parent frame, the link's fixed offset, then each of its hinges turned
    by ``q - qpos0`` about the hinge axis through the hinge anchor. Velocities are those of the link-frame origin, the
    convention of MuJoCo ``xpos`` and of Genesis ``get_links_vel(ref="link_origin")``. The pass runs on per-component
    ``(..., L)`` tensors, which ``torch.compile`` fuses into few kernels.

    Use it where link states are needed inside a differentiable or compiled computation, or for frames chosen at run
    time. A dataset that precomputes the links of every frame at load uses the simulator's own kinematics
    (``DatasetBase._forward_velocity_kinematics``), which reads any joint type; this module reads a free root and
    hinges only, and raises on anything else.

    Parameters
    ----------
    mjcf_path : str
        The robot MJCF (or URDF). Its first body below the world must carry the free joint; every other body on a path
        to a selected link may carry hinges only.
    joint_names : Sequence[str]
        Hinge joints in the column order of the ``joint_pos`` / ``joint_vel`` the module is called with.
    link_names : Sequence[str]
        Links returned, in output order.
    """

    def __init__(self, mjcf_path: str, joint_names: Sequence[str], link_names: Sequence[str]):
        super().__init__()
        model = mujoco.MjModel.from_xml_path(mjcf_path)
        self.joint_names = list(joint_names)
        self.link_names = list(link_names)

        free, hinge = mujoco.mjtJoint.mjJNT_FREE, mujoco.mjtJoint.mjJNT_HINGE
        body_of = {model.body(b).name: b for b in range(model.nbody)}
        joint_of = {model.joint(j).name: j for j in range(model.njnt)}
        missing_links = [n for n in self.link_names if n not in body_of]
        missing_joints = [n for n in self.joint_names if n not in joint_of]
        if missing_links or missing_joints:
            raise ValueError(
                f"{mjcf_path} has no link(s) {missing_links} / joint(s) {missing_joints}. "
                f"Links: {sorted(body_of)}; joints: {sorted(joint_of)}."
            )
        col_of = {joint_of[n]: c for c, n in enumerate(self.joint_names)}
        roots = [b for b in range(1, model.nbody) if model.body_parentid[b] == 0]
        if len(roots) != 1 or model.body_jntnum[roots[0]] != 1 or model.jnt_type[model.body_jntadr[roots[0]]] != free:
            raise ValueError(f"{mjcf_path}: needs exactly one body under the world, carrying a free joint.")
        root = roots[0]

        # Steps: (parent step, joint column, qpos0, axis, anchor, offset pos, offset quat). A step without a hinge reads
        # column J, a zero pad after the J joints. Step 0, the root, is the identity.
        zero, identity = np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])
        no_hinge = (len(self.joint_names), 0.0, zero, zero)
        steps: list[tuple] = [(0, *no_hinge, zero, identity)]
        step_of = {root: 0}

        def add_body(body: int) -> int:
            if body in step_of:
                return step_of[body]
            parent = int(model.body_parentid[body])
            if parent == 0:
                raise ValueError(f"{mjcf_path}: body {model.body(body).name} is not below the root.")
            step = add_body(parent)
            joints = range(int(model.body_jntadr[body]), int(model.body_jntadr[body] + model.body_jntnum[body]))
            if any(model.jnt_type[j] != hinge for j in joints):
                raise ValueError(f"{mjcf_path}: body {model.body(body).name} has a joint that is not a hinge.")
            unlisted = [model.joint(j).name for j in joints if j not in col_of]
            if unlisted:
                raise ValueError(f"Joints {unlisted} move link {model.body(body).name} but are not in joint_names.")
            hinges = [
                (col_of[j], model.qpos0[model.jnt_qposadr[j]], model.jnt_axis[j], model.jnt_pos[j]) for j in joints
            ]
            offset = model.body_pos[body], model.body_quat[body]
            for joint in hinges or [no_hinge]:
                steps.append((step, *joint, *offset))
                step, offset = len(steps) - 1, (zero, identity)
            step_of[body] = step
            return step

        # Each link's path of steps, root first, front-padded with the root step to one depth.
        paths = []
        for name in self.link_names:
            step, path = add_body(body_of[name]), []
            while step:
                path.append(step)
                step = steps[step][0]
            paths.append(path[::-1])
        depth = max(1, max(map(len, paths)))
        paths = [[0] * (depth - len(path)) + path for path in paths]

        # Per path step, with o its offset quaternion: the hinge anchor and axis in the parent frame, the anchor in
        # the step frame, o, and o * (0, axis); the step's rotation is cos(h) o + sin(h) o * (0, axis), h = (q-q0)/2.
        consts = np.zeros((depth, len(paths), 17))
        for r, path in enumerate(paths):
            for d, step in enumerate(path):
                _, _, _, axis, anchor, offset_pos, offset_quat = steps[step]
                parent_anchor, parent_axis, turned = np.zeros(3), np.zeros(3), np.zeros(4)
                mujoco.mju_rotVecQuat(parent_anchor, anchor, offset_quat)
                mujoco.mju_rotVecQuat(parent_axis, axis, offset_quat)
                mujoco.mju_mulQuat(turned, offset_quat, np.concatenate([[0.0], axis]))
                consts[d, r] = np.concatenate([offset_pos + parent_anchor, parent_axis, anchor, offset_quat, turned])
        by_depth = [[steps[s] for s in column] for column in zip(*paths, strict=True)]
        self.register_buffer("consts", torch.as_tensor(consts, dtype=torch.float32), persistent=False)
        self.register_buffer("col", torch.tensor([[s[1] for s in d] for d in by_depth]), persistent=False)
        self.register_buffer(
            "qpos0", torch.tensor([[s[2] for s in d] for d in by_depth], dtype=torch.float32), persistent=False
        )

    def forward(
        self,
        root_pos: torch.Tensor,
        root_quat: torch.Tensor,
        joint_pos: torch.Tensor,
        root_lin_vel: torch.Tensor,
        root_ang_vel: torch.Tensor,
        joint_vel: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Positions ``(..., L, 3)``, wxyz quaternions ``(..., L, 4)``, linear and angular velocities ``(..., L, 3)``
        of the links in the world frame, from the root link's world pose (wxyz) and twist and the joints ``(..., J)``.
        """
        joint_pos, joint_vel = (torch.nn.functional.pad(x, (0, 1)) for x in (joint_pos, joint_vel))
        zero, one = root_pos.new_zeros(()), root_pos.new_ones(())
        pos: _Components = (zero, zero, zero)
        quat: _Components = (one, zero, zero, zero)
        lin: _Components = (zero, zero, zero)
        ang: _Components = (zero, zero, zero)
        for col, qpos0, consts in zip(self.col, self.qpos0, self.consts, strict=True):
            c = consts.unbind(-1)
            half = (joint_pos[..., col] - qpos0) / 2
            # The hinge anchor and twist in the root frame, from the parent's pose.
            anchor = _add(pos, _rotate(quat, c[:3]))
            twist = tuple(a * joint_vel[..., col] for a in _rotate(quat, c[3:6]))
            turn = tuple(half.cos() * o + half.sin() * t for o, t in zip(c[9:13], c[13:], strict=True))
            quat = _mul(quat, turn)
            pos = _sub(anchor, _rotate(quat, c[6:9]))
            ang = _add(ang, twist)
            lin = _add(lin, _cross(anchor, twist))

        w, x, y, z = root_quat.unsqueeze(-2).unbind(-1)
        norm = (w * w + x * x + y * y + z * z).sqrt()
        root = (w / norm, x / norm, y / norm, z / norm)
        pos = _rotate(root, pos)
        ang = _add(root_ang_vel.unsqueeze(-2).unbind(-1), _rotate(root, ang))
        lin = _add(root_lin_vel.unsqueeze(-2).unbind(-1), _rotate(root, lin))
        lin = _add(lin, _cross(ang, pos))
        pos = _add(root_pos.unsqueeze(-2).unbind(-1), pos)
        state = torch.stack([*pos, *_mul(root, quat), *lin, *ang], dim=-1)
        pos_w, quat_w, lin_vel_w, ang_vel_w = state.split((3, 4, 3, 3), dim=-1)
        return pos_w, quat_w, lin_vel_w, ang_vel_w


def _add(a: _Components, b: _Components) -> _Components:
    return tuple(x + y for x, y in zip(a, b, strict=True))


def _sub(a: _Components, b: _Components) -> _Components:
    return tuple(x - y for x, y in zip(a, b, strict=True))


def _cross(a: _Components, b: _Components) -> _Components:
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _rotate(q: _Components, v: _Components) -> _Components:
    """``v`` turned by the unit wxyz quaternion ``q``."""
    t = tuple(2 * x for x in _cross(q[1:], v))
    return _add(_add(v, tuple(q[0] * x for x in t)), _cross(q[1:], t))


def _mul(a: _Components, b: _Components) -> _Components:
    """The Hamilton product of wxyz quaternions."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )
