"""BVH (Biovision Hierarchy) file parser with forward kinematics."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation

from eden.utils.geom import xyzw_to_wxyz


@dataclass
class BvhJoint:
    name: str
    offset: np.ndarray
    channels: list[str] = field(default_factory=list)
    children: list[BvhJoint] = field(default_factory=list)
    parent: BvhJoint | None = field(default=None, repr=False)


@dataclass
class BvhData:
    root: BvhJoint
    joints: list[BvhJoint]
    joint_names: list[str]
    framerate: float
    num_frames: int
    frame_data: np.ndarray  # (num_frames, total_channels)


def parse_bvh(filepath: str) -> BvhData:
    """
    Parse a BVH file into structured data.

    Parameters
    ----------
    filepath : str
        Path to the BVH file.

    Returns
    -------
    BvhData
        Parsed skeleton hierarchy and motion data.
    """
    with open(filepath, encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines()]

    joints: list[BvhJoint] = []
    stack: list[BvhJoint] = []
    i = 0

    # Skip to HIERARCHY
    while i < len(lines) and not lines[i].startswith("HIERARCHY"):
        i += 1
    i += 1

    while i < len(lines):
        line = lines[i]

        if line.startswith("ROOT") or line.startswith("JOINT"):
            name = line.split(None, 1)[1] if len(line.split(None, 1)) > 1 else ""
            # Strip the mocap take prefix (e.g. "AA_50_T2:Hips" -> "Hips")
            if ":" in name:
                name = name.split(":")[-1]

            joint = BvhJoint(name=name, offset=np.zeros(3))
            if stack:
                joint.parent = stack[-1]
                stack[-1].children.append(joint)
            joints.append(joint)

        elif line.startswith("End Site"):
            # Skip the entire End Site block
            i += 1
            depth = 0
            while i < len(lines):
                if lines[i].startswith("{"):
                    depth += 1
                elif lines[i].startswith("}"):
                    depth -= 1
                    if depth == 0:
                        break
                i += 1

        elif line == "{":
            stack.append(joints[-1])

        elif line.startswith("}"):
            if stack:
                stack.pop()

        elif line.startswith("OFFSET"):
            parts = line.split()
            joints[-1].offset = np.array([float(parts[1]), float(parts[2]), float(parts[3])])

        elif line.startswith("CHANNELS"):
            parts = line.split()
            n = int(parts[1])
            joints[-1].channels = parts[2 : 2 + n]

        elif line.startswith("MOTION"):
            i += 1
            break

        i += 1

    # Parse Frames and Frame Time
    num_frames = 0
    frame_time = 0.0
    while i < len(lines):
        line = lines[i]
        if line.startswith("Frames"):
            num_frames = int(line.split(":")[1].strip())
        elif line.startswith("Frame Time"):
            frame_time = float(line.split(":")[1].strip())
            i += 1
            break
        i += 1

    # Read motion data
    total_channels = sum(len(j.channels) for j in joints)
    frame_data = np.zeros((num_frames, total_channels))
    for f_idx in range(num_frames):
        if i + f_idx >= len(lines) or not lines[i + f_idx]:
            break
        values = lines[i + f_idx].split()
        frame_data[f_idx] = [float(v) for v in values[:total_channels]]

    framerate = 1.0 / frame_time if frame_time > 0 else 120.0

    return BvhData(
        root=joints[0],
        joints=joints,
        joint_names=[j.name for j in joints],
        framerate=framerate,
        num_frames=num_frames,
        frame_data=frame_data,
    )


def forward_kinematics(
    bvh: BvhData,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute global joint positions and orientations via forward kinematics.

    Parameters
    ----------
    bvh : BvhData
        Parsed BVH data.

    Returns
    -------
    positions : np.ndarray, shape (num_frames, num_joints, 3)
        Global positions of all joints.
    quaternions : np.ndarray, shape (num_frames, num_joints, 4)
        Global orientations in wxyz format.
    """
    positions, rot_matrices = forward_kinematics_matrices(bvh)
    flat_quat_xyzw = Rotation.from_matrix(rot_matrices.reshape(-1, 3, 3)).as_quat()
    quaternions = xyzw_to_wxyz(flat_quat_xyzw).reshape(*rot_matrices.shape[:2], 4)
    return positions, quaternions


def forward_kinematics_matrices(bvh: BvhData) -> tuple[np.ndarray, np.ndarray]:
    """
    Global joint positions and rotation matrices: :func:`forward_kinematics` without the quaternion conversion.

    A joint's rotation channels compose in the order they are listed (``Zrotation Yrotation Xrotation`` is
    ``Rz @ Ry @ Rx``), and its position channels, on any joint, replace the matching axes of its offset: a
    SOMA-rig BVH gives every joint all six channels.

    Parameters
    ----------
    bvh : BvhData
        Parsed BVH data.

    Returns
    -------
    positions : np.ndarray, shape (num_frames, num_joints, 3)
        Global positions of all joints.
    rot_matrices : np.ndarray, shape (num_frames, num_joints, 3, 3)
        Global orientations.
    """
    num_frames = bvh.num_frames
    num_joints = len(bvh.joints)

    positions = np.zeros((num_frames, num_joints, 3))
    rot_matrices = np.zeros((num_frames, num_joints, 3, 3))
    joint_id_map = {id(j): i for i, j in enumerate(bvh.joints)}

    # Process each joint (parse order guarantees parent before child)
    ch_off = 0
    for j_idx, joint in enumerate(bvh.joints):
        local_pos = np.tile(np.asarray(joint.offset, dtype=np.float64), (num_frames, 1))
        rot_order = []
        rot_angles = []
        for ch_name in joint.channels:
            ch_lower = ch_name.lower()
            axis = ch_lower[0]
            if "position" in ch_lower:
                local_pos[:, "xyz".index(axis)] = bvh.frame_data[:, ch_off]
            elif "rotation" in ch_lower:
                rot_order.append(axis)
                rot_angles.append(bvh.frame_data[:, ch_off])
            ch_off += 1

        # Uppercase axes: scipy's intrinsic convention, i.e. the channels' matrices multiplied in listed order
        if rot_order:
            angles = np.stack(rot_angles, axis=-1)  # (num_frames, n_rot)
            local_rot = Rotation.from_euler("".join(rot_order).upper(), angles, degrees=True).as_matrix()
        else:
            local_rot = np.tile(np.eye(3), (num_frames, 1, 1))

        if joint.parent is None:
            positions[:, j_idx] = local_pos
            rot_matrices[:, j_idx] = local_rot
        else:
            p_idx = joint_id_map[id(joint.parent)]
            parent_rot = rot_matrices[:, p_idx]
            positions[:, j_idx] = positions[:, p_idx] + np.einsum("fij,fj->fi", parent_rot, local_pos)
            rot_matrices[:, j_idx] = np.einsum("fij,fjk->fik", parent_rot, local_rot)

    return positions, rot_matrices
