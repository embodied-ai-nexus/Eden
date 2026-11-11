"""The demonstrator's surface: a skinned body mesh posed per frame, plus its joint frames.

OTRetarget reads interactions off the human *surface*, not its skeleton (Sec. III-A): probes are
sampled on the posed mesh and each one is assigned to the body part that skins it. The skeleton's
joint frames still drive the style cost (Sec. III-D). :class:`HumanSurfaceMotion` is that pair, in a
z-up metric world, and is all :class:`~eden.extensions.retargeting.otretarget.OTRetargeter` needs
from a body model — so SMPL-X (OMOMO, AMASS) and the SOMA rig (BONES-SEED BVH) feed it alike.

:class:`~eden.datasets.utils.body_model.SkinnedBodyModel`, Eden's linear-blend-skinning rig, poses both:
:func:`smplx_motion` builds it from an SMPL-X model file, and
:func:`~eden.extensions.retargeting.otretarget.sources.load_soma_bvh` from the SOMA rig.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from eden.datasets.utils.body_model import body_model_arrays, skinned_body_model
from eden.utils.geom import quat_from_axis_angle, quat_to_R

#: SMPL-X's 55 skeleton joints, in its joint order.
SMPLX_JOINT_NAMES: tuple[str, ...] = (
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "jaw",
    "left_eye",
    "right_eye",
    *(f"left_{f}{i}" for f in ("index", "middle", "pinky", "ring", "thumb") for i in (1, 2, 3)),
    *(f"right_{f}{i}" for f in ("index", "middle", "pinky", "ring", "thumb") for i in (1, 2, 3)),
)

#: A y-up world (SMPL-X's native frame, OMOMO, BVH) to Eden's z-up world: +90 deg about x.
Y_UP_TO_Z_UP = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])

#: A body model's canonical frame (y-up, facing +z) to Eden's robot frame (z-up, facing +x), so a rest
#: pose and a robot T-pose face the same way when the transport plan is solved.
CANONICAL_TO_ROBOT = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


@dataclass
class HumanSurfaceMotion:
    """A posed human surface over time, with its skeleton, in a z-up metric world.

    Attributes
    ----------
    vertices : np.ndarray
        ``(T, V, 3)`` posed mesh vertices (world).
    faces : np.ndarray
        ``(F, 3)`` triangle indices.
    skin_weights : np.ndarray
        ``(V, J)`` skinning weights; a vertex belongs to the part of its heaviest joint.
    joint_names : list[str]
        The ``J`` skeleton joints.
    joint_pos : np.ndarray
        ``(T, J, 3)`` joint world positions.
    joint_rot : np.ndarray
        ``(T, J, 3, 3)`` joint world orientations.
    rest_vertices : np.ndarray
        ``(V, 3)`` the mesh in the model's rest (T-) pose, facing +x, z-up: the pose the transport
        plan is solved in and the style alignment offsets are measured at.
    rest_joint_pos : np.ndarray
        ``(J, 3)`` joint positions in that rest pose.
    rest_joint_rot : np.ndarray
        ``(J, 3, 3)`` joint orientations in that rest pose.
    fps : float
        Frame rate.
    """

    vertices: np.ndarray
    faces: np.ndarray
    skin_weights: np.ndarray
    joint_names: list[str]
    joint_pos: np.ndarray
    joint_rot: np.ndarray
    rest_vertices: np.ndarray
    rest_joint_pos: np.ndarray
    rest_joint_rot: np.ndarray
    fps: float
    vertex_parts: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        n_frames = len(self.vertices)
        n_joints = len(self.joint_names)
        expect = {
            "vertices": (n_frames, len(self.rest_vertices), 3),
            "skin_weights": (len(self.rest_vertices), n_joints),
            "joint_pos": (n_frames, n_joints, 3),
            "joint_rot": (n_frames, n_joints, 3, 3),
            "rest_joint_pos": (n_joints, 3),
            "rest_joint_rot": (n_joints, 3, 3),
        }
        for name, shape in expect.items():
            if np.shape(getattr(self, name)) != shape:
                raise ValueError(
                    f"HumanSurfaceMotion: {name} has shape {np.shape(getattr(self, name))}, expected {shape} for "
                    f"{n_frames} frames, {len(self.rest_vertices)} vertices and {n_joints} joints."
                )
        self.vertex_parts = np.argmax(self.skin_weights, axis=1)

    @property
    def num_frames(self) -> int:
        return len(self.vertices)

    def joint_index(self, name: str) -> int:
        """Index of joint ``name``, raising with the available names if absent."""
        if name not in self.joint_names:
            raise KeyError(f"Human joint {name!r} is not in this body model. Available: {list(self.joint_names)}.")
        return self.joint_names.index(name)

    def translated(self, offset: np.ndarray) -> HumanSurfaceMotion:
        """The same motion moved by a constant world offset ``(3,)`` (the rest pose is unchanged)."""
        offset = np.asarray(offset, dtype=np.float64)
        return HumanSurfaceMotion(
            vertices=(self.vertices + offset).astype(self.vertices.dtype),
            faces=self.faces,
            skin_weights=self.skin_weights,
            joint_names=list(self.joint_names),
            joint_pos=self.joint_pos + offset,
            joint_rot=self.joint_rot,
            rest_vertices=self.rest_vertices,
            rest_joint_pos=self.rest_joint_pos,
            rest_joint_rot=self.rest_joint_rot,
            fps=self.fps,
        )

    def floor_offset(self, foot_joints: tuple[str, ...], quantile: float = 0.5) -> float:
        """Height of the floor the feet stand on, read off the motion itself.

        The ``quantile`` of the per-frame lowest foot-vertex height. With at least one foot down in most
        frames (standing, walking, carrying, dancing), the median is the stance height: a capture whose
        body fit floats (or sinks) by a constant amount reports that amount, and the short flight phase of
        a jump does not move it. A low quantile would instead pick the frames where the fit penetrates.

        Parameters
        ----------
        foot_joints : tuple[str, ...]
            Skeleton joints whose skinned vertices make up the feet.
        quantile : float
            Quantile over frames.
        """
        feet = np.isin(self.vertex_parts, [self.joint_index(n) for n in foot_joints])
        return float(np.quantile(self.vertices[:, feet, 2].min(axis=1), quantile))

    def mirrored(
        self,
        vertex_mirror: np.ndarray,
        joint_mirror: np.ndarray,
        local_mirror: np.ndarray,
        normal: np.ndarray,
        point: np.ndarray,
    ) -> HumanSurfaceMotion:
        """The mirror image of this motion through a world plane: a left-handed demonstration made right-handed.

        A reflection alone would turn the left hand into a right-shaped hand still labelled "left". The image is
        re-indexed so that every vertex and joint keeps its anatomical name: vertex ``k`` of the result is the
        reflection of vertex ``vertex_mirror[k]`` (its mirror partner on the body), joint ``j`` that of joint
        ``joint_mirror[j]``. Joint frames become ``M R L`` (``M`` the world reflection, ``L`` the reflection in
        the joint's own frame), proper rotations again; the rest pose is mirrored about its sagittal plane,
        which leaves a symmetric rest frame (and so every style alignment offset) unchanged. Face winding is
        reversed so normals still point out of the body.

        Parameters
        ----------
        vertex_mirror : np.ndarray
            ``(V,)`` permutation: the mirror partner of each vertex.
        joint_mirror : np.ndarray
            ``(J,)`` permutation: the mirror partner of each joint (``left_*`` <-> ``right_*``, the rest to themselves).
        local_mirror : np.ndarray
            ``(3, 3)`` the body model's sagittal reflection in a joint frame (SMPL-X: ``diag(-1, 1, 1)``).
        normal : np.ndarray
            ``(3,)`` normal of the world mirror plane.
        point : np.ndarray
            ``(3,)`` a point on that plane.
        """
        n_verts, n_joints = len(self.rest_vertices), len(self.joint_names)
        vertex_mirror = np.asarray(vertex_mirror, dtype=np.int64)
        joint_mirror = np.asarray(joint_mirror, dtype=np.int64)
        for name, perm, size in (("vertex_mirror", vertex_mirror, n_verts), ("joint_mirror", joint_mirror, n_joints)):
            if perm.shape != (size,) or not np.array_equal(np.sort(perm), np.arange(size)):
                raise ValueError(f"HumanSurfaceMotion.mirrored: {name} must be a permutation of range({size}).")
        L = np.asarray(local_mirror, dtype=np.float64)
        if L.shape != (3, 3) or not np.isclose(np.linalg.det(L), -1.0) or not np.allclose(L @ L.T, np.eye(3)):
            raise ValueError(f"HumanSurfaceMotion.mirrored: local_mirror must be a 3x3 reflection; got {L.tolist()}.")
        n = np.asarray(normal, dtype=np.float64)
        n = n / np.linalg.norm(n)
        M = np.eye(3) - 2.0 * np.outer(n, n)
        shift = 2.0 * float(np.dot(np.asarray(point, dtype=np.float64), n)) * n

        def reflect(x: np.ndarray) -> np.ndarray:
            return x @ M + shift  # M is symmetric

        R_rest = np.diag([1.0, -1.0, 1.0])  # the rest pose faces +x with its lateral axis along y
        inverse = np.argsort(vertex_mirror)
        return HumanSurfaceMotion(
            vertices=reflect(self.vertices[:, vertex_mirror].astype(np.float64)).astype(self.vertices.dtype),
            faces=inverse[np.asarray(self.faces)][:, ::-1].copy(),
            skin_weights=self.skin_weights[vertex_mirror][:, joint_mirror],
            joint_names=list(self.joint_names),
            joint_pos=reflect(self.joint_pos[:, joint_mirror]),
            joint_rot=M @ self.joint_rot[:, joint_mirror] @ L,
            rest_vertices=self.rest_vertices[vertex_mirror] @ R_rest,
            rest_joint_pos=self.rest_joint_pos[joint_mirror] @ R_rest,
            rest_joint_rot=R_rest @ self.rest_joint_rot[joint_mirror] @ L,
            fps=self.fps,
        )

    def slice(self, start: int = 0, stop: int | None = None, step: int = 1) -> HumanSurfaceMotion:
        """The same motion restricted to frames ``start:stop:step`` (``fps`` divided by ``step``)."""
        sl = np.s_[start:stop:step]
        return HumanSurfaceMotion(
            vertices=self.vertices[sl],
            faces=self.faces,
            skin_weights=self.skin_weights,
            joint_names=list(self.joint_names),
            joint_pos=self.joint_pos[sl],
            joint_rot=self.joint_rot[sl],
            rest_vertices=self.rest_vertices,
            rest_joint_pos=self.rest_joint_pos,
            rest_joint_rot=self.rest_joint_rot,
            fps=self.fps / step,
        )


def _rest_pose_motion(
    rest_vertices: np.ndarray, rest_joint_pos: np.ndarray, rest_joint_rot: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Re-express a rest (T-) pose in Eden's robot frame (z-up, facing +x), feet at z = 0."""
    verts = rest_vertices @ CANONICAL_TO_ROBOT.T
    pos = rest_joint_pos @ CANONICAL_TO_ROBOT.T
    rot = CANONICAL_TO_ROBOT[None] @ rest_joint_rot
    lift = -verts[:, 2].min()
    verts = verts + np.array([0.0, 0.0, lift])
    pos = pos + np.array([0.0, 0.0, lift])
    return verts, pos, rot


def smplx_motion(
    model_path: str,
    poses: np.ndarray,
    trans: np.ndarray,
    betas: np.ndarray,
    fps: float,
    flat_hand_mean: bool = False,
    up_axis: str = "y",
    v_template: np.ndarray | None = None,
) -> HumanSurfaceMotion:
    """Pose SMPL-X from axis-angle parameters into a :class:`HumanSurfaceMotion`.

    Parameters
    ----------
    model_path : str
        An ``SMPLX_<GENDER>`` model file (``.npz`` or ``.pkl``).
    poses : np.ndarray
        ``(T, 156)`` ``[global_orient 3, body 63, left_hand 45, right_hand 45]`` (jaw and eyes are held
        at zero) or ``(T, 165)`` with ``[jaw 3, leye 3, reye 3]`` between body and hands.
    trans : np.ndarray
        ``(T, 3)`` SMPL-X translation.
    betas : np.ndarray
        Shape coefficients, up to the model's shape components: 300 for an SMPL-X v1.1 file (300 shape + 100
        expression blend shapes), 10 for v1.0 (10 + 10). Coefficients past that must be zero.
    fps : float
        Frame rate.
    flat_hand_mean : bool
        False adds the model's mean hand pose to the hand parameters (the ``smplx`` package's default
        with ``use_pca=False``); True reads them as offsets from a flat hand.
    up_axis : {"y", "z"}
        The world up axis of ``poses``/``trans``; a y-up world is rotated to Eden's z-up.
    v_template : np.ndarray, optional
        ``(10475, 3)`` a personal template mesh that replaces the model's mean shape (GRAB ships each
        subject as one, with ``betas`` = 0). ``betas`` are still added on top of it.

    Returns
    -------
    HumanSurfaceMotion
    """
    if up_axis not in ("y", "z"):
        raise ValueError(f"smplx_motion: up_axis={up_axis!r}; expected 'y' or 'z'.")
    data = body_model_arrays(str(model_path))
    poses = np.asarray(poses, dtype=np.float64)
    n_frames = len(poses)
    if poses.shape[1] == 156:
        full = np.concatenate([poses[:, :66], np.zeros((n_frames, 9)), poses[:, 66:]], axis=1)
    elif poses.shape[1] == 165:
        full = poses
    else:
        raise ValueError(
            f"smplx_motion: poses has {poses.shape[1]} columns; expected 156 "
            f"([global 3, body 63, lhand 45, rhand 45]) or 165 (with jaw/leye/reye before the hands)."
        )
    if not flat_hand_mean:
        full = full.copy()
        full[:, 75:120] += data["hands_meanl"]
        full[:, 120:165] += data["hands_meanr"]
    model = skinned_body_model(model_path, "smplx", betas, SMPLX_JOINT_NAMES, v_template)
    joints = model.bind_pos
    v_shaped = model.rest_vertices
    local_rot = quat_to_R(quat_from_axis_angle(full.reshape(n_frames, 55, 3)))
    # SMPL-X places the root joint at trans + J_0 (the root rotates about its joint, not the origin).
    root_pos = np.asarray(trans, dtype=np.float64) + joints[0]
    vertices, joint_pos, joint_rot = model.forward(local_rot, root_pos)
    if up_axis == "y":
        vertices = (vertices.astype(np.float64) @ Y_UP_TO_Z_UP.T).astype(np.float32)
        joint_pos = joint_pos @ Y_UP_TO_Z_UP.T
        joint_rot = Y_UP_TO_Z_UP[None, None] @ joint_rot
    rest_v, rest_p, rest_r = _rest_pose_motion(v_shaped, joints, model.bind_rot)
    return HumanSurfaceMotion(
        vertices=vertices,
        faces=model.faces,
        skin_weights=model.skin_weights,
        joint_names=list(SMPLX_JOINT_NAMES),
        joint_pos=joint_pos,
        joint_rot=joint_rot,
        rest_vertices=rest_v,
        rest_joint_pos=rest_p,
        rest_joint_rot=rest_r,
        fps=float(fps),
    )
