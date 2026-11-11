"""Parametric body models behind the opt-in body-model fields of a motion sequence.

A dataset loaded with ``body_model_params=True`` stores, per sequence that has one, the model's pose and shape:
``body_pose_quat`` (the local rotation of every joint below the root, wxyz, relative to the model's rest pose),
``body_shape`` (``betas``) and the per-clip ``body_model`` name; the root is the sequence's base, ``base_pos_w``
being the root joint's position. This module names the models (:data:`BODY_MODELS`), converts a release's own
MANO parameterisation into that form, and runs the model back to joints and vertices.

The model files are licence-gated (MANO: mano.is.tue.mpg.de) and never shipped; they are read from a directory the
caller passes (``DatasetBase.body_model_dir``). The models run on :class:`SkinnedBodyModel`, a numpy linear-blend-
skinning rig that matches the ``smplx`` package to well under a micrometre (``tests/datasets/utils``) and also
skins rigs outside the SMPL family (OTRetarget's SOMA). A model is read from its ``.npz`` when the directory has
one, else its ``.pkl``. The MANO, SMPL and SMPL-H ``.pkl`` files pickle a ``chumpy`` array or two (``shapedirs``);
:func:`body_model_arrays` reads those without chumpy (:class:`~eden.utils.chumpy_pickle.ChumpyFreeUnpickler`), so
no model file needs an extra.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch

from eden.datasets.constants import MANO_KEYPOINTS_NAME
from eden.utils.chumpy_pickle import ChumpyFreeUnpickler
from eden.utils.geom import quat_from_axis_angle, quat_to_R

#: The parametric model families :data:`BODY_MODELS` names.
BodyModelFamily = Literal["mano", "smpl", "smplh", "smplx"]


@dataclass(frozen=True)
class BodyModelSpec:
    """One parametric body model.

    Parameters
    ----------
    family : BodyModelFamily
        ``mano``, ``smpl``, ``smplh`` or ``smplx``.
    variant : str
        The hand side (MANO) or the gender (SMPL family).
    num_body_joints : int
        Joints below the root, i.e. ``body_pose_quat``'s second dimension.
    file : str
        The model file's name inside the body-model directory.
    """

    family: BodyModelFamily
    variant: str
    num_body_joints: int
    file: str


#: ``body_model`` name -> its spec. MANO: 15 finger joints; SMPL: 23; SMPL-H: 21 body + 2 x 15 hand; SMPL-X: 21 body
#: + jaw + 2 eyes + 2 x 15 hand.
BODY_MODELS: dict[str, BodyModelSpec] = {
    "mano_right": BodyModelSpec("mano", "right", 15, "MANO_RIGHT.pkl"),
    "mano_left": BodyModelSpec("mano", "left", 15, "MANO_LEFT.pkl"),
    **{
        f"{family}_{gender}": BodyModelSpec(family, gender, joints, f"{family.upper()}_{gender.upper()}.pkl")
        for family, joints in (("smpl", 23), ("smplh", 51), ("smplx", 54))
        for gender in ("neutral", "male", "female")
    },
}

#: MANO vertices taken as the fingertips of :data:`~eden.datasets.constants.MANO_KEYPOINTS_NAME` (thumb, index,
#: middle, ring, pinky): manopth's, which DexYCB's labelled keypoints use too.
MANO_TIP_VERTICES: tuple[int, ...] = (745, 317, 444, 556, 673)

#: The fingertip (finger-pad) vertices CHORD and FlashDexRetarget track and segment HOT3D contact by. Not Eden's
#: keypoint tips (:data:`MANO_TIP_VERTICES`, the nail ends): their own convention, kept apart so neither borrows it.
CHORD_TIP_VERTICES: dict[str, int] = {"thumb": 766, "index": 353, "middle": 467, "ring": 576, "pinky": 695}

assert len(MANO_KEYPOINTS_NAME) == 16 + len(MANO_TIP_VERTICES), "MANO keypoints are its 16 joints and 5 tips"


def mano_model_name(side: str) -> str:
    """The :data:`BODY_MODELS` name of the MANO hand on ``side`` (``right`` / ``left``)."""
    if side not in ("right", "left"):
        raise ValueError(f"A MANO hand is 'right' or 'left', got {side!r}.")
    return f"mano_{side}"


def model_file_names(body_model: str) -> list[str]:
    """The file names ``body_model`` goes by in a body-model directory, in the order they are looked for.

    The :data:`BODY_MODELS` name (``SMPLH_MALE``), then for the SMPL family the official downloads' spelling
    (``SMPLH_male``) and AMASS's ``<gender>/model.npz``; each ``.npz`` before its ``.pkl`` (no chumpy needed).
    """
    spec = _spec(body_model)
    stems = [spec.file.removesuffix(".pkl")]
    if spec.family != "mano":
        stems.append(f"{spec.family.upper()}_{spec.variant}")
    names = [f"{stem}{suffix}" for stem in stems for suffix in (".npz", ".pkl")]
    if spec.family != "mano":
        names.append(f"{spec.variant}/model.npz")
    return names


def body_model_file(model_dir: str | Path | None, body_model: str) -> Path:
    """The file of ``body_model`` in ``model_dir`` (the first of :func:`model_file_names` present); raises naming
    where to get it when either is missing.
    """
    spec = _spec(body_model)
    if model_dir is None:
        raise ValueError(
            f"Reading {body_model} needs its model file ({spec.file}), but no body-model directory was given. Set "
            f"body_model_dir on the dataset options to the directory holding it."
        )
    names = model_file_names(body_model)
    for name in names:
        if (Path(model_dir) / name).exists():
            return Path(model_dir) / name
    where = (
        "register at mano.is.tue.mpg.de and download 'Models & Code' (mano_v1_2.zip); its models/ directory holds it"
        if spec.family == "mano"
        else f"download {spec.family.upper()} from its project page"
    )
    raise FileNotFoundError(
        f"{Path(model_dir) / spec.file} missing (also looked for {', '.join(n for n in names if n != spec.file)}). "
        f"The body-model files are licence-gated and never shipped: {where}."
    )


#: Arrays every body-model file holds; a file without them (an ``.npz`` converted by another project under other
#: key names, say) is refused naming what it lacks.
_REQUIRED_ARRAYS = ("v_template", "shapedirs", "posedirs", "J_regressor", "weights", "kintree_table", "f")


def body_model_arrays(path: str | Path) -> dict[str, np.ndarray]:
    """Every array of a body-model file (``.npz`` or ``.pkl``), as plain numpy in the file's own dtypes.

    A ``.pkl`` that pickles chumpy objects (MANO, SMPL, SMPL-H) is read without chumpy
    (:class:`~eden.utils.chumpy_pickle.ChumpyFreeUnpickler`); its sparse ``J_regressor`` is densified. The last two
    files read stay cached (a MANO pair; an SMPL-X file holds ~170 MB), keyed by modification time and size so a
    replaced file is read again. The returned dict is shared: copy before writing into it.
    """
    stat = Path(path).stat()
    return _load_arrays(str(path), stat.st_mtime_ns, stat.st_size)


@functools.lru_cache(maxsize=2)
def _load_arrays(path: str, mtime_ns: int, size: int) -> dict[str, np.ndarray]:
    if path.endswith(".npz"):
        with np.load(path, allow_pickle=True) as data:
            raw = {k: data[k] for k in data.files}
    else:
        with open(path, "rb") as f:
            raw = ChumpyFreeUnpickler(f, encoding="latin1").load()
    # A pickled J_regressor is scipy.sparse; chumpy arrays convert through np.asarray.
    arrays = {k: np.asarray(v.toarray() if hasattr(v, "toarray") else v) for k, v in raw.items()}
    missing = [k for k in _REQUIRED_ARRAYS if k not in arrays]
    if missing:
        raise ValueError(
            f"body_model_arrays: {path} lacks {missing} (it holds {sorted(arrays)}); it is not an SMPL/MANO-format "
            f"model file. Use the release's own .pkl, or an .npz with the same keys."
        )
    return arrays


def shape_components(arrays: dict[str, np.ndarray], family: BodyModelFamily) -> int:
    """How many of ``shapedirs``' blend shapes are body shape (``betas``) rather than facial expression.

    SMPL-X stacks the shape blend shapes, then the expression ones: 300 + 100 in v1.1, 10 + 10 in v1.0 (the
    ``smplx`` package's split). The other families' ``shapedirs`` are all shape.
    """
    n = arrays["shapedirs"].shape[-1]
    if family != "smplx":
        return n
    return 300 if n >= 400 else n // 2


class SkinnedBodyModel:
    """A linear-blend-skinning rig with a fixed identity.

    Parameters
    ----------
    rest_vertices : np.ndarray
        ``(V, 3)`` mesh in the bind pose (model frame).
    faces : np.ndarray
        ``(F, 3)``.
    skin_weights : np.ndarray
        ``(V, J)``, rows summing to one.
    parents : np.ndarray
        ``(J,)`` parent joint index, ``-1`` for the root.
    bind_rot : np.ndarray
        ``(J, 3, 3)`` joint world orientations in the bind pose.
    bind_pos : np.ndarray
        ``(J, 3)`` joint world positions in the bind pose.
    joint_names : list[str]
        Joint names.
    posedirs : np.ndarray, optional
        ``(V, 3, 9 * (J - 1))`` pose-corrective blend shapes, driven by ``(R_local - I)`` of every
        non-root joint (SMPL convention).
    """

    def __init__(
        self,
        rest_vertices: np.ndarray,
        faces: np.ndarray,
        skin_weights: np.ndarray,
        parents: np.ndarray,
        bind_rot: np.ndarray,
        bind_pos: np.ndarray,
        joint_names: list[str],
        posedirs: np.ndarray | None = None,
    ) -> None:
        self.rest_vertices = np.asarray(rest_vertices, dtype=np.float64)
        self.faces = np.asarray(faces, dtype=np.int64)
        self.skin_weights = np.asarray(skin_weights, dtype=np.float64)
        self.parents = np.asarray(parents, dtype=np.int64)
        self.bind_rot = np.asarray(bind_rot, dtype=np.float64)
        self.bind_pos = np.asarray(bind_pos, dtype=np.float64)
        self.joint_names = list(joint_names)
        self.posedirs = None if posedirs is None else np.asarray(posedirs, dtype=np.float64)
        n = len(self.parents)
        for j in range(n):
            if self.parents[j] >= j:
                raise ValueError(
                    f"SkinnedBodyModel: joint {j} ({self.joint_names[j]!r}) has parent {self.parents[j]}, but parents "
                    f"must precede their children so a single forward pass poses the chain."
                )
        # Bone offsets in the parent's bind frame.
        self._offsets = np.zeros((n, 3))
        for j in range(1, n):
            p = self.parents[j]
            self._offsets[j] = self.bind_rot[p].T @ (self.bind_pos[j] - self.bind_pos[p])

    def forward(
        self, local_rot: np.ndarray, root_pos: np.ndarray, chunk: int = 256
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Pose the rig.

        Parameters
        ----------
        local_rot : np.ndarray
            ``(T, J, 3, 3)`` joint rotations relative to the bind pose, each in the joint's own bind
            frame: ``W_j = W_parent @ (B_parent^T B_j) @ local_j`` and ``W_root = B_root @ local_root``.
        root_pos : np.ndarray
            ``(T, 3)`` root joint world position.
        chunk : int
            Frames skinned per batch (bounds memory).

        Returns
        -------
        vertices : np.ndarray
            ``(T, V, 3)``.
        joint_pos : np.ndarray
            ``(T, J, 3)``.
        joint_rot : np.ndarray
            ``(T, J, 3, 3)`` joint world orientations.
        """
        local_rot = np.asarray(local_rot, dtype=np.float64)
        n_frames, n_joints = local_rot.shape[:2]
        if n_joints != len(self.parents):
            raise ValueError(
                f"SkinnedBodyModel.forward: local_rot has {n_joints} joints per frame; the rig has {len(self.parents)} "
                f"({self.joint_names[:4]}...), root included."
            )
        joint_rot = np.zeros((n_frames, n_joints, 3, 3))
        joint_pos = np.zeros((n_frames, n_joints, 3))
        for j in range(n_joints):
            p = self.parents[j]
            if p < 0:
                joint_rot[:, j] = self.bind_rot[j] @ local_rot[:, j]
                joint_pos[:, j] = root_pos
            else:
                joint_rot[:, j] = joint_rot[:, p] @ (self.bind_rot[p].T @ self.bind_rot[j]) @ local_rot[:, j]
                joint_pos[:, j] = joint_pos[:, p] + np.einsum("tab,b->ta", joint_rot[:, p], self._offsets[j])
        pose_feature = None
        if self.posedirs is not None:
            pose_feature = (local_rot[:, 1:] - np.eye(3)).reshape(n_frames, -1)
        vertices = self.skin(joint_rot, joint_pos, pose_feature, chunk)
        return vertices, joint_pos, joint_rot

    def skin(
        self, joint_rot: np.ndarray, joint_pos: np.ndarray, pose_feature: np.ndarray | None = None, chunk: int = 256
    ) -> np.ndarray:
        """Skin the mesh to given joint world transforms.

        Parameters
        ----------
        joint_rot : np.ndarray
            ``(T, J, 3, 3)`` joint world orientations.
        joint_pos : np.ndarray
            ``(T, J, 3)`` joint world positions.
        pose_feature : np.ndarray, optional
            ``(T, 9 * (J - 1))`` pose-corrective drive (see ``posedirs``); None skips the correctives.
        chunk : int
            Frames skinned per batch.

        Returns
        -------
        np.ndarray
            ``(T, V, 3)`` float32 vertices.
        """
        n_frames, n_verts = len(joint_rot), len(self.rest_vertices)
        vertices = np.zeros((n_frames, n_verts, 3), dtype=np.float32)
        # Skinning transform per joint: bind frame -> posed frame, packed (rotation 9, translation 3) so one matrix
        # product blends both: einsums over (T, V, J) without a contraction path ran 2.4x slower than these BLAS calls.
        skin_rot = joint_rot @ np.transpose(self.bind_rot, (0, 2, 1))[None]
        skin_trans = joint_pos - np.einsum("tjab,jb->tja", skin_rot, self.bind_pos)
        packed = np.concatenate([skin_rot.reshape(n_frames, -1, 9), skin_trans], axis=2)  # (T, J, 12)
        posedirs = None if self.posedirs is None else self.posedirs.reshape(n_verts * 3, -1)
        for s in range(0, n_frames, chunk):
            e = min(s + chunk, n_frames)
            rest = np.broadcast_to(self.rest_vertices, (e - s, n_verts, 3))
            if pose_feature is not None and posedirs is not None:
                rest = rest + (pose_feature[s:e] @ posedirs.T).reshape(e - s, n_verts, 3)
            blended = (self.skin_weights @ packed[s:e]).reshape(e - s, n_verts, 12)
            rot = blended[..., :9].reshape(e - s, n_verts, 3, 3)
            vertices[s:e] = np.einsum("tvab,tvb->tva", rot, rest, optimize=True) + blended[..., 9:]
        return vertices


def skinned_body_model(
    path: str | Path,
    family: BodyModelFamily,
    betas: np.ndarray,
    joint_names: list[str] | None = None,
    v_template: np.ndarray | None = None,
    posedirs: np.ndarray | None = None,
) -> SkinnedBodyModel:
    """The rig of one body-model file shaped by ``betas``: SMPL-style bind pose (identity joint frames).

    Parameters
    ----------
    path : str | Path
        The model file (``.npz`` or ``.pkl``).
    family : BodyModelFamily
        Its family; decides how many of ``shapedirs`` are shape (:func:`shape_components`).
    betas : np.ndarray
        Shape coefficients, up to the model's shape components; coefficients past them must be zero.
    joint_names : list[str], optional
        Names for the joints (default ``joint_0`` ...).
    v_template : np.ndarray, optional
        A personal template mesh replacing the model's mean shape (GRAB ships each subject as one); ``betas`` still
        apply on top of it.
    posedirs : np.ndarray, optional
        The file's ``posedirs`` already in float64, shared by every rig built from it: an SMPL-X file stores them in
        float32, and each rig converting its own copied 122 MB.

    Returns
    -------
    SkinnedBodyModel
    """
    data = body_model_arrays(str(path))
    betas = np.asarray(betas, dtype=np.float64).reshape(-1)
    n_model_shape = shape_components(data, family)
    if np.any(betas[n_model_shape:] != 0.0):
        raise ValueError(
            f"skinned_body_model: {len(betas)} betas, but {path} has {n_model_shape} shape components (shapedirs "
            f"{data['shapedirs'].shape}; any others are expression blend shapes). For SMPL-X use the v1.1 model files "
            f"(300 shape components); truncating the betas would change the body the motion was fitted to."
        )
    n_shape = min(len(betas), n_model_shape)
    template = data["v_template"] if v_template is None else np.asarray(v_template)
    if template.shape != data["v_template"].shape:
        raise ValueError(
            f"skinned_body_model: v_template has shape {template.shape}; {path} expects {data['v_template'].shape} "
            f"(one row per model vertex)."
        )
    shaped = template.astype(np.float64) + np.einsum(
        "vck,k->vc", data["shapedirs"][:, :, :n_shape].astype(np.float64), betas[:n_shape]
    )
    joints = data["J_regressor"].astype(np.float64) @ shaped
    parents = data["kintree_table"][0].astype(np.int64)
    parents[0] = -1
    return SkinnedBodyModel(
        rest_vertices=shaped,
        faces=data["f"].astype(np.int64),
        skin_weights=data["weights"],
        parents=parents,
        bind_rot=np.tile(np.eye(3), (len(parents), 1, 1)),
        bind_pos=joints,
        joint_names=list(joint_names) if joint_names is not None else [f"joint_{j}" for j in range(len(parents))],
        posedirs=data["posedirs"] if posedirs is None else posedirs,
    )


def _mano_arrays(path: str) -> dict[str, np.ndarray]:
    data = body_model_arrays(path)
    missing = [k for k in ("hands_components", "hands_mean") if k not in data]
    if missing:
        raise ValueError(f"{path} lacks the MANO hand-pose basis {missing} (it holds {sorted(data)}).")
    return {k: data[k].astype(np.float64) for k in ("hands_components", "hands_mean")}


def mano_pose_from_release(
    pose: np.ndarray, model_dir: str | Path | None, side: str, *, num_pca_comps: int | None, flat_hand_mean: bool
) -> np.ndarray:
    """A release's MANO finger pose -> the 45 axis-angles relative to the flat hand that ``body_pose_quat`` holds.

    Parameters
    ----------
    pose : np.ndarray
        (T, num_pca_comps) PCA coefficients, or (T, 45) axis-angles when ``num_pca_comps`` is None.
    model_dir : str | Path | None
        Directory holding ``MANO_<SIDE>.pkl``; read only when the pose is PCA or relative to the mean hand.
    side : str
        ``right`` or ``left``.
    num_pca_comps : int | None
        The release's PCA size, or None for axis-angles. All 45 components are still PCA: ``smplx`` turns PCA off
        when asked for 45 and would read the coefficients as axis-angles, which is why this exists.
    flat_hand_mean : bool
        Whether the release's pose is relative to the flat hand (MANO's ``flat_hand_mean=True``) rather than the
        mean hand.

    Returns
    -------
    np.ndarray
        (T, 45) axis-angles, relative to the flat hand.
    """
    pose = np.asarray(pose, dtype=np.float64)
    if num_pca_comps is None and flat_hand_mean:
        return pose
    arrays = _mano_arrays(str(body_model_file(model_dir, mano_model_name(side))))
    if num_pca_comps is not None:
        if pose.shape[-1] != num_pca_comps:
            raise ValueError(f"Expected {num_pca_comps} MANO PCA coefficients per frame, got {pose.shape[-1]}.")
        pose = pose @ arrays["hands_components"][:num_pca_comps]
    if not flat_hand_mean:
        pose = pose + arrays["hands_mean"]
    return pose


def axis_angles_to_body_pose_quat(pose: np.ndarray) -> torch.Tensor:
    """(T, 3 * J) axis-angles -> (T, J, 4) wxyz, float32."""
    pose = torch.as_tensor(np.asarray(pose), dtype=torch.float64)
    return quat_from_axis_angle(pose.reshape(len(pose), -1, 3)).to(torch.float32)


def body_model_forward(
    body_model: str,
    model_dir: str | Path | None,
    root_pos: torch.Tensor,
    root_quat: torch.Tensor,
    body_pose_quat: torch.Tensor,
    body_shape: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Joints and vertices of a body-model sequence, its root joint placed on ``root_pos``.

    Parameters
    ----------
    body_model : str
        A :data:`BODY_MODELS` name.
    model_dir : str | Path | None
        Directory holding the model file.
    root_pos, root_quat : torch.Tensor
        (T, 3) root joint positions and (T, 4) wxyz global orientations: the sequence's ``base_pos_w`` /
        ``base_quat_w``.
    body_pose_quat, body_shape : torch.Tensor
        (T, J, 4) and (T, B): the sequence's body-model fields. The pose is relative to the rest pose (MANO: the flat
        hand), and the jaw, eyes and expression of SMPL-X are held at rest.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        (T, J + 1, 3) joints, root first, and (T, V, 3) vertices, float32 on ``root_pos``'s device.
    """
    spec = _spec(body_model)
    if body_pose_quat.shape[1] != spec.num_body_joints:
        raise ValueError(
            f"body_model_forward: {body_model} has {spec.num_body_joints} joints below the root, but body_pose_quat "
            f"has {body_pose_quat.shape[1]} per frame (shape {tuple(body_pose_quat.shape)})."
        )
    path = str(body_model_file(model_dir, body_model))
    device = root_pos.device
    quats = torch.cat([root_quat[:, None], body_pose_quat], dim=1).detach().to("cpu", torch.float64).numpy()
    local_rot = quat_to_R(quats)
    root = root_pos.detach().to("cpu", torch.float64).numpy()
    shapes = body_shape.detach().to("cpu", torch.float64).numpy()
    joints = np.zeros((len(root), spec.num_body_joints + 1, 3))
    verts = None
    # One rig per distinct shape: a sequence's betas are almost always constant, but the fields allow per-frame ones.
    # The float64 pose correctives are made once and shared by every rig.
    posedirs = body_model_arrays(path)["posedirs"].astype(np.float64)
    unique, which = np.unique(shapes, axis=0, return_inverse=True)
    for k, betas in enumerate(unique):
        rows = np.flatnonzero(which.reshape(-1) == k)
        rig = skinned_body_model(path, spec.family, betas, posedirs=posedirs)
        v, j, _ = rig.forward(local_rot[rows], root[rows])
        if verts is None:
            verts = np.zeros((len(root), v.shape[1], 3), dtype=np.float32)
        joints[rows], verts[rows] = j, v
    as_tensor = functools.partial(torch.as_tensor, dtype=torch.float32, device=device)
    return as_tensor(joints), as_tensor(verts)


def mano_root_offset(model_dir: str | Path | None, side: str, betas: np.ndarray) -> np.ndarray:
    """(…, 3) position of MANO's root joint at zero translation: the offset from a release's ``transl`` to the wrist.

    The root joint is regressed from the shaped template alone, so the offset depends on ``betas`` and not on the
    pose: ``wrist = transl + mano_root_offset(betas)``.
    """
    path = body_model_file(model_dir, mano_model_name(side))
    data = body_model_arrays(path)
    betas = np.asarray(betas, dtype=np.float64)
    if betas.shape[-1] > data["shapedirs"].shape[-1]:
        raise ValueError(
            f"mano_root_offset: {betas.shape[-1]} betas per hand, but {path} has {data['shapedirs'].shape[-1]} shape "
            f"components; pass the first {data['shapedirs'].shape[-1]}."
        )
    regressor = data["J_regressor"][0].astype(np.float64)  # the wrist's row
    base = regressor @ data["v_template"].astype(np.float64)
    dirs = np.einsum("v,vck->kc", regressor, data["shapedirs"][..., : betas.shape[-1]].astype(np.float64))
    return base + betas @ dirs


def mano_vertex_links(model_dir: str | Path | None, side: str) -> np.ndarray:
    """(778,) the MANO joint (0-15) each vertex follows most: the argmax of its skinning weights."""
    return body_model_arrays(str(body_model_file(model_dir, mano_model_name(side))))["weights"].argmax(1)


def mano_keypoints(joints: torch.Tensor, vertices: torch.Tensor) -> torch.Tensor:
    """(T, 21, 3) keypoints in :data:`~eden.datasets.constants.MANO_KEYPOINTS_NAME` order from MANO's (T, 16, 3)
    joints and (T, 778, 3) vertices: the 16 joints in MANO's order, then the :data:`MANO_TIP_VERTICES` tips.
    """
    if joints.shape[-2] != 16:
        raise ValueError(f"Expected MANO's 16 joints, got {joints.shape[-2]}.")
    return torch.cat([joints, vertices[..., list(MANO_TIP_VERTICES), :]], dim=-2)


def _spec(body_model: str) -> BodyModelSpec:
    if body_model not in BODY_MODELS:
        raise ValueError(f"Unknown body_model {body_model!r}; known: {sorted(BODY_MODELS)}.")
    return BODY_MODELS[body_model]
