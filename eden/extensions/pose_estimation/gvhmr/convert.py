"""GVHMR's world-frame SMPL-X output → an AMASS-style, Z-up SMPL-X clip.

Two things change. **Up axis:** GVHMR predicts in a gravity-aligned frame whose **+Y is up**
(its ``"ay"`` frame — ``hmr4d.utils.geo.hmr_global``'s ``"az"->"ay"`` is -90° about X), while
Eden/Genesis/AMASS are Z-up. **Pose layout:** GVHMR returns ``global_orient (T, 3)`` +
``body_pose (T, 63)``; AMASS packs one ``poses (T, 165)``, whose jaw/eyes/hands tail stays zero
because GVHMR does not estimate it.

Does not import GVHMR, so the conversion is testable without the backend installed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from genesis.utils.geom import transform_by_quat

from eden.utils.geom import axis_angle_from_quat, quat_from_axis_angle, quat_mul

#: Packed SMPL-X pose width used by AMASS: root(3) + body(63) + jaw(3) + eyes(6) + hands(90).
SMPLX_POSE_DIM = 165
ROOT_ORIENT_SLICE = slice(0, 3)
BODY_POSE_SLICE = slice(3, 66)

#: Axis-angle taking GVHMR's Y-up gravity frame to a Z-up world (+90° about X).
Y_UP_TO_Z_UP_AXIS_ANGLE = (0.5 * np.pi, 0.0, 0.0)


def y_up_to_z_up_quat() -> torch.Tensor:
    """The wxyz quaternion of :data:`Y_UP_TO_Z_UP_AXIS_ANGLE` (maps ``+Y`` to ``+Z``)."""
    return quat_from_axis_angle(torch.tensor(Y_UP_TO_Z_UP_AXIS_ANGLE, dtype=torch.float32))


def y_up_to_z_up_root(
    global_orient: np.ndarray, transl: np.ndarray, pelvis_j0: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate an SMPL-X root ``(global_orient, transl)`` pair from Y-up into Z-up.

    ``body_pose`` is unaffected — SMPL joint rotations are parent-relative, so only the root
    sees a change of world frame.

    The translation is **not** simply ``R @ transl``: SMPL-X rotates the body about the
    rest-pose pelvis ``J0(betas)`` (``BodyModelSMPLX.get_skeleton(betas)[..., 0, :]``), which
    puts the pelvis at ``J0 + transl`` in the world, so rotating about the origin needs::

        transl' = R @ (transl + J0) - J0

    ``global_orient`` / ``transl`` are ``(T, 3)``, ``pelvis_j0`` ``(T, 3)`` or ``(3,)``;
    returns the Z-up pair as float32.
    """
    orient = torch.as_tensor(np.asarray(global_orient), dtype=torch.float32)
    trans = torch.as_tensor(np.asarray(transl), dtype=torch.float32)
    if orient.ndim != 2 or orient.shape[-1] != 3 or trans.shape != orient.shape:
        raise ValueError(
            f"Expected per-frame (T, 3) arrays; got global_orient {tuple(orient.shape)}, transl {tuple(trans.shape)}."
        )
    j0 = torch.as_tensor(np.asarray(pelvis_j0), dtype=torch.float32).expand_as(trans)
    quat = y_up_to_z_up_quat()

    orient_z = axis_angle_from_quat(quat_mul(quat, quat_from_axis_angle(orient)))
    trans_z = transform_by_quat(trans + j0, quat) - j0
    return orient_z.numpy(), trans_z.numpy()


def _require_keys(mapping: dict, keys: tuple[str, ...], where: str) -> None:
    """Raise a ``ValueError`` naming every missing key, rather than a bare ``KeyError`` later."""
    missing = [key for key in keys if key not in mapping]
    if missing:
        raise ValueError(f"{where} is missing {missing}; got {sorted(mapping)}.")


def to_amass_smplx_clip(
    result: dict, *, motion_fps: float, source_video: str | Path | None = None
) -> dict[str, np.ndarray]:
    """Pack one :meth:`GvhmrEstimator.estimate` result into AMASS-style SMPL-X npz fields.

    Emits the AMASS fields Eden's retargeting path reads (``poses``, ``trans``, ``betas``,
    ``gender``, ``mocap_frame_rate``, ``surface_model_type``) plus ``gvhmr_*`` copies of the
    raw Y-up estimate, so a saved clip stays traceable to what the model predicted.

    ``betas`` is averaged over the clip — AMASS carries one shape vector per sequence, and
    GVHMR's per-frame betas are already a clip-average repeated per frame.
    """
    _require_keys(result, ("global", "pelvis_j0", "K_fullimg"), "GVHMR result")
    g = result["global"]
    _require_keys(g, ("global_orient", "body_pose", "betas", "transl"), "GVHMR result['global']")

    orient, body_pose, betas, transl = g["global_orient"], g["body_pose"], g["betas"], g["transl"]
    pelvis_j0 = result["pelvis_j0"]
    num_frames = orient.shape[0]
    for name, (array, shape) in {
        "global_orient": (orient, (num_frames, 3)),
        "body_pose": (body_pose, (num_frames, 63)),
        "betas": (betas, (num_frames, 10)),
        "transl": (transl, (num_frames, 3)),
        "pelvis_j0": (pelvis_j0, (num_frames, 3)),
    }.items():
        if array.shape != shape:
            raise ValueError(f"GVHMR result field {name!r} has shape {array.shape}, expected {shape}.")

    orient_z, trans_z = y_up_to_z_up_root(orient, transl, pelvis_j0)
    poses = np.zeros((num_frames, SMPLX_POSE_DIM), dtype=np.float32)
    poses[:, ROOT_ORIENT_SLICE] = orient_z
    poses[:, BODY_POSE_SLICE] = body_pose

    payload = {
        "poses": poses,
        "trans": trans_z,
        "betas": betas.mean(axis=0).astype(np.float32),
        "gender": np.array("neutral"),  # GVHMR predicts with the SMPL-X *neutral* model
        "mocap_frame_rate": np.float32(motion_fps),
        "surface_model_type": np.array("smplx"),
        "gvhmr_global_orient": orient,
        "gvhmr_transl": transl,
        "gvhmr_betas": betas,
        "gvhmr_pelvis_j0": pelvis_j0,
        "gvhmr_K_fullimg": result["K_fullimg"],
    }
    if result.get("video_fps") is not None:
        payload["gvhmr_video_fps"] = np.float32(result["video_fps"])
    if source_video is not None:
        payload["source_video"] = np.array(str(source_video))
    return payload


def save_amass_npz(payload: dict[str, np.ndarray], path: str | Path) -> Path:
    """Write a :func:`to_amass_smplx_clip` payload to ``path`` with an ``.npz`` suffix (parents created)."""
    out = Path(path).with_suffix(".npz")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **payload)
    return out
