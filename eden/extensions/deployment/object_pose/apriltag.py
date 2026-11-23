"""The object's pose from AprilTags fixed on it (``pupil_apriltags``).

Each tag on the object is a :class:`TagMount`: its id, printed size and pose in the object's frame.
Every tag seen gives one estimate of the object, ``T_report_obj = T_report_tag · T_obj_tag⁻¹``; the
estimates are averaged, weighted by the detector's decision margin. One tag is enough for a pose,
and more tags keep the object tracked as it turns (a cube with a tag per face, a pen with a ring).

The report frame is the camera's optical frame (x right, y down, z forward), or, with ``anchor``,
an anchor tag's frame: a tag fixed to whatever the deployment places the pose relative to (the
robot's wrist, the table). Reporting relative to an anchor removes the camera extrinsic from the
chain, the way the Wuji rig's observer reports in its wrist-tag frame. With
``anchor_latch_frames > 0`` the anchor is averaged over that many frames and then held, for a
camera that does not move relative to it: the object stays tracked while the hand covers the anchor.

Tag frames follow AprilTag 3's convention, which ``pupil_apriltags`` returns: origin at the tag's
centre, x to the right of the printed tag, y down, z into the tag (away from a camera facing it).
``TagMount.pos`` / ``TagMount.quat`` are the pose of *that* frame in the object's frame.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from eden.extensions.deployment.object_pose.pose_source import CameraFrame, PoseEstimator
from eden.utils.geom import R_to_quat, inv_transform_pos_quat_by_trans_quat, transform_pos_quat_by_trans_quat

_ORIGIN, _IDENTITY = np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])


@dataclass(frozen=True)
class TagMount:
    """One AprilTag on the object (or the anchor).

    Attributes
    ----------
    tag_id : int
        The tag's id in ``family``.
    size_m : float
        Edge length of the tag's black square, in metres (not the white border's).
    pos, quat : tuple
        The tag frame's pose in the object's frame (wxyz). Ignored for an anchor, whose frame *is*
        the report frame.
    """

    tag_id: int
    size_m: float
    pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        if not self.size_m > 0:
            raise ValueError(
                f"TagMount(tag_id={self.tag_id}): size_m={self.size_m!r} must be a positive length in metres."
            )


def _fuse(poses: list[tuple[np.ndarray, np.ndarray]], weights: list[float]) -> tuple[np.ndarray, np.ndarray]:
    """Weighted mean of nearby poses: positions averaged, quaternions sign-aligned to the first and normalised."""
    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum() if w.sum() > 0 else np.full(len(poses), 1.0 / len(poses))
    pos = sum(wi * p for wi, (p, _) in zip(w, poses))
    ref = poses[0][1]
    quat = sum(wi * (q if np.dot(q, ref) >= 0 else -q) for wi, (_, q) in zip(w, poses))
    return pos, quat / np.linalg.norm(quat)


class AprilTagEstimator(PoseEstimator):
    """Object pose from the AprilTags on it, in the camera's or an anchor tag's frame.

    Parameters
    ----------
    tags : sequence of TagMount
        The tags on the object. Ids must be unique and must not include the anchor's.
    family : str
        The tag family (``"tag36h11"``, ``"tagStandard41h12"``, ...).
    anchor : TagMount | None
        A tag fixed in the frame the pose is placed in; ``None`` reports in the camera's frame.
    anchor_latch_frames : int
        Average the anchor over this many frames, then hold it (``0``: detect it every frame, and
        report nothing in a frame without it). Only for a camera fixed relative to the anchor.
    max_tag_spread_m : float
        Drop a frame whose per-tag object estimates spread wider than this (a misdetection, or a
        ``TagMount`` with the wrong pose), rather than average them into a pose between the two.
    detector_kwargs : dict
        Passed to ``pupil_apriltags.Detector`` (``nthreads``, ``quad_decimate``, ``refine_edges``, ...).
    """

    def __init__(
        self,
        tags: list[TagMount] | tuple[TagMount, ...],
        family: str = "tag36h11",
        anchor: TagMount | None = None,
        anchor_latch_frames: int = 0,
        max_tag_spread_m: float = 0.02,
        detector_kwargs: dict | None = None,
    ) -> None:
        if not tags:
            raise ValueError("AprilTagEstimator: tags is empty; give at least one TagMount on the object.")
        ids = [t.tag_id for t in tags]
        if len(set(ids)) != len(ids):
            raise ValueError(f"AprilTagEstimator: duplicate tag ids in tags {ids}.")
        if anchor is not None and anchor.tag_id in ids:
            raise ValueError(
                f"AprilTagEstimator: anchor tag id {anchor.tag_id} is also one of the object's tags {ids}."
            )
        if anchor_latch_frames < 0 or (anchor_latch_frames and anchor is None):
            raise ValueError(
                f"AprilTagEstimator: anchor_latch_frames={anchor_latch_frames} needs an anchor tag and must be >= 0."
            )
        try:
            from pupil_apriltags import Detector
        except ImportError as e:
            raise ImportError("AprilTagEstimator needs pupil-apriltags: pip install pupil-apriltags") from e
        self.tags = {t.tag_id: t for t in tags}
        self.anchor = anchor
        self.anchor_latch_frames = int(anchor_latch_frames)
        self.max_tag_spread_m = float(max_tag_spread_m)
        self._detector = Detector(families=family, **(detector_kwargs or {}))
        # One detect() call estimates every tag's pose at one size; a size per distinct size_m.
        self._sizes = sorted({t.size_m for t in tags} | ({anchor.size_m} if anchor is not None else set()))
        self.reset()

    def reset(self) -> None:
        self._anchor_samples: list[tuple[np.ndarray, np.ndarray]] = []
        self._anchor_latched: tuple[np.ndarray, np.ndarray] | None = None

    def _detect(self, frame: CameraFrame) -> dict[int, tuple[np.ndarray, np.ndarray, float]]:
        """``{tag_id: (pos, quat, decision_margin)}`` in the camera frame, for the object's tags and the anchor."""
        import cv2

        gray = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2GRAY)
        K = frame.K
        if frame.dist_coeffs is not None and np.any(frame.dist_coeffs):
            gray = cv2.undistort(gray, K, frame.dist_coeffs)
        params = (K[0, 0], K[1, 1], K[0, 2], K[1, 2])
        wanted = {tid: t.size_m for tid, t in self.tags.items()}
        if self.anchor is not None:
            wanted[self.anchor.tag_id] = self.anchor.size_m
        found: dict[int, tuple[np.ndarray, np.ndarray, float]] = {}
        for size in self._sizes:
            for d in self._detector.detect(gray, estimate_tag_pose=True, camera_params=params, tag_size=size):
                if wanted.get(d.tag_id) != size or d.pose_R is None:
                    continue
                found[d.tag_id] = (
                    np.asarray(d.pose_t, dtype=np.float64).reshape(3),
                    R_to_quat(d.pose_R),
                    d.decision_margin,
                )
        return found

    def _anchor_pose(self, found: dict) -> tuple[np.ndarray, np.ndarray] | None:
        """The anchor's pose in the camera frame this frame: detected, latched, or ``None``."""
        if self._anchor_latched is not None:
            return self._anchor_latched
        seen = found.get(self.anchor.tag_id)
        if seen is None:
            return None
        if self.anchor_latch_frames == 0:
            return seen[0], seen[1]
        self._anchor_samples.append((seen[0], seen[1]))
        if len(self._anchor_samples) < self.anchor_latch_frames:
            return None  # still latching: report nothing rather than a pose relative to a noisier anchor
        self._anchor_latched = _fuse(self._anchor_samples, [1.0] * len(self._anchor_samples))
        return self._anchor_latched

    def estimate(self, frame: CameraFrame) -> tuple[np.ndarray, np.ndarray, float] | None:
        found = self._detect(frame)
        # Before the object's tags: the anchor latches over frames in which the object is not seen too.
        anchor = self._anchor_pose(found) if self.anchor is not None else None
        estimates, weights = [], []
        for tid, (pos, quat, margin) in found.items():
            mount = self.tags.get(tid)
            if mount is None:
                continue
            # T_cam_obj = T_cam_tag · T_obj_tag⁻¹
            tag_in_obj_inv = inv_transform_pos_quat_by_trans_quat(
                _ORIGIN, _IDENTITY, np.asarray(mount.pos, dtype=np.float64), np.asarray(mount.quat, dtype=np.float64)
            )
            estimates.append(transform_pos_quat_by_trans_quat(*tag_in_obj_inv, pos, quat))
            weights.append(margin)
        if not estimates:
            return None
        if len(estimates) > 1:
            spread = np.ptp(np.stack([p for p, _ in estimates]), axis=0).max()
            if spread > self.max_tag_spread_m:
                return None
        pos, quat = _fuse(estimates, weights)
        if self.anchor is not None:
            if anchor is None:
                return None
            pos, quat = inv_transform_pos_quat_by_trans_quat(pos, quat, *anchor)  # T_anchor_obj
        return pos, quat, float(sum(weights))
