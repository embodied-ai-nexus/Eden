"""The object's pose from RGB-D and its mesh, with NVlabs FoundationPose (model-based, no markers).

FoundationPose (https://github.com/NVlabs/FoundationPose) is not a pip package: point ``repo_dir``
at a checkout built per its README (its ``estimater.py``, the ``nvdiffrast`` extension and the
downloaded weights). It runs in two modes, and this estimator switches between them:

- **register**: a pose from scratch, given a binary mask of the object in the frame. The mask comes
  from ``mask_fn`` (a fixed box, a colour threshold, SAM2 prompted once, ...); a frame for which it
  returns ``None`` is reported as not found.
- **track**: refines the previous frame's pose. Fast, but it has no notion of losing the object:
  :meth:`FoundationPoseEstimator.estimate` drops back to registering when the tracked pose jumps
  further than ``max_jump_m`` in one frame, projects out of the image, or every
  ``reregister_every`` frames.

The pose is the **mesh file's own frame** in the camera's optical frame (x right, y down, z
forward). That is the scene entity's authored frame only if the entity's visual mesh carries no
origin offset of its own; otherwise give the offset as the tracked object's
:attr:`~eden.extensions.deployment.object_pose.backend.TrackedObjectSpec.object_offset_pos`.

FoundationPose holds a CUDA context and a few GB of GPU memory, and a registration takes hundreds of
milliseconds. Run it in its own process (:func:`~eden.extensions.deployment.object_pose.sources.serve_pose_stream`)
and read it with :class:`~eden.extensions.deployment.object_pose.sources.ZmqPoseSource`, not on a
thread of the control process.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np

import eden as en
from eden.extensions.deployment.object_pose.sources import CameraFrame, PoseEstimator
from eden.utils.geom import T_to_trans_quat

#: ``mask_fn``: one frame -> an ``(H, W)`` bool mask of the object, or ``None`` when it is not in view.
MaskFn = Callable[[CameraFrame], "np.ndarray | None"]


def box_mask(x0: int, y0: int, x1: int, y1: int) -> MaskFn:
    """A ``mask_fn`` that masks a fixed pixel box: for an object placed in a known spot before START."""

    def fn(frame: CameraFrame) -> np.ndarray:
        mask = np.zeros(frame.rgb.shape[:2], dtype=bool)
        mask[y0:y1, x0:x1] = True
        return mask

    return fn


class FoundationPoseEstimator(PoseEstimator):
    """6D pose of a known mesh from RGB-D, registered from a mask and then tracked.

    Parameters
    ----------
    mesh_path : str
        The object's mesh (anything ``trimesh`` loads), the same geometry the scene entity uses.
    mask_fn : callable
        ``frame -> (H, W) bool mask | None``; called only to register. See :func:`box_mask`.
    repo_dir : str | None
        A FoundationPose checkout, prepended to ``sys.path``. ``None``: it is importable already.
    mesh_scale : float
        Multiplies the mesh's vertices into metres (``0.001`` for a mesh authored in millimetres).
    register_iterations, track_iterations : int
        Refinement iterations (FoundationPose's ``est_refine_iter`` / ``track_refine_iter``; 5 and 2 upstream).
    max_jump_m : float
        A tracked pose that moves further than this in one frame is taken as lost, and the next
        frame registers afresh.
    reregister_every : int | None
        Register afresh every this many tracked frames, bounding drift; ``None`` tracks indefinitely.
    min_depth_m : float
        Depth below this counts as invalid (zeroed, as FoundationPose's demo does).
    """

    needs_depth = True

    def __init__(
        self,
        mesh_path: str,
        mask_fn: MaskFn,
        repo_dir: str | None = None,
        mesh_scale: float = 1.0,
        register_iterations: int = 5,
        track_iterations: int = 2,
        max_jump_m: float = 0.05,
        reregister_every: int | None = None,
        min_depth_m: float = 0.001,
    ) -> None:
        if not Path(mesh_path).is_file():
            raise FileNotFoundError(f"FoundationPoseEstimator: mesh_path={mesh_path!r} is not a file.")
        if reregister_every is not None and reregister_every < 1:
            raise ValueError(f"FoundationPoseEstimator: reregister_every={reregister_every!r} must be >= 1, or None.")
        if repo_dir is not None:
            if not (Path(repo_dir) / "estimater.py").is_file():
                raise FileNotFoundError(
                    f"FoundationPoseEstimator: repo_dir={repo_dir!r} has no estimater.py; point it at a FoundationPose "
                    "checkout (git clone https://github.com/NVlabs/FoundationPose)."
                )
            sys.path.insert(0, str(Path(repo_dir).resolve()))
        try:
            import nvdiffrast.torch as dr
            import trimesh
            from estimater import FoundationPose, PoseRefinePredictor, ScorePredictor
        except ImportError as e:
            raise ImportError(
                f"FoundationPoseEstimator could not import FoundationPose ({e}). Fix: build a FoundationPose checkout "
                "per its README (nvdiffrast, the weights under weights/) and pass repo_dir=<checkout>."
            ) from e
        mesh = trimesh.load(mesh_path, force="mesh")
        if mesh_scale != 1.0:
            mesh.apply_scale(mesh_scale)
        extent = float(np.max(mesh.extents))
        if not 0.005 < extent < 2.0:
            en.logger.warning(
                f"[object_pose] {mesh_path}: the mesh is {extent:.4g} across after mesh_scale={mesh_scale:g}; "
                "FoundationPose works in metres, so check mesh_scale (0.001 for millimetres)."
            )
        self._est = FoundationPose(
            model_pts=mesh.vertices,
            model_normals=mesh.vertex_normals,
            mesh=mesh,
            scorer=ScorePredictor(),
            refiner=PoseRefinePredictor(),
            glctx=dr.RasterizeCudaContext(),
            debug=0,
        )
        self.mask_fn = mask_fn
        self.register_iterations, self.track_iterations = int(register_iterations), int(track_iterations)
        self.max_jump_m, self.reregister_every, self.min_depth_m = (
            float(max_jump_m),
            reregister_every,
            float(min_depth_m),
        )
        self.reset()

    def reset(self) -> None:
        self._last_pos: np.ndarray | None = None
        self._tracked_frames = 0

    def _in_image(self, pos: np.ndarray, frame: CameraFrame) -> bool:
        if pos[2] <= 0:
            return False
        u, v, _ = frame.K @ (pos / pos[2])
        h, w = frame.rgb.shape[:2]
        return 0 <= u < w and 0 <= v < h

    def estimate(self, frame: CameraFrame) -> tuple[np.ndarray, np.ndarray, float] | None:
        depth = np.where(frame.depth >= self.min_depth_m, frame.depth, 0.0).astype(np.float32)
        K = frame.K.astype(np.float64)
        due = self.reregister_every is not None and self._tracked_frames >= self.reregister_every
        if self._last_pos is None or due:
            mask = self.mask_fn(frame)
            if mask is None or not mask.any():
                self.reset()
                return None
            pose = self._est.register(
                K=K, rgb=frame.rgb, depth=depth, ob_mask=mask.astype(bool), iteration=self.register_iterations
            )
            self._tracked_frames = 0
        else:
            pose = self._est.track_one(rgb=frame.rgb, depth=depth, K=K, iteration=self.track_iterations)
            self._tracked_frames += 1
        pos, quat = T_to_trans_quat(np.asarray(pose, dtype=np.float64).reshape(4, 4))
        lost = not self._in_image(pos, frame) or (
            self._last_pos is not None and not due and np.linalg.norm(pos - self._last_pos) > self.max_jump_m
        )
        if lost:
            self.reset()  # the next frame registers from a fresh mask
            return None
        self._last_pos = pos
        return pos, quat, float("nan")
