"""Eden wrapper around GVHMR (world-grounded human motion recovery from monocular video).

GVHMR (`zju3dv/GVHMR <https://github.com/zju3dv/GVHMR>`_, SIGGRAPH Asia 2024) is consumed
through the packaged fork `Kashu7100/eden_gvhmr <https://github.com/Kashu7100/eden_gvhmr>`_
(import name ``hmr4d``). :meth:`GvhmrEstimator.estimate_clip` writes what
``eden retarget soma --body-model smplx`` reads, so video → clip → robot qpos is two steps.
See ``docs/guides/pose-estimation.md`` for setup.

**In-process.** The fork installs beside Eden: it drops upstream's ``pytorch3d`` requirement for
inference, relaxes a stale ``numpy<1.24`` cap, does not compose Hydra configs (so it does not fight
a host that owns Hydra's global state), and carries no VCS dependency, so it ships on PyPI.
``pip install 'eden-world[pose]'`` resolves additively into Eden's environment and this class
simply imports ``hmr4d``.

One consequence worth knowing: GVHMR runs on *Eden's* video decoder and person detector
rather than the versions its numerics were tuned against. Measured one change at a time on the
bundled tennis clip, ``av`` 13 → 17 moves body joints by 3.6° max and ``ultralytics``
8.2.42 → 8.4.113 moves the detected boxes by 8.5 px and body joints by 3.4°; numpy (including
the 1.x/2.x jump), timm and imageio are all bit-identical. The estimate stays valid — it is just
not bit-comparable with the published checkpoint's reference stack.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path

import numpy as np

import eden as en
from eden.extensions.pose_estimation.base import PoseEstimatorBase
from eden.extensions.pose_estimation.gvhmr.convert import to_amass_smplx_clip
from eden.options.extensions.pose_estimation import PoseEstimatorOptions
from eden.utils.distribution import install_hint

#: Per-call overrides accepted by :meth:`GvhmrEstimator.estimate` (defaults come from options).
#: ``render`` is deliberately absent — see the note on overlay videos in the class docstring.
_RECOVER_KEYS = ("static_cam", "use_dpvo", "f_mm", "output_root", "verbose")

#: GVHMR relabels every input video to this rate and integrates root motion at it.
_ASSUMED_VIDEO_FPS = 30.0
_FPS_TOLERANCE = 0.5

#: ``hmr4d.CHECKPOINT_ROOT_ENV``. Spelled out rather than imported so this module keeps working
#: without GVHMR installed — the import guard below is what should report a missing backend.
_CHECKPOINT_ROOT_ENV = "GVHMR_CHECKPOINT_ROOT"

_INSTALL_HINT = (
    f"GVHMR (`hmr4d`, shipped by `eden-gvhmr`) is not installed. Install it with {install_hint('pose')}. Its "
    "checkpoints are a separate download: run `gvhmr-download-checkpoints`. See docs/guides/pose-estimation.md."
)


def _require_hmr4d():
    """Import GVHMR's entry point, turning a missing install into an actionable error."""
    try:
        from hmr4d.demo import GVHMR
    except ImportError as exc:
        raise ImportError(f"{_INSTALL_HINT}\nImport failed with: {exc}") from exc
    return GVHMR


def _to_numpy(tensor) -> np.ndarray:
    """Detach a torch tensor to a contiguous float32 numpy array."""
    return np.ascontiguousarray(tensor.detach().cpu().numpy(), dtype=np.float32)


@contextmanager
def _checkpoint_root_env(root: str):
    """Pin GVHMR's checkpoint-root environment variable to ``root`` for the block, then restore.

    GVHMR resolves *every* external weight through ``$GVHMR_CHECKPOINT_ROOT`` — the tracker, pose
    and feature checkpoints, and the SMPL-X body model behind ``make_smplx`` — and its constructor
    sets that variable process-wide. Two estimators configured with different roots in one process
    would otherwise fight over it: constructing the second silently re-points the first, whose
    model is already cached and so never re-exports its own root. Scoping the variable here makes
    each call resolve against its own configuration and leaves no residue for other GVHMR users in
    the process. An empty ``root`` means "keep GVHMR's own resolution", so it is a no-op.
    """
    if not root:
        yield
        return
    previous = os.environ.get(_CHECKPOINT_ROOT_ENV)
    os.environ[_CHECKPOINT_ROOT_ENV] = str(Path(root).expanduser())
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(_CHECKPOINT_ROOT_ENV, None)
        else:
            os.environ[_CHECKPOINT_ROOT_ENV] = previous


def _video_fps(video: str | Path) -> float | None:
    """Frame rate of the *source* video (GVHMR relabels its working copy to 30 fps)."""
    import av

    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        rate = stream.average_rate or stream.guessed_rate
    return float(rate) if rate else None


class GvhmrEstimator(PoseEstimatorBase):
    """Monocular video → world-grounded SMPL-X motion.

    The shared options (``checkpoint_root``, ``ckpt_path``, ``output_root``, ``motion_fps``,
    ``verbose``) are documented on
    :class:`~eden.options.extensions.pose_estimation.PoseEstimatorOptions`; the fields below
    are GVHMR's own and reach the options as extras.

    The model is loaded once, on the first :meth:`estimate` call, and reused across videos.
    GVHMR is CUDA-only; select a GPU with ``CUDA_VISIBLE_DEVICES``.

    **No overlay rendering.** GVHMR can draw in-camera / world overlay videos, but that path
    needs two dependencies inference does not — pytorch3d (built against your torch/CUDA) and
    chumpy from git — neither of which can be declared, so the ``pose`` extra does not install
    them. Rather than offer a flag that fails after the whole pipeline has run, Eden does not
    expose rendering at all: the product here is the ``.npz``. To eyeball an estimate, install
    those two and use GVHMR's own CLI (``python -m hmr4d.demo``). ``verbose`` still works — its
    bbox / keypoint overlays are drawn with cv2 and need nothing extra.

    Parameters
    ----------
    static_cam : bool
        Skip visual odometry (assume the camera does not move). Much faster, and the right
        choice for tripod footage; wrong for handheld video, where camera motion would be
        absorbed into the body motion.
    use_dpvo : bool
        Use DPVO instead of the default SimpleVO for camera motion. Requires GVHMR's
        ``dpvo`` extra and its checkpoint.
    f_mm : float | None
        Full-frame-equivalent focal length in mm. ``None`` lets GVHMR estimate it.
    """

    static_cam: bool = False
    use_dpvo: bool = False
    f_mm: float | None = None

    def __init__(self, options: PoseEstimatorOptions) -> None:
        # Set before anything that can raise, so close() is safe even on a failed construction.
        self._model = None
        self._skeleton_model = None
        super().__init__(options)
        if abs(self.motion_fps - _ASSUMED_VIDEO_FPS) > _FPS_TOLERANCE:
            raise ValueError(
                f"motion_fps={self.motion_fps:g} but GVHMR always expresses its output at "
                f"{_ASSUMED_VIDEO_FPS:g} fps — it relabels every input to that rate and integrates root motion "
                f"at it. Stamping any other rate mislabels the clip without changing the motion, which is worse "
                f"than the default. Re-encode the source video instead if you need a different rate."
            )

    def _ensure_model(self):
        """Load GVHMR on first use, so constructing the estimator stays cheap."""
        if self._model is None:
            GVHMR = _require_hmr4d()
            self._model = GVHMR(
                ckpt_path=self.ckpt_path or None,
                checkpoint_root=self.checkpoint_root or None,
            )
        return self._model

    def _pelvis_rest_offset(self, betas) -> np.ndarray:
        """Rest-pose pelvis ``J0(betas)`` ``(T, 3)``.

        The Y-up → Z-up conversion needs it to change world frame exactly: SMPL-X rotates the
        body about ``J0``, so the correct root translation is ``R @ (transl + J0) - J0``.

        Call under :func:`_checkpoint_root_env`: ``make_smplx`` finds the body model through the
        checkpoint-root environment variable, so outside the pin it would resolve against whatever
        the process last set.
        """
        if self._skeleton_model is None:
            from hmr4d.utils.smplx_utils import make_smplx

            self._skeleton_model = make_smplx("supermotion")
        return _to_numpy(self._skeleton_model.get_skeleton(betas.float().cpu())[..., 0, :])

    def close(self) -> None:
        """Drop the loaded model and free its GPU memory (idempotent)."""
        self._model = None
        self._skeleton_model = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:  # torch is always present in Eden; be safe during interpreter teardown
            pass

    # ------------------------------------------------------------------ public

    def estimate(self, video: str | Path, **overrides) -> dict:
        """Recover the motion in ``video``, verbatim in GVHMR's own frames.

        ``**overrides`` take any of :data:`_RECOVER_KEYS` for this call only.

        Returns
        -------
        dict
            ``global`` — SMPL-X params in GVHMR's gravity-aligned **Y-up** world
            (``global_orient (T, 3)``, ``body_pose (T, 63)``, ``betas (T, 10)``,
            ``transl (T, 3)``); ``incam`` — the same in the camera frame; ``pelvis_j0
            (T, 3)``; ``K_fullimg``; ``output_dir``; ``video_fps`` (the source rate).
            :meth:`estimate_clip` returns the Z-up, AMASS-style form Eden reads.
        """
        path = Path(video).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Video does not exist: {path}")
        unknown = set(overrides) - set(_RECOVER_KEYS)
        if unknown:
            raise TypeError(f"Unknown estimate() override(s) {sorted(unknown)}; expected {list(_RECOVER_KEYS)}.")

        kwargs = {key: getattr(self, key) for key in _RECOVER_KEYS}
        kwargs.update(overrides)
        kwargs["output_root"] = kwargs["output_root"] or None  # GVHMR keys its defaults on None

        video_path = str(path.resolve())
        with _checkpoint_root_env(self.checkpoint_root):
            # Load first: a missing backend must surface as _require_hmr4d's actionable
            # ImportError, not as `av` failing to import inside _video_fps.
            model = self._ensure_model()

            # Vet the source rate up front. Warning only after `recover()` would mean spending the
            # whole pipeline before telling the caller the clip they get is time-warped.
            fps = _video_fps(video_path)
            if fps is not None and abs(fps - _ASSUMED_VIDEO_FPS) > _FPS_TOLERANCE:
                en.logger.warning(
                    f"{path.name} is {fps:g} fps, but GVHMR relabels every input to {_ASSUMED_VIDEO_FPS:g} fps "
                    f"without resampling and integrates root motion at that rate — the recovered trajectory is "
                    f"time-warped by {fps / _ASSUMED_VIDEO_FPS:.2f}x. Re-encode the video to "
                    f"{_ASSUMED_VIDEO_FPS:g} fps for a physically correct clip."
                )

            out = model.recover(video_path, **kwargs)
            pelvis_j0 = self._pelvis_rest_offset(out["smpl_params_global"]["betas"])

        # Plain numpy on the way out: the caller should not have to care that GVHMR returns
        # CUDA tensors, and `convert` takes numpy so a saved clip stays framework-agnostic.
        return {
            "global": {key: _to_numpy(value) for key, value in out["smpl_params_global"].items()},
            "incam": {key: _to_numpy(value) for key, value in out["smpl_params_incam"].items()},
            "pelvis_j0": pelvis_j0,
            "K_fullimg": _to_numpy(out["K_fullimg"]),
            "output_dir": str(out["output_dir"]),
            "video_fps": fps,
        }

    def estimate_clip(self, video: str | Path, **overrides) -> dict[str, np.ndarray]:
        """Recover ``video`` as an AMASS-style, **Z-up** SMPL-X clip.

        Ready for :func:`~eden.extensions.pose_estimation.gvhmr.convert.save_amass_npz`;
        :func:`~eden.extensions.pose_estimation.gvhmr.convert.to_amass_smplx_clip` lists the fields.
        """
        return to_amass_smplx_clip(self.estimate(video, **overrides), motion_fps=self.motion_fps, source_video=video)
