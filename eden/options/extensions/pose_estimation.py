"""Configuration options for pose-estimation extensions (video → human motion).

:class:`PoseEstimatorOptions` mirrors
:class:`~eden.extensions.pose_estimation.base.PoseEstimatorBase`'s attributes, so the shared
fields are type-checked at ``configure()`` time. Backend-specific knobs stay as class
attributes on the backend and arrive as extras — e.g. GVHMR's ``static_cam`` /
``use_dpvo`` / ``f_mm``.
"""

from __future__ import annotations

from eden.options.options import DynamicConfigurableOptions


class PoseEstimatorOptions(DynamicConfigurableOptions):
    """Options shared by every pose-estimation backend.

    Parameters
    ----------
    checkpoint_root : str
        Directory of externally downloaded weights. Empty keeps the backend's own resolution
        (GVHMR: ``$GVHMR_CHECKPOINT_ROOT``, else ``./inputs/checkpoints``).
    ckpt_path : str
        Model checkpoint file. Empty derives it from ``checkpoint_root``.
    output_root : str
        Where the backend writes — and reuses, which is what makes a re-run cheap — its
        per-video intermediates. Empty keeps the backend default, usually relative to cwd.
    motion_fps : float
        Frame rate stamped on the clip: the rate the *output* is expressed in, not
        necessarily the source video's. GVHMR relabels every input to 30 fps without
        resampling frames and integrates root motion at that rate, so 30 is right there even
        for a 60 fps source.
    verbose : bool
        Save the backend's intermediate preprocessing overlays.
    """

    checkpoint_root: str = ""
    ckpt_path: str = ""
    output_root: str = ""
    motion_fps: float = 30.0
    verbose: bool = False
