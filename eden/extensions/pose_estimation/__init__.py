"""Pose-estimation extensions: recover human motion from monocular video.

The estimate is SMPL-family (AMASS-style) motion — exactly what the retargeting extensions
consume, so ``video → SMPL clip → robot qpos`` is this package followed by
:mod:`eden.extensions.retargeting`, with an ``.npz`` in between.

The estimation packages themselves (e.g. GVHMR's ``hmr4d``) are optional extras, imported
lazily so that ``import eden`` never needs them and a missing one reports the extra to install
rather than a bare ``ModuleNotFoundError``.
"""

from eden.extensions.pose_estimation.base import PoseEstimatorBase
from eden.extensions.pose_estimation.gvhmr import GvhmrEstimator

__all__ = ["GvhmrEstimator", "PoseEstimatorBase"]
