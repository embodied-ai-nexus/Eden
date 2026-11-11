"""Base class for pose-estimation backends (monocular video → human motion)."""

from __future__ import annotations

from abc import abstractmethod
from pathlib import Path

import numpy as np

from eden.options.extensions.pose_estimation import PoseEstimatorOptions
from eden.utils.common import ConfigurableMixin


class PoseEstimatorBase(ConfigurableMixin[PoseEstimatorOptions]):
    """Abstract base class for recovering human motion from video.

    Takes no ``env``, unlike the retargeting and grasp-sampling bases: estimation is offline
    preprocessing whose output is a motion clip, and the robot only enters at retargeting.

    Backends expose the estimate at two levels — :meth:`estimate` keeps the backend's own
    frames and layout (so camera-space work and provenance survive), :meth:`estimate_clip`
    packs it into the AMASS-style Z-up SMPL clip the retargeting extensions consume.

    No registry, as in ``retargeting``: backends are chosen by CLI subcommand, never resolved
    by name from a config.
    """

    checkpoint_root: str = ""
    ckpt_path: str = ""
    output_root: str = ""
    motion_fps: float = 30.0
    verbose: bool = False

    @abstractmethod
    def estimate(self, video: str | Path, **overrides) -> dict:
        """Recover the motion in ``video``, in the backend's own frames and layout."""
        raise NotImplementedError

    @abstractmethod
    def estimate_clip(self, video: str | Path, **overrides) -> dict[str, np.ndarray]:
        """Recover ``video`` as an AMASS-style, Z-up SMPL clip ready for ``np.savez``."""
        raise NotImplementedError

    def close(self) -> None:
        """Release whatever the backend holds (a loaded model, GPU memory); no-op if nothing.

        Always safe to call, so callers can use ``try``/``finally`` regardless of backend.
        """
