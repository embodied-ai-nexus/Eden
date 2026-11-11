"""Batched Gaussian-splat rendering for Eden (gsplat backend).

Public entry point: :class:`BatchSplatRenderer`, a batched renderer with
a shared / per-env Gaussian switch. The gsplat rasterization core
(``batch_rasterization``, ``gaussiandata``, ``ply_io``) is vendored from the
MIT-licensed https://github.com/discoverse-dev/GaussianRenderer.
"""

from .gaussiandata import GaussianBatchData, GaussianData
from .ply_io import load_ply
from .renderer import BatchSplatRenderer, SplatEntry

__all__ = [
    "BatchSplatRenderer",
    "SplatEntry",
    "GaussianData",
    "GaussianBatchData",
    "load_ply",
]
