"""How a painted copy is graded: RoboPaint's picture rules for its paintings, batched in torch.

A painting is judged by its colour, compared with its target at the placement shown (RoboPaint first registers the
copy to the target within 5 mm; here the placement must match, which is stricter). The colour, structure and edges
are read in a crop round the target's paint (``crop_margin`` beyond it either way): RoboPaint's paintings cover their
canvas, and a small work's bare paper round it would otherwise dilute every measure. Both are blurred first, by a
Gaussian of ``blur`` (m) standard deviation (RoboPaint's 1 mm), so a stroke a hair off does not count as a stroke
missed; then:

* **colour** — the CIELAB colour difference per cell (ΔE*ab, CIE 1976), its mean (``de_mean``) and its 90th
  percentile (``de_p90``);
* **structure** — the SSIM of the lightness ``L*`` (``ssim_l``), with an 11-cell Gaussian window (sigma 1.5 cells);
* **edges** — the correlation of the lightness gradients' magnitudes (Sobel), ``edge_corr``;
* **coverage** — the IoU of the cells painted at all with those the target paints, over the whole sheet,
  ``paint_iou``: a copy must not leave paper bare where the target is painted, nor paint where it is bare.

The hand rules (strokes, path, scribbles) are counted by the metric
(:class:`~eden.tasks.art.painting.custom_terms.PaintingCopy`) as the brush moves.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from eden.utils.plane_grid import PlaneGrid

_RGB_TO_XYZ = (
    (0.4124564, 0.3575761, 0.1804375),
    (0.2126729, 0.7151522, 0.0721750),
    (0.0193339, 0.1191920, 0.9503041),
)
_D65_WHITE = (0.95047, 1.0, 1.08883)


def srgb_to_linear(rgb: torch.Tensor) -> torch.Tensor:
    """Linear RGB of sRGB colours ``rgb (..., 3)`` in ``[0, 1]``."""
    return torch.where(rgb <= 0.04045, rgb / 12.92, ((rgb.clamp_min(0.0) + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(linear: torch.Tensor) -> torch.Tensor:
    """The sRGB of linear RGB colours ``linear (..., 3)`` in ``[0, 1]``: the inverse of :func:`srgb_to_linear`."""
    return torch.where(linear <= 0.0031308, linear * 12.92, 1.055 * linear.clamp_min(0.0031308) ** (1.0 / 2.4) - 0.055)


def srgb_to_lab(rgb: torch.Tensor) -> torch.Tensor:
    """CIELAB ``(..., 3)`` (D65) of sRGB colours ``rgb (..., 3)`` in ``[0, 1]``."""
    channels = srgb_to_linear(rgb).unbind(-1)
    # Weighted with Python floats, not tensors of them: those would be a blocking copy to the GPU on every call.
    xyz = torch.stack(
        [sum(c * channel for c, channel in zip(row, channels)) / white for row, white in zip(_RGB_TO_XYZ, _D65_WHITE)],
        dim=-1,
    )
    delta = 6.0 / 29.0
    f = torch.where(xyz > delta**3, xyz.clamp_min(1e-12) ** (1.0 / 3.0), xyz / (3.0 * delta**2) + 4.0 / 29.0)
    return torch.stack([116.0 * f[..., 1] - 16.0, 500.0 * (f[..., 0] - f[..., 1]), 200.0 * (f[..., 1] - f[..., 2])], -1)


def gaussian_kernel(sigma: float, device) -> torch.Tensor:
    """A normalized 1-D Gaussian of ``sigma`` (cells), ``3 sigma`` either way (at least one cell)."""
    if not sigma > 0.0:
        raise ValueError(f"gaussian_kernel: sigma={sigma} cells; a Gaussian's standard deviation must be positive.")
    radius = max(1, math.ceil(3.0 * sigma))
    x = torch.arange(-radius, radius + 1, dtype=torch.float32, device=device)
    kernel = torch.exp(-(x**2) / (2.0 * sigma**2))
    return kernel / kernel.sum()


def blur(images: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """``images (B, C, rows, cols)`` blurred by the separable ``kernel``, the edges replicated."""
    radius = kernel.numel() // 2
    channels = images.shape[1]
    horizontal = kernel.view(1, 1, 1, -1).expand(channels, 1, 1, -1)
    vertical = kernel.view(1, 1, -1, 1).expand(channels, 1, -1, 1)
    out = F.conv2d(F.pad(images, (radius, radius, 0, 0), mode="replicate"), horizontal, groups=channels)
    return F.conv2d(F.pad(out, (0, 0, radius, radius), mode="replicate"), vertical, groups=channels)


def _sobel_kernels(device) -> torch.Tensor:
    """``(2, 1, 3, 3)``: the Sobel kernels across (``x``) and down (``y``), normalized by 8."""
    kx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], device=device) / 8.0
    return torch.stack([kx, kx.T])[:, None]


def _sobel_magnitude(lightness: torch.Tensor, kernels: torch.Tensor) -> torch.Tensor:
    """``lightness (B, 1, rows, cols)``: the magnitude of its Sobel gradient (:func:`_sobel_kernels`), the edges
    replicated.
    """
    gx, gy = F.conv2d(F.pad(lightness, (1, 1, 1, 1), mode="replicate"), kernels).unbind(1)
    return torch.hypot(gx, gy)[:, None]


@dataclass(frozen=True)
class PaintingRule:
    """RoboPaint's picture thresholds for an acrylic work (its ``rule``), and the blur they are read after.

    A rule no copy could pass is refused: a ΔE limit or a blur that is not positive, or an SSIM, an edge correlation
    or an IoU asked above 1, its best.
    """

    de_mean: float = 2.0
    de_p90: float = 6.0
    ssim_l: float = 0.92
    edge_corr: float = 0.8
    paint_iou: float = 0.95
    #: The blur's standard deviation (m).
    blur: float = 0.001

    def __post_init__(self) -> None:
        for name, unit in (("de_mean", "a ΔE"), ("de_p90", "a ΔE"), ("blur", "a standard deviation (m)")):
            if not getattr(self, name) > 0.0:
                raise ValueError(
                    f"PaintingRule: {name}={getattr(self, name)}; it is {unit} and must be positive (RoboPaint's "
                    f"acrylic rule: de_mean=2.0, de_p90=6.0, blur=0.001)."
                )
        for name in ("ssim_l", "edge_corr", "paint_iou"):
            if not getattr(self, name) <= 1.0:
                raise ValueError(
                    f"PaintingRule: {name}={getattr(self, name)}; it is at most 1 for any copy, so no copy would pass. "
                    "Ask for at most 1 (RoboPaint's acrylic rule: ssim_l=0.92, edge_corr=0.8, paint_iou=0.95)."
                )


class PaintingJudge:
    """The picture rules of :class:`PaintingRule` for ``target (rows, cols, 3)`` (sRGB) on ``grid``, every env at once.

    ``target_painted (rows, cols)`` (bool) is where the target has paint at all; the colour, structure and edges are
    read in the crop of the sheet round it, ``crop_margin`` (m) beyond it either way.
    """

    def __init__(
        self,
        grid: PlaneGrid,
        target: torch.Tensor,
        target_painted: torch.Tensor,
        rule: PaintingRule,
        *,
        device,
        crop_margin: float = 0.005,
    ) -> None:
        self.grid, self.rule = grid, rule
        if not bool(target_painted.any()):
            raise ValueError("PaintingJudge: the target has no painted cell; render the reference on this grid first.")
        self._blur = gaussian_kernel(rule.blur / grid.cell, device)
        self._window = gaussian_kernel(1.5, device)
        painted = target_painted.cpu().numpy()
        rows, cols = np.flatnonzero(painted.any(axis=1)), np.flatnonzero(painted.any(axis=0))
        margin = math.ceil(crop_margin / grid.cell)
        self.crop = (
            slice(max(0, rows[0] - margin), min(grid.rows, rows[-1] + margin + 1)),
            slice(max(0, cols[0] - margin), min(grid.cols, cols[-1] + margin + 1)),
        )
        target = target.to(device=device, dtype=torch.float32)[self.crop]
        self.target_lab = self._lab(target[None])  # (1, 3, crop rows, crop cols)
        self._target_l = self.target_lab[:, :1]
        self._target_l_mean, self._target_l_var = self._moments(self._target_l)
        self._sobel = _sobel_kernels(device)
        self._target_edges = self._centred(_sobel_magnitude(self._target_l, self._sobel))
        self.target_painted = target_painted.to(device)
        self.target_area = int(self.target_painted.sum())

    def _lab(self, rgb: torch.Tensor) -> torch.Tensor:
        """``rgb (B, rows, cols, 3)`` blurred, in CIELAB ``(B, 3, rows, cols)``."""
        blurred = blur(rgb.permute(0, 3, 1, 2), self._blur)
        return srgb_to_lab(blurred.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

    def _moments(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = blur(x, self._window)
        return mean, blur(x * x, self._window) - mean * mean

    @staticmethod
    def _centred(x: torch.Tensor) -> torch.Tensor:
        """``x (B, 1, rows, cols)`` flattened per env, less its mean, and of unit norm."""
        flat = x.flatten(1)
        flat = flat - flat.mean(dim=1, keepdim=True)
        return flat / flat.norm(dim=1, keepdim=True).clamp_min(1e-12)

    def judge(self, rgb: torch.Tensor, painted: torch.Tensor) -> dict[str, torch.Tensor]:
        """Every picture measure of the copies ``rgb (B, rows, cols, 3)`` (sRGB) painted at ``painted (B, rows,
        cols)`` (bool), ``(B,)`` each: ``de_mean``, ``de_p90``, ``ssim_l``, ``edge_corr``, ``paint_iou``, and
        ``passed``: all five rules met.
        """
        rule = self.rule
        lab = self._lab(rgb[:, self.crop[0], self.crop[1]])
        delta_e = (lab - self.target_lab).norm(dim=1).flatten(1)  # (B, cells)
        de_mean = delta_e.mean(dim=1)
        k = max(1, math.ceil(0.9 * delta_e.shape[1]))
        de_p90 = delta_e.kthvalue(k, dim=1).values
        lightness = lab[:, :1]
        mean, var = self._moments(lightness)
        cov = blur(lightness * self._target_l, self._window) - mean * self._target_l_mean
        c1, c2 = (0.01 * 100.0) ** 2, (0.03 * 100.0) ** 2
        ssim = ((2 * mean * self._target_l_mean + c1) * (2 * cov + c2)) / (
            (mean**2 + self._target_l_mean**2 + c1) * (var + self._target_l_var + c2)
        )
        ssim_l = ssim.flatten(1).mean(dim=1)
        edge_corr = (self._centred(_sobel_magnitude(lightness, self._sobel)) * self._target_edges).sum(dim=1)
        overlap = (painted & self.target_painted).flatten(1).sum(dim=1).float()
        union = (painted | self.target_painted).flatten(1).sum(dim=1).float()
        paint_iou = overlap / union
        passed = (
            (de_mean <= rule.de_mean)
            & (de_p90 <= rule.de_p90)
            & (ssim_l >= rule.ssim_l)
            & (edge_corr >= rule.edge_corr)
            & (paint_iou >= rule.paint_iou)
        )
        return {
            "de_mean": de_mean,
            "de_p90": de_p90,
            "ssim_l": ssim_l,
            "edge_corr": edge_corr,
            "paint_iou": paint_iou,
            "passed": passed,
        }
