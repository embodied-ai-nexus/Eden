"""Carry the shadows the sim casts on background surfaces over to a composited background.

A camera's background composite (:class:`~eden.entities.camera_overlays.BackgroundComposite`) replaces the pixels
of its ``entities`` with another image (a static background image or a Gaussian-splat render). The replacement is by
segmentation mask, so the shadows the robot and objects cast onto those surfaces in the sim pass are thrown away with
them, and the objects look like they float above the real table.

:func:`background_shadow_factor` recovers those shadows from the sim pass itself. Each background
surface is still rendered (with shadows) before it is replaced, so its brightness divided by its
brightness without shadows is the fraction of light that reached it. The result is a per-pixel
multiplier for the replacement image. The brightness without shadows comes from one of two places:

* a second render of the same view with shadows turned off (``unshadowed_rgb``). This is exact for
  any surface, texture and lighting, and is what the rasterizer camera uses;
* otherwise, an estimate: a high quantile of each surface's brightness in view, per segmentation id
  and per env. This is exact for a flat, untextured surface with diffuse shading under directional
  lights (every lit pixel is then equally bright), which is what the batch renderer draws; a
  brightness falloff across the surface (a point light, a highlight, a texture) is read as shadow.
  It also needs part of each surface to be lit in every view: the reference is taken from the
  brightest ``1 - _LIT_QUANTILE`` (3 %) of the surface's pixels, so when more of the surface than
  that is in shadow (a wrist camera looking at the table under the gripper) a shadowed pixel
  becomes the reference and the shadow is lost. The camera rejects a textured surface at build
  time, since that one is known up front; the lit fraction depends on the view and cannot be.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

# Quantile of a surface's brightness taken as its unshadowed level: high enough that the lit part
# dominates, below 1.0 so a few specular or aliased pixels do not set the reference.
_LIT_QUANTILE = 0.97


def _masked_quantile(values: torch.Tensor, mask: torch.Tensor, q: float) -> torch.Tensor:
    """Per-row ``q``-quantile of ``values`` over the entries where ``mask`` is set.

    The lower quantile: the masked entry at ascending rank ``floor((count - 1) * q)``. For a high
    ``q`` that entry sits among the top ``(1 - q)`` of the row, so only those are selected
    (``topk``) instead of sorting the whole row: a batch of camera images is ``N x H*W`` entries,
    and a full sort (values plus int64 indices) of that costs hundreds of MB per call at a few
    thousand envs. There is no input-size limit either (``torch.nanquantile`` caps the input at
    2**24 entries). A row with no masked entry returns an arbitrary value; the caller only reads it
    through the same mask.

    Parameters
    ----------
    values : torch.Tensor
        ``(N, M)`` non-negative values.
    mask : torch.Tensor
        ``(N, M)`` boolean mask selecting the entries of each row to take the quantile over.
    q : float
        Quantile in ``[0, 1]``.

    Returns
    -------
    torch.Tensor
        ``(N,)`` quantile of each row.
    """
    n_cols = values.shape[-1]
    count = mask.sum(dim=-1)
    last = (count - 1).clamp(min=0)
    # Rank of the quantile counted from the top of the masked entries of each row.
    from_top = last - (last.to(values.dtype) * q).long()
    # A bound on `from_top` from the shape alone, so the size of the selection needs no GPU sync;
    # the margin covers the float rounding of the product above.
    k = min(n_cols, int((n_cols - 1) * (1.0 - q)) + 3)
    # Unmasked entries become -1, below every (non-negative) masked value, so they rank last.
    top, _ = torch.where(mask, values, -1.0).topk(k, dim=-1)
    return top.gather(-1, from_top.clamp(max=k - 1)[:, None]).squeeze(-1)


def background_shadow_factor(
    rgb: torch.Tensor,
    segm: torch.Tensor,
    surface_ids: Sequence[int] | torch.Tensor,
    strength: float,
    unshadowed_rgb: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-pixel multiplier that puts the sim's shadows on a replacement background.

    Parameters
    ----------
    rgb : torch.Tensor
        Sim-pass colour, ``(H, W, C)`` or ``(N, H, W, C)`` with ``C >= 3``, any numeric dtype.
    segm : torch.Tensor
        Segmentation ids matching ``rgb``, ``(H, W)`` or ``(N, H, W)``.
    surface_ids : Sequence[int] | torch.Tensor
        Segmentation ids of the background surfaces that receive shadows. Each id is normalised on
        its own, so surfaces of different colours can share a frame. A tensor on ``segm``'s device is
        used without a host-to-device copy (a caller rendering every frame should keep one rather than
        pass a list); it is only cast if its dtype differs from ``segm``'s.
    strength : float
        How much a fully shadowed pixel is darkened, in ``[0, 1]``: ``0`` leaves the background
        unchanged, ``1`` turns a pixel that receives no light black.
    unshadowed_rgb : torch.Tensor | None
        The same view rendered without shadows, shaped like ``rgb``. When given, each pixel is compared
        with its own unshadowed brightness; when ``None``, each surface's unshadowed brightness is
        estimated from ``rgb`` (see the module docstring).

    Returns
    -------
    torch.Tensor
        Float multiplier in ``[1 - strength, 1]`` with shape ``segm.shape + (1,)``, equal to ``1``
        outside the surfaces.
    """
    if not 0.0 <= strength <= 1.0:
        raise ValueError(f"strength must be in [0, 1], got {strength}.")
    if rgb.shape[:-1] != segm.shape:
        raise ValueError(f"rgb {tuple(rgb.shape)} and segm {tuple(segm.shape)} do not cover the same pixels.")
    if unshadowed_rgb is not None and unshadowed_rgb.shape != rgb.shape:
        raise ValueError(f"unshadowed_rgb {tuple(unshadowed_rgb.shape)} does not match rgb {tuple(rgb.shape)}.")

    n_envs = 1 if segm.ndim == 2 else segm.shape[0]
    # Channel sums accumulate in float32 directly: `.float()` first would materialise a float copy
    # of the whole (N, H, W, C) frame.
    brightness = rgb[..., :3].sum(dim=-1, dtype=torch.float32).reshape(n_envs, -1)
    segm = segm.reshape(n_envs, -1)
    surface_ids = torch.as_tensor(surface_ids, device=segm.device, dtype=segm.dtype)
    on_surface = torch.isin(segm, surface_ids)

    if unshadowed_rgb is not None:
        lit = unshadowed_rgb[..., :3].sum(dim=-1, dtype=torch.float32).reshape(n_envs, -1)
    else:
        lit = torch.zeros_like(brightness)
        for surface_id in surface_ids:
            mask = segm == surface_id
            lit = torch.where(mask, _masked_quantile(brightness, mask, _LIT_QUANTILE)[:, None], lit)
    # A pixel that is black even without shadows carries no information about light: leave it alone.
    keep = ~(on_surface & (lit > 0))
    # Both are channel sums, so their ratio is the ratio of the channel means.
    light = brightness.div_(lit.clamp_(min=1e-6)).clamp_(max=1.0).masked_fill_(keep, 1.0)

    # 1 - strength * (1 - light), in place.
    return light.mul_(strength).add_(1.0 - strength).reshape(*rgb.shape[:-1], 1)
