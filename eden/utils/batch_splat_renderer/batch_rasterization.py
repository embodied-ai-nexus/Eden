"""Batched gsplat rasterization: single-scene multi-camera (``batch_render``) and per-env
(``batch_env_render``).

Vendored from https://github.com/discoverse-dev/GaussianRenderer with Eden edits (tagged
``# EDEN``): ``torch.as_tensor`` camera inputs for zero-copy torch-cuda tensors, an optional
``return_alpha`` for depth/alpha compositing, and the shared ``genesis``/``eden`` geom helpers
in place of the vendored quaternion/point math.
"""

from typing import Optional, Tuple

import numpy as np
import torch
from genesis.utils.geom import transform_by_quat
from gsplat.rendering import rasterization
from torch import Tensor

from eden.utils.geom import quat_mul

from .gaussiandata import GaussianBatchData, GaussianData


@torch.no_grad()
def batch_render(
    gaussians: GaussianData,
    cam_pos: Tensor,  # (Ncam, 3)
    cam_xmat: Tensor,  # (Ncam, 9)
    height: int,
    width: int,
    fovy: np.ndarray,  # (Ncam,) degree
    bg_imgs: Optional[Tensor] = None,  # (Ncam, H, W, 3)
    y_up: bool = True,
    return_alpha: bool = False,  # EDEN
) -> Tuple[Tensor, ...]:
    """Render one shared Gaussian set from ``Ncam`` cameras (shared-scene mode)."""
    device = gaussians.device

    if gaussians.sh.dim() == 2:
        gaussians.sh = gaussians.sh.reshape(gaussians.sh.shape[0], -1, 3).contiguous()
    sh_degree = int(np.round(np.sqrt(gaussians.sh.shape[1]))) - 1

    Ncam = cam_pos.shape[0]
    # EDEN: as_tensor keeps torch-cuda inputs zero-copy; still converts np.
    cam_pos_t = torch.as_tensor(cam_pos, dtype=torch.float32, device=device)  # (N, 3)
    cam_xmat_t = torch.as_tensor(cam_xmat, dtype=torch.float32, device=device).reshape(Ncam, 3, 3)  # (N, 3, 3)
    fovy_t = torch.as_tensor(np.radians(fovy), dtype=torch.float32, device=device)  # (N,)

    # Intrinsics: focal from vertical FOV; square pixels (fx = fy); principal point centered.
    tan_half_fovy = torch.tan(fovy_t / 2.0)
    focal_y = height / (2.0 * tan_half_fovy)
    focal_x = focal_y
    cx = width / 2.0
    cy = height / 2.0

    Ks = torch.zeros((Ncam, 3, 3), dtype=torch.float32, device=device)
    Ks[:, 0, 0] = focal_x
    Ks[:, 1, 1] = focal_y
    Ks[:, 0, 2] = cx
    Ks[:, 1, 2] = cy
    Ks[:, 2, 2] = 1.0

    Tmats = torch.eye(4, dtype=torch.float32, device=device).unsqueeze(0).repeat(Ncam, 1, 1)
    Tmats[:, :3, :3] = cam_xmat_t
    Tmats[:, :3, 3] = cam_pos_t

    # Flip Y and Z columns of rotation (world/OpenGL camera -> gsplat/OpenCV).
    if y_up:
        Tmats[:, 0:3, 1] *= -1
        Tmats[:, 0:3, 2] *= -1
    viewmats = torch.inverse(Tmats)

    renders, alphas, meta = rasterization(
        means=gaussians.xyz,  # [G, 3]
        quats=gaussians.rot,  # [G, 4]
        scales=gaussians.scale,  # [G, 3]
        opacities=gaussians.opacity,  # [G]
        colors=gaussians.sh,  # [G, K, 3]
        viewmats=viewmats,  # [Ncam, 4, 4]
        Ks=Ks,  # [Ncam, 3, 3]
        width=width,
        height=height,
        sh_degree=sh_degree,
        render_mode="RGB+D",
        packed=False,
    )
    color_img = renders[..., :3]  # renders: (Ncam, H, W, 4) RGBD
    depth_img = renders[..., 3:4]

    if bg_imgs is not None:
        if bg_imgs.device != device:
            bg_imgs = bg_imgs.to(device)
        color_img.addcmul_(bg_imgs, 1.0 - alphas)

    if return_alpha:  # EDEN
        return color_img, depth_img, alphas
    return color_img, depth_img


@torch.no_grad()
def batch_env_render(
    gaussians: GaussianBatchData,
    cam_pos: Tensor,  # (Nenv, Ncam, 3)
    cam_xmat: Tensor,  # (Nenv, Ncam, 9)
    height: int,
    width: int,
    fovy: np.ndarray,  # (Nenv, Ncam) degree
    bg_imgs: Optional[Tensor] = None,  # (Nenv, Ncam, H, W, 3)
    minibatch: Optional[int] = None,
    y_up: bool = True,
    return_alpha: bool = False,  # EDEN
) -> Tuple[Tensor, ...]:
    """Render per-environment Gaussians (per-env mode).

    Each environment carries its own transformed Gaussian set, so the leading
    ``Nenv`` axis is a genuine batch dimension for gsplat (not a camera axis).
    """
    device = gaussians.device
    Nenv = cam_pos.shape[0]
    Ncam = cam_pos.shape[1]

    if minibatch is not None and minibatch > 0 and minibatch < Nenv:
        out_color = torch.empty((Nenv, Ncam, height, width, 3), dtype=torch.float32, device=device)
        out_depth = torch.empty((Nenv, Ncam, height, width, 1), dtype=torch.float32, device=device)
        out_alpha = (
            torch.empty((Nenv, Ncam, height, width, 1), dtype=torch.float32, device=device) if return_alpha else None
        )

        for i in range(0, Nenv, minibatch):
            end = min(i + minibatch, Nenv)

            g_slice = GaussianBatchData(
                xyz=gaussians.xyz[i:end],
                rot=gaussians.rot[i:end],
                scale=gaussians.scale[i:end],
                opacity=gaussians.opacity[i:end],
                sh=gaussians.sh[i:end],
            )

            bg_slice = bg_imgs[i:end] if bg_imgs is not None else None

            out = batch_env_render(
                g_slice,
                cam_pos[i:end],
                cam_xmat[i:end],
                height,
                width,
                fovy[i:end] if len(fovy) == Nenv else fovy,
                bg_imgs=bg_slice,
                minibatch=None,
                y_up=y_up,
                return_alpha=return_alpha,
            )
            out_color[i:end] = out[0]
            out_depth[i:end] = out[1]
            if return_alpha:
                out_alpha[i:end] = out[2]
        if return_alpha:
            return out_color, out_depth, out_alpha
        return out_color, out_depth

    if gaussians.sh.dim() == 3:  # (Nenv, N, D)
        gaussians.sh = gaussians.sh.reshape(gaussians.sh.shape[0], gaussians.sh.shape[1], -1, 3).contiguous()
    sh_degree = int(np.round(np.sqrt(gaussians.sh.shape[2]))) - 1

    fovy_t = torch.as_tensor(np.radians(fovy), dtype=torch.float32, device=device)  # (Nenv, Ncam)

    # Intrinsics: focal from vertical FOV; square pixels (fx = fy); principal point centered.
    tan_half_fovy = torch.tan(fovy_t / 2.0)
    focal_y = height / (2.0 * tan_half_fovy)
    focal_x = focal_y
    cx = width / 2.0
    cy = height / 2.0

    Ks = torch.zeros((Nenv, Ncam, 3, 3), dtype=torch.float32, device=device)
    Ks[..., 0, 0] = focal_x
    Ks[..., 1, 1] = focal_y
    Ks[..., 0, 2] = cx
    Ks[..., 1, 2] = cy
    Ks[..., 2, 2] = 1.0

    Tmats = torch.eye(4, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0).repeat(Nenv, Ncam, 1, 1)
    Tmats[..., :3, :3] = torch.as_tensor(cam_xmat, dtype=torch.float32, device=device).reshape(Nenv, Ncam, 3, 3)
    Tmats[..., :3, 3] = torch.as_tensor(cam_pos, dtype=torch.float32, device=device)

    # Flip Y and Z columns of rotation (world/OpenGL camera -> gsplat/OpenCV).
    if y_up:
        Tmats[..., 0:3, 1] *= -1
        Tmats[..., 0:3, 2] *= -1
    viewmats = torch.inverse(Tmats)

    renders, alphas, meta = rasterization(
        means=gaussians.xyz,  # [Nenv, G, 3]
        quats=gaussians.rot,  # [Nenv, G, 4]
        scales=gaussians.scale,  # [Nenv, G, 3]
        opacities=gaussians.opacity,  # [Nenv, G]
        colors=gaussians.sh,  # [Nenv, G, K, 3]
        viewmats=viewmats,  # [Nenv, Ncam, 4, 4]
        Ks=Ks,  # [Nenv, Ncam, 3, 3]
        width=width,
        height=height,
        sh_degree=sh_degree,
        render_mode="RGB+D",
        packed=False,
    )
    color_img = renders[..., :3]  # renders: (Nenv, Ncam, H, W, 4) RGBD
    depth_img = renders[..., 3:4]

    if bg_imgs is not None:
        if bg_imgs.device != device:
            bg_imgs = bg_imgs.to(device)
        color_img.addcmul_(bg_imgs, 1.0 - alphas)

    if return_alpha:  # EDEN
        return color_img, depth_img, alphas
    return color_img, depth_img


@torch.no_grad()
def batch_update_gaussians(
    gaussian_template: GaussianData,
    body_pos: Tensor,  # (Nenv, Nbody, 3)
    body_quat: Tensor,  # (Nenv, Nbody, 4)
    point_to_body_idx: Optional[Tensor],  # (N_points,)
    dynamic_mask: Optional[Tensor],  # (N_points,)
    scalar_first: bool = True,
) -> GaussianBatchData:
    """Rigidly transform template Gaussians by each env's per-body poses.

    xyz/rot are materialized per env: static points (``dynamic_mask == False``,
    e.g. background) keep the template's, dynamic points are moved by their
    body's pose. Scale, opacity and SH are broadcast across envs as strided views.
    """
    device = body_pos.device
    Nenv = body_pos.shape[0]
    Ngs = len(gaussian_template)

    # Materialize a numpy template to tensors once (cache in practice).
    if isinstance(gaussian_template.xyz, np.ndarray):
        tmpl_xyz = torch.tensor(gaussian_template.xyz, dtype=torch.float32, device=device)
        tmpl_rot = torch.tensor(gaussian_template.rot, dtype=torch.float32, device=device)
        tmpl_scale = torch.tensor(gaussian_template.scale, dtype=torch.float32, device=device)
        tmpl_opacity = torch.tensor(gaussian_template.opacity, dtype=torch.float32, device=device)
        tmpl_sh = torch.tensor(gaussian_template.sh, dtype=torch.float32, device=device)
    else:
        tmpl_xyz = gaussian_template.xyz
        tmpl_rot = gaussian_template.rot
        tmpl_scale = gaussian_template.scale
        tmpl_opacity = gaussian_template.opacity
        tmpl_sh = gaussian_template.sh

    xyz_out = tmpl_xyz.unsqueeze(0).expand(Nenv, Ngs, 3).clone()
    rot_out = tmpl_rot.unsqueeze(0).expand(Nenv, Ngs, 4).clone()

    mask = dynamic_mask
    if mask is not None and mask.any():
        body_indices = point_to_body_idx[mask]  # (N_dynamic,)

        # Gather body poses per env: (Nenv, N_dynamic, 3/4)
        pos_expanded = body_pos[:, body_indices]
        quat_expanded = body_quat[:, body_indices]

        # Template properties: (N_dynamic, 3/4) -> (1, N_dynamic, 3/4)
        xyz_ori = tmpl_xyz[mask].unsqueeze(0)
        rot_ori = tmpl_rot[mask].unsqueeze(0)

        if not scalar_first:
            quat_expanded = quat_expanded[..., [3, 0, 1, 2]]  # xyzw -> wxyz

        # Apply transform (broadcasts (1, N_dyn, .) with (Nenv, N_dyn, .))
        xyz_new = transform_by_quat(xyz_ori, quat_expanded) + pos_expanded  # EDEN: was transform_points
        rot_new = quat_mul(quat_expanded, rot_ori)  # EDEN: was quaternion_multiply

        xyz_out[:, mask, :] = xyz_new
        rot_out[:, mask, :] = rot_new

    # Static properties as strided views (not materialized).
    scale_out = tmpl_scale.unsqueeze(0).expand(Nenv, Ngs, 3)
    opacity_out = tmpl_opacity.unsqueeze(0).expand(Nenv, Ngs)
    sh_out = tmpl_sh.unsqueeze(0).expand(Nenv, Ngs, -1, 3)

    return GaussianBatchData(xyz=xyz_out, rot=rot_out, scale=scale_out, opacity=opacity_out, sh=sh_out)
