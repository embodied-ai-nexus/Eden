"""Camera observation terms (RGB, depth, segmentation, normals, point cloud)."""

from __future__ import annotations
from typing import TYPE_CHECKING

import torch

from eden.managers import OBSERVATION_TERM_REGISTRY

if TYPE_CHECKING:
    from eden.envs.base import RLEnvBase


@OBSERVATION_TERM_REGISTRY.register()
def rgb_image(
    env: RLEnvBase,
    *,
    camera_name: str,
    normalize: bool = False,
    channels_first: bool = False,
) -> torch.Tensor:
    """RGB image from a camera, optionally CNN-ready.

    By default returns the renderer's native output (``(N, H, W, C)``, uint8).
    For feeding a CNN policy (rsl_rl's ``ActorCriticCNN`` expects float
    ``(N, C, H, W)``), set ``normalize=True`` (uint8 -> float in [0, 1]) and
    ``channels_first=True``. Either flag also drops an alpha channel (the
    madrona batch renderer returns RGBA).
    """
    rgb = env.cameras[camera_name].render_rgb()
    if normalize or channels_first:
        rgb = rgb[..., :3]
    if normalize:
        rgb = rgb.float() / 255.0 if rgb.dtype == torch.uint8 else rgb.float()
    if channels_first:
        rgb = rgb.permute(0, 3, 1, 2).contiguous()
    return rgb


@OBSERVATION_TERM_REGISTRY.register()
def depth_image(
    env: RLEnvBase,
    *,
    camera_name: str,
) -> torch.Tensor:
    return env.cameras[camera_name].render_depth()


@OBSERVATION_TERM_REGISTRY.register()
def segmentation_image(
    env: RLEnvBase,
    *,
    camera_name: str,
) -> torch.Tensor:
    return env.cameras[camera_name].render_segm()


@OBSERVATION_TERM_REGISTRY.register()
def normal_image(
    env: RLEnvBase,
    *,
    camera_name: str,
) -> torch.Tensor:
    return env.cameras[camera_name].render_normal()


@OBSERVATION_TERM_REGISTRY.register()
def pointcloud_image(
    env: RLEnvBase,
    *,
    camera_name: str,
    world_frame: bool = False,
) -> torch.Tensor:
    pc, valid_mask = env.cameras[camera_name].render_pointcloud(world_frame=world_frame)
    # Zero invalid/background pixels so inf depth points never reach the policy.
    pc[~valid_mask] = 0.0
    return pc
