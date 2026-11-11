"""Image conversion and resizing helpers.

Eden camera terms (:func:`eden.managers.terms.observations.camera.rgb_image`) return
``(num_envs, H, W, C)`` ``uint8`` tensors. Consumers (policy servers, recorders, ...)
differ only in layout (HWC vs CHW) and in how they want the image resized, so those are
the two axes exposed here:

* :func:`to_uint8_hwc` / :func:`to_chw_uint8` — layout, no resampling.
* :func:`resize_uint8` — plain resize (cv2 bilinear; Isaac-GR00T's SimplerEnv eval).
* :func:`resize_with_pad` — aspect-preserving letterbox resize, a bit-exact port of
  ``openpi_client.image_tools.resize_with_pad`` (pinned by ``tests/utils/test_image.py``)
  so OpenPI-family clients do not need ``openpi_client`` installed.
"""

from __future__ import annotations

import numpy as np
import torch


def to_uint8_hwc(image: torch.Tensor | np.ndarray) -> np.ndarray:
    """``(1, H, W, C)`` or ``(H, W, C)`` -> ``(H, W, 3)`` ``uint8`` numpy.

    A 4-channel (RGBA, as the madrona/Nyx renderers return) input has its alpha channel
    dropped. Any other channel count, dtype, or a batch axis larger than one raises: a
    float image or a batched render reaching a server is a wiring error, not something
    to rescale or slice quietly.
    """
    arr = image.detach().cpu().numpy() if isinstance(image, torch.Tensor) else np.asarray(image)
    if arr.ndim == 4:
        if arr.shape[0] != 1:
            raise ValueError(f"Expected a single-env image with leading axis 1, got shape {arr.shape}.")
        arr = arr[0]
    if arr.ndim != 3:
        raise ValueError(f"Expected an HxWxC image (optionally with a leading batch axis), got shape {arr.shape}.")
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.shape[-1] != 3:
        raise ValueError(f"Expected 3 (or 4 with alpha) channels, got shape {arr.shape}.")
    if arr.dtype != np.uint8:
        raise TypeError(
            f"Expected a uint8 image, got dtype {arr.dtype}. Eden's rgb_image term returns uint8; "
            "convert explicitly at the source if a renderer produces floats."
        )
    return arr


def to_chw_uint8(image: torch.Tensor | np.ndarray) -> np.ndarray:
    """``(1, H, W, C)`` or ``(H, W, C)`` -> contiguous ``(3, H, W)`` ``uint8`` numpy."""
    return np.ascontiguousarray(to_uint8_hwc(image).transpose(2, 0, 1))


def resize_uint8(image_np: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    """Resize an ``(H, W, 3)`` ``uint8`` image to ``image_size=(h, w)`` with cv2 bilinear filtering.

    No-op when the size already matches. Matches ``gr00t/eval/sim/SimplerEnv/simpler_env.py``.
    """
    import cv2

    h, w = image_size
    if image_np.shape[:2] != (h, w):
        image_np = cv2.resize(image_np, (w, h))
    if image_np.dtype != np.uint8:
        image_np = image_np.astype(np.uint8)
    return image_np


def resize_with_pad(images: np.ndarray, height: int, width: int) -> np.ndarray:
    """Letterbox-resize ``[..., H, W, C]`` images to ``(height, width)`` without distortion.

    Port of ``openpi_client.image_tools.resize_with_pad`` (itself replicating
    ``tf.image.resize_with_pad``): scale by the larger of the two ratios with PIL bilinear
    filtering, then paste centred on a zero canvas. Returns the input unchanged when the
    size already matches.
    """
    from PIL import Image

    if images.shape[-3:-1] == (height, width):
        return images

    original_shape = images.shape
    flat = images.reshape(-1, *original_shape[-3:])
    resized = np.stack([_resize_with_pad_pil(Image.fromarray(im), height, width) for im in flat])
    return resized.reshape(*original_shape[:-3], *resized.shape[-3:])


def _resize_with_pad_pil(image, height: int, width: int) -> np.ndarray:
    from PIL import Image

    cur_width, cur_height = image.size
    if cur_width == width and cur_height == height:
        return np.asarray(image)

    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_image = image.resize((resized_width, resized_height), resample=Image.BILINEAR)

    zero_image = Image.new(resized_image.mode, (width, height), 0)
    pad_height = max(0, int((height - resized_height) / 2))
    pad_width = max(0, int((width - resized_width) / 2))
    zero_image.paste(resized_image, (pad_width, pad_height))
    return np.asarray(zero_image)


def convert_to_uint8(img: np.ndarray) -> np.ndarray:
    """``[0, 1]`` float image -> ``uint8``; ``uint8`` input is returned unchanged.

    Port of ``openpi_client.image_tools.convert_to_uint8``. Kept only for the OpenPI
    pipeline, where it is documented as the last preprocessing step; new code should
    hand :func:`to_uint8_hwc` a ``uint8`` render instead.
    """
    if np.issubdtype(img.dtype, np.floating):
        img = (255 * img).astype(np.uint8)
    return img


def prepare_image_padded(image: torch.Tensor | np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    """Eden render -> ``(h, w, 3)`` ``uint8`` via :func:`resize_with_pad` (the OpenPI client pipeline)."""
    img = to_uint8_hwc(image)
    h, w = image_size
    return convert_to_uint8(resize_with_pad(img, h, w))


def prepare_image_resized(image: torch.Tensor | np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    """Eden render -> ``(h, w, 3)`` ``uint8`` via :func:`resize_uint8` (the GR00T client pipeline)."""
    return resize_uint8(to_uint8_hwc(image), image_size)
