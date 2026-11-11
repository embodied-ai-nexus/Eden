"""Tests for ``eden.utils.image``: the uint8 layout helpers and the openpi-client ``resize_with_pad`` port."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from eden.utils.image import resize_with_pad, to_chw_uint8, to_uint8_hwc


def _rand_uint8(h, w, c=3, seed=0):
    return np.random.default_rng(seed).integers(0, 256, size=(h, w, c), dtype=np.uint8)


def test_to_uint8_hwc_drops_single_batch_axis_and_alpha():
    rgba = torch.from_numpy(_rand_uint8(4, 6, 4)).unsqueeze(0)
    out = to_uint8_hwc(rgba)
    assert out.shape == (4, 6, 3)
    np.testing.assert_array_equal(out, rgba[0, ..., :3].numpy())


def test_to_uint8_hwc_rejects_float_and_batched_inputs():
    with pytest.raises(TypeError, match="uint8"):
        to_uint8_hwc(torch.zeros(1, 4, 4, 3))
    with pytest.raises(ValueError, match="leading axis 1"):
        to_uint8_hwc(torch.zeros(2, 4, 4, 3, dtype=torch.uint8))
    with pytest.raises(ValueError, match="channels"):
        to_uint8_hwc(torch.zeros(4, 4, 2, dtype=torch.uint8))


def test_to_chw_uint8_is_contiguous_transpose():
    hwc = _rand_uint8(3, 5)
    chw = to_chw_uint8(hwc)
    assert chw.shape == (3, 3, 5) and chw.flags["C_CONTIGUOUS"]
    np.testing.assert_array_equal(chw, hwc.transpose(2, 0, 1))


@pytest.mark.parametrize(("h", "w", "th", "tw"), [(12, 16, 8, 8), (16, 12, 8, 8), (10, 10, 12, 20), (8, 8, 8, 8)])
def test_resize_with_pad_matches_openpi_client(h, w, th, tw):
    """The inline port must be bit-exact with ``openpi_client.image_tools.resize_with_pad``."""
    image_tools = pytest.importorskip("openpi_client.image_tools")
    img = _rand_uint8(h, w)
    np.testing.assert_array_equal(resize_with_pad(img, th, tw), image_tools.resize_with_pad(img, th, tw))
    batched = np.stack([img, _rand_uint8(h, w, seed=1)])
    np.testing.assert_array_equal(resize_with_pad(batched, th, tw), image_tools.resize_with_pad(batched, th, tw))


def test_resize_with_pad_letterboxes_without_distortion():
    img = np.full((10, 20, 3), 255, dtype=np.uint8)
    out = resize_with_pad(img, 20, 20)
    assert out.shape == (20, 20, 3)
    # 10x20 -> 10x20 content centred vertically in a 20x20 zero canvas.
    assert out[:5].max() == 0 and out[15:].max() == 0 and out[5:15].min() == 255
