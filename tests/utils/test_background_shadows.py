"""``eden.utils.background_shadows``: the per-pixel factor that carries the sim's shadows onto a background.

Pure torch on synthetic frames. A frame's three channels all equal the brightness given, so each expected factor
is a ratio of two numbers in the test. The camera that applies the factor is tested in
``tests/entities/test_camera.py``.
"""

from __future__ import annotations

import pytest
import torch

from eden.utils.background_shadows import background_shadow_factor

SURFACE, OTHER_SURFACE, OBJECT = 3, 4, 7


def _frame(brightness, segm, *, dtype=torch.float32, channels=3):
    """An ``(..., channels)`` frame whose colour channels all equal ``brightness``, and its segmentation."""
    brightness = torch.as_tensor(brightness, dtype=torch.float32)
    rgb = brightness[..., None].expand(*brightness.shape, 3)
    if channels == 4:  # an alpha channel the factor must ignore
        rgb = torch.cat([rgb, torch.full((*brightness.shape, 1), 255.0)], dim=-1)
    return rgb.to(dtype).contiguous(), torch.as_tensor(segm)


def _half_shadowed(brightness=200.0):
    """A 4x4 surface whose top-left pair of pixels gets a quarter of the light the rest does."""
    frame = torch.full((4, 4), brightness)
    frame[0, :2] = brightness / 4
    return frame


@pytest.mark.parametrize(
    ("frame", "ids", "strength", "unshadowed"),
    [
        (_frame(torch.full((4, 4), 200.0), torch.full((4, 4), SURFACE)), [SURFACE], 1.0, None),
        # A dark object is not a shadow: only the listed surface ids are read.
        (_frame([[200.0] * 4, [10.0] * 4], [[SURFACE] * 4, [OBJECT] * 4]), [SURFACE], 1.0, None),
        # A dark floor next to a bright table is not in shadow: each id has its own lit level.
        (_frame([[220.0] * 4, [60.0] * 4], [[SURFACE] * 4, [OTHER_SURFACE] * 4]), [SURFACE, OTHER_SURFACE], 1.0, None),
        # A surface out of view in one env.
        (_frame(torch.full((2, 2, 2), 80.0), [[[SURFACE] * 2] * 2, [[OBJECT] * 2] * 2]), [SURFACE], 1.0, None),
        # A shadow, but strength 0.
        (_frame(_half_shadowed(), torch.full((4, 4), SURFACE)), [SURFACE], 0.0, None),
        # A pixel that is black even without shadows says nothing about light.
        (_frame(torch.zeros((1, 2)), torch.full((1, 2), SURFACE)), [SURFACE], 1.0, "same"),
    ],
    ids=["evenly_lit", "dark_object", "surfaces_of_different_colours", "surface_out_of_view", "zero_strength", "black"],
)
def test_nothing_that_is_not_a_shadow_is_darkened(frame, ids, strength, unshadowed):
    rgb, segm = frame
    factor = background_shadow_factor(rgb, segm, ids, strength, unshadowed_rgb=rgb if unshadowed == "same" else None)
    assert torch.equal(factor, torch.ones((*segm.shape, 1)))


@pytest.mark.parametrize(("dtype", "channels"), [(torch.float32, 3), (torch.uint8, 4)], ids=["float_rgb", "uint8_rgba"])
@pytest.mark.parametrize("as_tensor", [False, True], ids=["id_list", "id_tensor"])
def test_a_shadowed_pixel_is_darkened_by_the_light_it_lost_times_strength(dtype, channels, as_tensor):
    """A quarter of the light reaches two pixels, so at strength 0.8 they keep ``1 - 0.8 * 0.75`` of the background."""
    rgb, segm = _frame(_half_shadowed(), torch.full((4, 4), SURFACE), dtype=dtype, channels=channels)
    ids = torch.tensor([SURFACE]) if as_tensor else [SURFACE]

    factor = background_shadow_factor(rgb, segm, ids, strength=0.8)

    expected = torch.ones((4, 4, 1))
    expected[0, :2] = 1.0 - 0.8 * 0.75
    torch.testing.assert_close(factor, expected)


def test_each_env_is_measured_against_its_own_lit_level():
    """Env 1 is lit half as brightly overall, which is lighting, not a shadow; its one darker pixel is."""
    brightness = torch.stack([torch.full((3, 3), 200.0), torch.full((3, 3), 100.0)])
    brightness[1, 0, 0] = 25.0
    rgb, segm = _frame(brightness, torch.full((2, 3, 3), SURFACE))

    factor = background_shadow_factor(rgb, segm, [SURFACE], strength=1.0)

    expected = torch.ones((2, 3, 3, 1))
    expected[1, 0, 0] = 0.25
    torch.testing.assert_close(factor, expected)


def test_an_unshadowed_frame_is_compared_pixel_by_pixel():
    """A falloff the unshadowed frame shares (a point light, a texture) is not shadow; the extra darkening is."""
    unshadowed = torch.tensor([[200.0, 150.0, 100.0, 50.0]])
    shadowed = unshadowed.clone()
    shadowed[0, 1] = 75.0  # half of this pixel's light
    rgb, segm = _frame(shadowed, torch.full((1, 4), SURFACE))
    reference, _ = _frame(unshadowed, segm)

    factor = background_shadow_factor(rgb, segm, [SURFACE], strength=1.0, unshadowed_rgb=reference)

    assert factor[0, :, 0].tolist() == pytest.approx([1.0, 0.5, 1.0, 1.0])


def test_without_an_unshadowed_frame_a_falloff_reads_as_shadow():
    """The documented limit of the estimate, pinned so a change to it is deliberate: 190 is the ramp's 97th percentile."""
    rgb, segm = _frame(torch.linspace(100.0, 200.0, 11)[None], torch.full((1, 11), SURFACE))
    factor = background_shadow_factor(rgb, segm, [SURFACE], strength=1.0)
    assert factor[0, 0, 0] == pytest.approx(100.0 / 190.0)


def test_the_estimate_takes_each_surfaces_lower_97th_percentile_per_env():
    """The lit level is the surface's lower 97th percentile, whatever its pixel count, in each env on its own.

    The counts span a full row down to one pixel, so the top of the row the estimate keeps covers the rank for any
    count. The oracle sorts each surface's pixels independently.
    """
    gen = torch.Generator().manual_seed(0)
    counts = [300, 150, 9, 2, 1]
    # One (1, 300) image per env: the leading axis is the env axis, so each row is measured on its own.
    brightness = 20.0 + 200.0 * torch.rand((len(counts), 1, 300), generator=gen)
    segm = torch.full((len(counts), 1, 300), OBJECT)
    for env, count in enumerate(counts):
        segm[env, 0, torch.randperm(300, generator=gen)[:count]] = SURFACE
    rgb, segm = _frame(brightness, segm)

    factor = background_shadow_factor(rgb, segm, [SURFACE], strength=1.0)[..., 0]

    for env in range(len(counts)):
        on_surface = segm[env] == SURFACE
        ordered = brightness[env][on_surface].sort().values
        lit = ordered[int((ordered.numel() - 1) * 0.97)]
        torch.testing.assert_close(factor[env][on_surface], (brightness[env][on_surface] / lit).clamp(max=1.0))
        assert factor[env][~on_surface].eq(1.0).all()


def test_frames_past_torch_quantiles_input_limit_are_shaded():
    """``torch.quantile`` rejects more than 2**24 entries, which a batch of camera images passes."""
    width = 2**22 + 1
    brightness = torch.full((2, 2, width), 200.0)
    brightness[:, 0, :3] = 50.0
    rgb, segm = _frame(brightness, torch.full((2, 2, width), SURFACE), dtype=torch.uint8)
    assert rgb[..., 0].numel() > 2**24, "test premise"

    factor = background_shadow_factor(rgb, segm, [SURFACE], strength=1.0)

    torch.testing.assert_close(factor[:, 0, :3, 0], torch.full((2, 3), 0.25))
    assert factor[:, 1].eq(1.0).all()


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"strength": -0.1}, "strength must be in"),
        ({"strength": 1.5}, "strength must be in"),
        ({"segm": torch.zeros((3, 3), dtype=torch.long)}, "do not cover the same pixels"),
        ({"unshadowed_rgb": torch.ones((1, 2, 3))}, "does not match rgb"),
    ],
    ids=["strength_below_0", "strength_above_1", "segm_shape", "unshadowed_shape"],
)
def test_malformed_inputs_are_rejected(kwargs, match):
    rgb, segm = _frame(torch.ones((2, 2)), torch.full((2, 2), SURFACE))
    call = {"rgb": rgb, "segm": segm, "surface_ids": [SURFACE], "strength": 1.0} | kwargs
    with pytest.raises(ValueError, match=match):
        background_shadow_factor(**call)
