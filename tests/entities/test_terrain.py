"""``eden.entities.terrain.Terrain``: placement checks, the checker texture's UV mapping, and ``get_height``.

The UV and validation cases construct ``Terrain`` against a real, unbuilt env: the morph and surface are resolved
at construction, so no scene build is needed until ``get_height``.
"""

from __future__ import annotations

import genesis as gs
import numpy as np
import pytest
import torch

from eden.entities.terrain import Terrain
from eden.envs.base import EnvBase
from eden.options import EnvOptions, SceneOptions
from eden.options.entities import TerrainOptions
from eden.options.surfaces import PlasticSurfaceOptions
from eden.utils.configs import EdenConfig
from eden.utils.terrains import TERRAIN_GENERATOR_REGISTRY
from eden.utils.terrains.io import export_height_field, terrain_options_from_height_field_file
from tests.support.envs import build_env

CHECKER_PX = 512  # the checker asset is 512 x 512


def _flat_grid(rows, cols):
    flat = TERRAIN_GENERATOR_REGISTRY.get("flat_terrain").configure()
    return [[flat] * cols for _ in range(rows)]


@pytest.fixture(scope="module")
def unbuilt_env():
    env = EnvBase.from_config(EdenConfig(env_options=EnvOptions(num_envs=1)), show_viewer=False)
    yield env
    env.destroy()


@pytest.fixture
def terrain(unbuilt_env):
    """``terrain(**TerrainOptions fields)``: a constructed, never-built ``Terrain``."""
    return lambda **options: Terrain(unbuilt_env, TerrainOptions(**options))


def _checker_px(terrain):
    return terrain.surface.diffuse_texture.image_array.shape[:2]


def test_terrain_options_become_a_height_field_morph_with_their_grid(terrain):
    t = terrain(
        terrain_generators=_flat_grid(2, 2), subterrain_size=(6.0, 6.0), horizontal_scale=0.5, vertical_scale=0.01
    )
    assert type(t.morph) is gs.morphs.Terrain
    assert (t.morph.n_subterrains, t.morph.subterrain_size) == ((2, 2), (6.0, 6.0))
    assert (t.morph.horizontal_scale, t.morph.vertical_scale) == (0.5, 0.01)
    assert t.morph.height_field.shape == (int(12.0 / 0.5) + 1,) * 2


def test_a_rotated_terrain_is_rejected(terrain):
    """``get_height`` / ``bounds`` assume an axis-aligned grid, so a rotation would return wrong heights silently."""
    with pytest.raises(ValueError, match="default_root_quat"):
        terrain(default_root_quat=(0.707, 0.0, 0.0, 0.707), terrain_generators=_flat_grid(1, 1))


def test_checker_texture_repeats_through_uv_not_pixels(terrain):
    """The asset is handed over untiled, with the repetition in the UV scale.

    Tiling it once per square grew the texture with terrain *area*: 40960 px square, 5 GB, for this 5 x 5 grid of
    32 m subterrains. 160 m at 1 m squares, two per repeat, is a UV scale of 80.
    """
    t = terrain(use_checker_surface=True, subterrain_size=(32.0, 32.0), terrain_generators=_flat_grid(5, 5))
    assert _checker_px(t) == (CHECKER_PX, CHECKER_PX)
    assert t.morph.uv_scale == 80.0


def test_checker_squares_stay_square_on_a_non_square_terrain(terrain):
    """Per-axis repeats, carried as a scalar UV scale plus whole texture copies.

    Genesis' scalar ``uv_scale`` alone stretches squares with the aspect ratio. 160 x 80 m wants 80 x 40 repeats: a
    UV scale of 40 plus two texture copies along x.
    """
    t = terrain(use_checker_surface=True, subterrain_size=(16.0, 4.0), terrain_generators=_flat_grid(10, 20))
    assert t.morph.uv_scale == 40.0
    assert _checker_px(t) == (2 * CHECKER_PX, CHECKER_PX), "anisotropy costs one extra copy, not an area"


@pytest.mark.parametrize(
    ("options", "uv_scale", "checker_px"),
    [
        ({"uv_scale": 4.0}, 4.0, None),
        ({"use_checker_surface": True, "uv_scale": (12.0, 4.0)}, 4.0, (3 * CHECKER_PX, CHECKER_PX)),
        ({}, 1.0, None),  # auto without the checker: one repeat per axis
    ],
    ids=["scalar", "pair", "auto"],
)
def test_uv_scale_accepts_a_scalar_or_a_pair(terrain, options, uv_scale, checker_px):
    """A pair is split into Genesis' scalar plus a texture tiling carrying the ratio."""
    t = terrain(terrain_generators=_flat_grid(1, 1), **options)
    assert t.morph.uv_scale == uv_scale
    if checker_px is not None:
        assert _checker_px(t) == checker_px


def test_an_anisotropic_uv_scale_needs_a_texture_eden_owns(terrain):
    """Anisotropy is carried by tiling the texture, which only the checker surface lets Eden do."""
    with pytest.raises(ValueError, match="uv_scale"):
        terrain(uv_scale=(8.0, 2.0), surface=PlasticSurfaceOptions(), terrain_generators=_flat_grid(1, 1))
    terrain(uv_scale=8.0, surface=PlasticSurfaceOptions(), terrain_generators=_flat_grid(1, 1))  # a scalar needs none


def test_an_aspect_ratio_past_the_tiling_cap_warns(terrain, eden_caplog):
    """320 x 4 m wants 160:2 repeats, a ratio of 80; the tiling cap (16) truncates it, so the pattern stretches."""
    t = terrain(use_checker_surface=True, subterrain_size=(16.0, 4.0), terrain_generators=_flat_grid(20, 1))
    assert _checker_px(t) == (16 * CHECKER_PX, CHECKER_PX)
    assert "160.0 x 2.0 were requested" in eden_caplog.text


def test_a_ratio_the_tiling_can_carry_does_not_warn(terrain, eden_caplog):
    terrain(use_checker_surface=True, subterrain_size=(16.0, 4.0), terrain_generators=_flat_grid(10, 20))
    terrain(use_checker_surface=True, subterrain_size=(32.0, 32.0), terrain_generators=_flat_grid(5, 5))
    assert "texture repeats" not in eden_caplog.text


@pytest.mark.parametrize("bad", [0.0, -1.0, (4.0, 0.0)])
def test_a_non_positive_uv_scale_is_rejected(bad):
    """``uv_scale`` counts repeats; zero or negative collapses or mirrors the mapping."""
    with pytest.raises(ValueError, match="uv_scale"):
        TerrainOptions(uv_scale=bad)


def test_export_height_field_requires_a_built_scene(terrain, tmp_path):
    with pytest.raises(RuntimeError, match="built scene"):
        terrain(terrain_generators=_flat_grid(1, 1)).export_height_field(str(tmp_path / "hf.npz"))


def test_get_height_matches_the_plane_and_genesis_own_query(tmp_path):
    """The terrain is an exact plane with distinct x and y slopes, on a non-square grid, offset in all three axes.

    A transposed grid, a mis-scaled normalization or a dropped origin each move the answer. Genesis'
    ``get_terrain_height`` interpolates the triangulated collision mesh, an independent implementation of the same
    query: on a plane its barycentric interpolation and Eden's bilinear ``grid_sample`` coincide.
    """
    hs, origin = 0.25, (-2.0, 1.0, 0.25)
    slope_x, slope_y, intercept = 0.3, -0.11, 0.05
    rows, cols = 41, 25
    grid_x = np.arange(rows, dtype=np.float32)[:, None] * hs
    grid_y = np.arange(cols, dtype=np.float32)[None, :] * hs
    path = str(tmp_path / "plane.npz")
    export_height_field(path, slope_x * grid_x + slope_y * grid_y + intercept, horizontal_scale=hs, origin=origin)

    num_envs = 2
    env = build_env(
        EdenConfig(
            env_options=EnvOptions(num_envs=num_envs),
            scene_options=SceneOptions(terrain=terrain_options_from_height_field_file(path)),
        )
    )
    try:
        terrain = env.entities["terrain"]
        x_min, x_max, y_min, y_max = terrain.bounds
        # One cell inside the rim: border clamping (Eden) and NaN-outside (Genesis) would mask an error there.
        torch.manual_seed(0)
        query = torch.rand((num_envs, 5, 3, 2), device=gs.device, dtype=gs.tc_float)
        query[..., 0] = query[..., 0] * (x_max - x_min - 2 * hs) + x_min + hs
        query[..., 1] = query[..., 1] * (y_max - y_min - 2 * hs) + y_min + hs
        expected = slope_x * (query[..., 0] - x_min) + slope_y * (query[..., 1] - y_min) + intercept + origin[2]

        torch.testing.assert_close(terrain.get_height(query), expected, atol=1e-4, rtol=0.0)
        genesis_heights = terrain._entity.get_terrain_height(query.reshape(num_envs, -1, 2).contiguous())
        torch.testing.assert_close(genesis_heights, expected.reshape(num_envs, -1), atol=1e-4, rtol=0.0)
    finally:
        env.destroy()
