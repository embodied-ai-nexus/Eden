"""``HeightFieldTerrain``: height queries on a stair field, and rasterizing terrain options and meshes into one."""

from __future__ import annotations

import numpy as np
import pytest

from eden.extensions.motion_optimization.tcrs.terrain_source import HeightFieldTerrain
from eden.options.entities import EntityOptions, TerrainOptions
from eden.utils.terrains.heightfield_terrains import StairsTerrain


def test_height_is_exact_on_step_centres_and_bilinear_between_cells(stair_terrain):
    terrain = stair_terrain()
    assert terrain.height_at(-0.5, 0.0) == pytest.approx(0.0)  # the apron
    for k in range(3):
        assert terrain.height_at(0.4 + 0.2 * k + 0.1, 0.0) == pytest.approx(0.1 * (k + 1), abs=1e-6)
    assert terrain.height_at(1.5, 0.3) == pytest.approx(0.3, abs=1e-6)  # the top platform
    # Across the first riser the rows at x=0.38 (h=0) and x=0.40 (h=0.1) interpolate linearly, in y too.
    assert terrain.height_at(0.39, 0.0) == pytest.approx(0.05, abs=1e-6)
    assert terrain.height_at(0.385, 0.0) == pytest.approx(0.025, abs=1e-6)
    assert terrain.height_at(0.39, 0.013) == pytest.approx(0.05, abs=1e-6)


def test_height_batch_matches_the_scalar_query_and_defaults_y_to_zero(stair_terrain):
    terrain = stair_terrain()
    rng = np.random.default_rng(0)
    x, y = rng.uniform(-0.9, 2.5, size=64), rng.uniform(-0.9, 0.9, size=64)
    scalar = np.array([terrain.height_at(float(xi), float(yi)) for xi, yi in zip(x, y)])
    np.testing.assert_allclose(terrain.height_batch(x, y), scalar, atol=1e-12)
    np.testing.assert_allclose(terrain.height_batch(x), terrain.height_batch(x, np.zeros_like(x)), atol=1e-12)
    assert terrain.height_at(0.5) == terrain.height_at(0.5, 0.0)


def test_out_of_bounds_returns_oob_height_and_warns_once(stair_terrain, eden_caplog):
    terrain = stair_terrain(oob_height=-1.5)
    assert terrain.height_at(50.0, 0.0) == pytest.approx(-1.5)
    assert terrain.height_at(-50.0, 0.0) == pytest.approx(-1.5)
    out = terrain.height_batch(np.array([0.5, 99.0]), np.array([0.0, 0.0]))
    assert out.tolist() == pytest.approx([0.1, -1.5], abs=1e-6)  # in-bounds entries stay bilinear
    assert sum("outside the terrain bounds" in r.getMessage() for r in eden_caplog.records) == 1


def test_foot_side_penetration_pushes_away_from_a_riser(stair_terrain):
    terrain = stair_terrain()
    # Toes facing the first riser from just before it: pushed back along -x.
    push_x, push_y = terrain.foot_side_penetration(0.35, 0.0, 0.06, 1.0, 0.0)
    assert push_x < 0.0 and push_y == pytest.approx(0.0)
    assert terrain.foot_side_penetration(-0.5, 0.0, 0.06, 1.0, 0.0) == (0.0, 0.0)  # open ground


def test_from_terrain_options_rasterizes_the_generated_stairs():
    options = TerrainOptions(
        terrain_generators=[[StairsTerrain.configure(step_width=0.4, step_height=0.05)]],
        subterrain_size=(4.0, 4.0),
        horizontal_scale=0.1,
        vertical_scale=0.005,
    )
    terrain = HeightFieldTerrain.from_terrain_options(options)
    # The first 0.4 m tread starts at 0.05 m, so x=0.2 is on the first step and x=0.6 on the second.
    assert terrain.height_at(0.2, 2.0) == pytest.approx(0.05, abs=1e-6)
    assert terrain.height_at(0.6, 2.0) == pytest.approx(0.10, abs=1e-6)
    assert terrain.height_at(3.5, 2.0) > terrain.height_at(0.6, 2.0)


@pytest.mark.parametrize(
    "extents, lift, mesh_options, probe, height",
    [
        # A surface below z=0 survives (a miss_height-initialized max-reduction used to flatten it to 0).
        ((2.0, 2.0, 0.4), -0.4, {}, (0.0, 0.0), -0.2),
        # scale doubles the top face to z=0.4, and the root pos adds (1.0, 0, 0.05).
        ((1.0, 1.0, 0.2), 0.1, {"scale": 2.0, "default_root_pos": (1.0, 0.0, 0.05)}, (1.0, 0.0), 0.45),
    ],
)
def test_from_entity_options_rasterizes_a_mesh_top_surface(tmp_path, extents, lift, mesh_options, probe, height):
    trimesh = pytest.importorskip("trimesh")
    box = trimesh.creation.box(extents=extents)
    box.apply_translation((0.0, 0.0, lift))
    path = str(tmp_path / "box.obj")
    box.export(path)
    terrain = HeightFieldTerrain.from_entity_options(EntityOptions(file=path, **mesh_options), resolution=0.05)
    assert terrain.height_at(*probe) == pytest.approx(height, abs=0.02)
    assert np.isfinite(terrain._grid).all()  # the grid itself, since a probe cannot see a NaN elsewhere
