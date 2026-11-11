"""``eden.utils.terrains.io``: the height-field ``.npz`` format, and exporting terrain options into it."""

from __future__ import annotations

import numpy as np
import pytest

from eden.options.entities import TerrainOptions
from eden.utils.terrains import generate_height_field
from eden.utils.terrains.heightfield_terrains import StairsTerrain
from eden.utils.terrains.io import (
    HEIGHT_FIELD_FORMAT_VERSION,
    export_height_field,
    height_field_from_terrain_options,
    load_height_field,
    terrain_options_from_height_field_file,
)


def test_height_field_file_round_trip(tmp_path):
    """Export, then rebuild through ``TerrainOptions``: grid, origin and scale all survive, the grid bit for bit."""
    rng = np.random.default_rng(0)
    grid = rng.uniform(-0.5, 1.5, size=(41, 25)).astype(np.float32)
    path = str(tmp_path / "hf.npz")
    export_height_field(path, grid, horizontal_scale=0.1, origin=(-2.0, 1.0, 0.25))

    options = terrain_options_from_height_field_file(path)
    assert options.default_root_pos == (-2.0, 1.0, 0.25)
    rebuilt = generate_height_field(
        terrain_generators=options.terrain_generators,
        subterrain_size=options.subterrain_size,
        n_subterrains=(1, 1),
        horizontal_scale=options.horizontal_scale,
        vertical_scale=options.vertical_scale,
        randomize=options.randomize,
    )
    # vertical_scale is 1.0, so the raw field is metres and must match the export exactly.
    np.testing.assert_array_equal(rebuilt.astype(np.float32), grid)


def _step_grid():
    grid = np.zeros((50, 20), dtype=np.float32)
    grid[25:] = 0.1
    return grid


def test_load_height_field_reads_back_scale_and_origin(tmp_path):
    path = str(tmp_path / "step.npz")
    export_height_field(path, _step_grid(), 0.02, origin=(-1.0, -1.0, 0.0))
    data = load_height_field(path)
    np.testing.assert_array_equal(data.height_field_m, _step_grid())
    assert data.horizontal_scale == pytest.approx(0.02)
    assert data.origin == (-1.0, -1.0, 0.0)


def test_export_rejects_a_non_finite_cell(tmp_path):
    grid = _step_grid()
    grid[3, 7] = -np.inf
    with pytest.raises(ValueError, match="non-finite"):
        export_height_field(str(tmp_path / "bad.npz"), grid, 0.02)


@pytest.mark.parametrize(
    ("arrays", "match"),
    [
        (
            {"origin": np.zeros(3, np.float32), "_format_version": np.int64(HEIGHT_FIELD_FORMAT_VERSION + 1)},
            "format version",
        ),
        ({}, "missing key"),
    ],
    ids=["newer_version", "missing_keys"],
)
def test_load_rejects_a_file_that_is_not_this_format(tmp_path, arrays, match):
    path = str(tmp_path / "other.npz")
    np.savez(path, height_field=_step_grid(), horizontal_scale=np.float64(0.02), **arrays)
    with pytest.raises(ValueError, match=match):
        load_height_field(path)


def test_terrain_options_export_ascending_stairs():
    options = TerrainOptions(
        terrain_generators=[[StairsTerrain.configure(step_width=0.4, step_height=0.05)]],
        subterrain_size=(4.0, 4.0),
        horizontal_scale=0.1,
        vertical_scale=0.005,
    )
    data = height_field_from_terrain_options(options)
    assert np.isfinite(data.height_field_m).all()
    assert data.horizontal_scale == pytest.approx(0.1)
    profile = data.height_field_m[:-1, 0]  # the generator leaves the final boundary row flat
    assert (np.diff(profile) >= 0).all() and profile[-1] > profile[0]
