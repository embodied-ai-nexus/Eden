"""``eden.utils.terrains.base``: generator grids into height fields and their goal / slope side channels."""

from __future__ import annotations

import numpy as np
import pytest

from eden.utils.terrains import TERRAIN_GENERATOR_REGISTRY, generate_height_field
from eden.utils.terrains.base import generate_terrain_fields
from eden.utils.terrains.io import export_height_field

pytestmark = pytest.mark.requires_genesis  # SubTerrain reads gs.np_float

HS, VS = 0.25, 0.005
LANE_HS = 0.08
# Course generators refuse a strip too short for their course; tests/utils/terrains/test_parkour_terrains.py covers
# them at and above their floors.
COURSE_GENERATORS = (
    "parkour_terrain",
    "parkour_gap_terrain",
    "parkour_hurdle_terrain",
    "parkour_step_terrain",
    "parkour_demo_terrain",
)


def _generator(name, **params):
    return TERRAIN_GENERATOR_REGISTRY.get(name).configure(**params)


def _flat_grid(rows, cols):
    return [[_generator("flat_terrain") for _ in range(cols)] for _ in range(rows)]


@pytest.mark.parametrize("name", [n for n in TERRAIN_GENERATOR_REGISTRY.keys() if n not in COURSE_GENERATORS])
@pytest.mark.parametrize("size_m", [6.0, 8.0, 12.0, 16.0, 20.0, 24.0, 32.0])
def test_every_generator_returns_a_finite_field_of_the_subterrains_shape(name, size_m, tmp_path, subterrain):
    cls = TERRAIN_GENERATOR_REGISTRY.get(name)
    for seed in range(8):
        np.random.seed(seed)
        st = subterrain(size_m)
        shape = st.height_field_raw.shape
        if name == "height_field_file_terrain":  # file-backed: export a matching field for it to read
            path = str(tmp_path / f"hf_{shape[0]}x{shape[1]}.npz")
            export_height_field(path, np.random.uniform(-1.0, 1.0, shape).astype(np.float32), HS)
            options = cls.configure(file=path)
        else:
            options = cls.configure()
        out = cls(options=options).compute(st)
        assert out.height_field_raw.shape == shape
        assert np.isfinite(out.height_field_raw).all()


def test_a_jagged_generator_grid_is_rejected():
    """A non-rectangular grid is a config error, not a silent ``IndexError``."""
    jagged = _flat_grid(2, 2)
    jagged[1] = jagged[1][:1]
    with pytest.raises(ValueError):
        generate_height_field(jagged, (8.0, 8.0), (2, 2), HS, VS, randomize=False)


def test_a_non_square_subterrain_keeps_its_axes():
    """With rows != cols a transposed placement would not even fit the field."""
    hf = generate_height_field([[_generator("parkour_terrain")]], (32.0, 8.0), (1, 1), HS, VS, randomize=False)
    assert hf.shape == (int(32.0 / HS) + 1, int(8.0 / HS) + 1)
    assert np.isfinite(hf).all()
    assert hf.max() > hf.min()


def test_a_raising_cell_names_its_row_and_column():
    """The generator's own message cannot say which cell; a grid is usually heterogeneous.

    The ``extreme_parkour`` teacher varies difficulty down the rows, so row and column point at the config to fix.
    """
    ok, too_long = _generator("parkour_hurdle_terrain", num_stones=6), _generator("parkour_terrain")
    with pytest.raises(ValueError, match=r"terrain cell \(row 0, col 1\).*does not fit"):
        generate_terrain_fields([[ok, too_long]], (16.0, 4.0), (1, 2), LANE_HS, VS, randomize=False)


def test_the_callers_rng_is_restored_when_a_generator_raises():
    """``randomize=False`` seeds numpy globally per cell; a raise in between must not leave the caller on seed 0."""
    np.random.seed(1234)
    before = np.random.get_state()[1].copy()
    with pytest.raises(ValueError, match="does not fit"):
        generate_terrain_fields([[_generator("parkour_terrain")]], (8.0, 8.0), (1, 1), HS, VS, randomize=False)
    np.testing.assert_array_equal(np.random.get_state()[1], before)


def test_course_goals_are_collected_world_ready():
    """Each cell's waypoints come back padded to one shape, with heights, inside the cell and marching down it."""
    grid = [[_generator("parkour_hurdle_terrain", num_stones=6)] * 2 for _ in range(2)]
    fields = generate_terrain_fields(grid, (16.0, 4.0), (2, 2), LANE_HS, VS, randomize=True)
    assert fields.goals.shape == (2, 2, 8, 3)
    assert (fields.goals_count == 8).all()
    goals = fields.goals[0, 0]
    assert np.isfinite(goals).all()
    assert (goals[:, 0] >= 0).all() and (goals[:, 0] < 16.0).all()
    assert (goals[:, 1] >= 0).all() and (goals[:, 1] <= 4.0).all()
    assert (np.diff(goals[:, 0]) > 0).all()


def test_a_shorter_course_is_padded_by_repeating_its_last_goal():
    grid = [[_generator("parkour_hurdle_terrain", num_stones=4), _generator("parkour_hurdle_terrain", num_stones=6)]]
    fields = generate_terrain_fields(grid, (16.0, 4.0), (1, 2), LANE_HS, VS, randomize=True)
    assert fields.goals.shape == (1, 2, 8, 3)
    assert fields.goals_count.tolist() == [[6, 8]]
    np.testing.assert_array_equal(fields.goals[0, 0, 6:], fields.goals[0, 0, [5, 5]])


def test_a_grid_mixing_course_and_plain_generators_is_rejected():
    """Cells without goals would leave a silent gap in the goal tensor."""
    grid = [[_generator("parkour_hurdle_terrain", num_stones=6), _generator("flat_terrain")]]
    with pytest.raises(ValueError, match="Mixed goal-producing"):
        generate_terrain_fields(grid, (16.0, 4.0), (1, 2), LANE_HS, VS, randomize=True)


def test_a_plain_grid_has_no_side_channels_and_the_wrapper_returns_its_field():
    fields = generate_terrain_fields(_flat_grid(1, 1), (8.0, 8.0), (1, 1), HS, VS, randomize=False)
    assert fields.goals is None and fields.goals_count is None and fields.slope_vectors is None
    hf = generate_height_field(_flat_grid(1, 1), (8.0, 8.0), (1, 1), HS, VS, randomize=False)
    np.testing.assert_array_equal(hf, fields.height_field)


def test_slope_vectors_are_collected_per_cell():
    """A sloped cell reports its unit slope direction, a flat one zero; slopes alone do not trip the goals check."""
    grid = [[_generator("half_sloped_terrain"), _generator("flat_terrain")]]
    fields = generate_terrain_fields(grid, (8.0, 8.0), (1, 2), HS, VS, randomize=False)
    assert fields.goals is None
    assert fields.slope_vectors.shape == (1, 2, 3)
    np.testing.assert_allclose(np.linalg.norm(fields.slope_vectors[0, 0]), 1.0, atol=1e-6)
    np.testing.assert_array_equal(fields.slope_vectors[0, 1], 0.0)
