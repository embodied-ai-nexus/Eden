"""Base terrain generator and flat terrain."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import numpy as np
import genesis as gs
from genesis.typing import Vec2FType, Vec2IType
from genesis.ext.isaacgym import terrain_utils as tu

from eden.utils.common import ConfigurableMixin
from eden.utils.registry import Registry

if TYPE_CHECKING:
    from eden.options.entities import TerrainTermOptions


TERRAIN_GENERATOR_REGISTRY = Registry("TERRAIN_GENERATOR")


@dataclasses.dataclass
class TerrainFields:
    """Outputs of a terrain-grid generation pass.

    Beyond the stitched height field, goal-producing generators (the ``Parkour*`` course
    generators) contribute per-cell waypoint ``goals``; ``HalfSlopedTerrain`` contributes a
    ``slope_vector``. Both are optional side channels: grids built from plain generators leave
    them ``None``.
    """

    height_field: np.ndarray
    """(X, Y) stitched height field in ``vertical_scale`` units (dim0 = x, dim1 = y)."""

    goals: np.ndarray | None
    """(rows, cols, max_goals, 3) float32 — per-cell waypoints in **subterrain-local meters**
    (xy from the cell's ``[0, 0]`` corner; z from the stitched height field, in meters).
    Cells with fewer goals are padded by repeating their last real goal. ``None`` when no
    generator in the grid produced goals."""

    goals_count: np.ndarray | None
    """(rows, cols) int64 — number of real (unpadded) goals per cell; ``None`` with ``goals``."""

    slope_vectors: np.ndarray | None
    """(rows, cols, 3) float32 — normalized slope direction per cell, zeros where a generator
    set none; ``None`` when no generator in the grid produced one."""


def generate_terrain_fields(
    terrain_generators: list[list[TerrainTermOptions]],
    subterrain_size: Vec2FType,
    n_subterrains: Vec2IType,
    horizontal_scale: float,
    vertical_scale: float,
    randomize: bool,
) -> TerrainFields:
    """Generate the height field plus goal/slope side channels for the terrain grid.

    ``terrain_generators`` must be a non-empty rectangular grid whose shape
    equals ``n_subterrains`` (rows x cols).
    """
    n_rows = len(terrain_generators)
    if n_rows == 0 or n_rows != n_subterrains[0] or any(len(row) != n_subterrains[1] for row in terrain_generators):
        raise ValueError(
            "terrain_generators must be a non-empty rectangular 2D list with shape "
            f"n_subterrains={tuple(n_subterrains)}; got rows of lengths "
            f"{[len(row) for row in terrain_generators]}."
        )
    subterrain_rows = int(subterrain_size[0] / horizontal_scale + gs.EPS) + 1
    subterrain_cols = int(subterrain_size[1] / horizontal_scale + gs.EPS) + 1
    heightfield = np.full(
        (
            n_subterrains[0] * (subterrain_rows - 1) + 1,
            n_subterrains[1] * (subterrain_cols - 1) + 1,
        ),
        fill_value=float("-inf"),
        dtype=gs.np_float,
    )
    cell_goals: dict[tuple[int, int], np.ndarray] = {}
    cell_slope_vectors: dict[tuple[int, int], np.ndarray] = {}

    for i, j in zip(*map(np.ravel, np.meshgrid(*map(range, n_subterrains), indexing="ij"))):
        subterrain = tu.SubTerrain(
            width=subterrain_rows,
            length=subterrain_cols,
            vertical_scale=vertical_scale,
            horizontal_scale=horizontal_scale,
        )
        if not randomize:
            saved_state = np.random.get_state()
            np.random.seed(0)

        term_option: TerrainTermOptions = terrain_generators[i][j]
        term = TERRAIN_GENERATOR_REGISTRY.get(term_option.name)(options=term_option)
        try:
            subterrain = term.compute(subterrain)
        except ValueError as e:
            # Say which cell. A generator only knows its own name and extent, but a grid is
            # usually heterogeneous — eden/tasks/extreme_parkour varies difficulty down the rows —
            # so the row and column are what point at the offending config.
            raise ValueError(f"terrain cell (row {i}, col {j}): {e}") from e
        finally:
            # compute() can raise (a parkour course that does not fit its subterrain), and the
            # seed(0) above is process-global: without the finally the caller's RNG stream would
            # be left mid-way through the seed-0 stream, silently rerolling every later draw.
            if not randomize:
                np.random.set_state(saved_state)

        data = subterrain.height_field_raw
        subterrain_heightfield = heightfield[
            i * (subterrain_rows - 1) : (i + 1) * (subterrain_rows - 1) + 1,
            j * (subterrain_cols - 1) : (j + 1) * (subterrain_cols - 1) + 1,
        ]
        subterrain_heightfield[:] = np.maximum(subterrain_heightfield, data)

        goals = getattr(subterrain, "goals", None)
        if goals is not None:
            cell_goals[(int(i), int(j))] = np.asarray(goals, dtype=np.float32)
        slope_vector = getattr(subterrain, "slope_vector", None)
        if slope_vector is not None:
            cell_slope_vectors[(int(i), int(j))] = np.asarray(slope_vector, dtype=np.float32)

    return TerrainFields(
        height_field=heightfield,
        **_assemble_goals(cell_goals, terrain_generators, n_subterrains, heightfield, horizontal_scale, vertical_scale),
        slope_vectors=_assemble_slope_vectors(cell_slope_vectors, n_subterrains),
    )


def _assemble_goals(
    cell_goals: dict[tuple[int, int], np.ndarray],
    terrain_generators: list[list[TerrainTermOptions]],
    n_subterrains: Vec2IType,
    heightfield: np.ndarray,
    horizontal_scale: float,
    vertical_scale: float,
) -> dict:
    """Pack per-cell goal lists into a padded (rows, cols, max_goals, 3) array with z filled in."""
    if not cell_goals:
        return {"goals": None, "goals_count": None}

    n_cells = n_subterrains[0] * n_subterrains[1]
    if len(cell_goals) != n_cells:
        missing = [
            f"({i}, {j}): {terrain_generators[i][j].name}"
            for i in range(n_subterrains[0])
            for j in range(n_subterrains[1])
            if (i, j) not in cell_goals
        ]
        raise ValueError(
            "Mixed goal-producing and non-goal terrain grids are not supported: every cell must use a "
            "goal-producing (Parkour*) generator, or none. Cells without goals: " + ", ".join(missing)
        )

    # Per-cell index stride in the stitched field (cells share their border row/col).
    stride_x = (heightfield.shape[0] - 1) // n_subterrains[0]
    stride_y = (heightfield.shape[1] - 1) // n_subterrains[1]
    max_goals = max(g.shape[0] for g in cell_goals.values())
    goals = np.zeros((n_subterrains[0], n_subterrains[1], max_goals, 3), dtype=np.float32)
    goals_count = np.zeros((n_subterrains[0], n_subterrains[1]), dtype=np.int64)

    for (i, j), g in cell_goals.items():
        if g.ndim != 2 or g.shape[1] != 2 or g.shape[0] == 0:
            raise ValueError(
                f"subterrain ({i}, {j}) ({terrain_generators[i][j].name}) produced goals with shape "
                f"{g.shape}; expected a non-empty (N, 2) array of local xy meters."
            )
        padded = np.concatenate([g, np.repeat(g[-1:], max_goals - g.shape[0], axis=0)], axis=0)
        # z from the stitched height field at the goal's global cell index (meters).
        gx = np.clip(np.round(padded[:, 0] / horizontal_scale).astype(int) + i * stride_x, 0, heightfield.shape[0] - 1)
        gy = np.clip(np.round(padded[:, 1] / horizontal_scale).astype(int) + j * stride_y, 0, heightfield.shape[1] - 1)
        goals[i, j, :, :2] = padded
        goals[i, j, :, 2] = heightfield[gx, gy] * vertical_scale
        goals_count[i, j] = g.shape[0]

    return {"goals": goals, "goals_count": goals_count}


def _assemble_slope_vectors(
    cell_slope_vectors: dict[tuple[int, int], np.ndarray], n_subterrains: Vec2IType
) -> np.ndarray | None:
    """Pack per-cell slope vectors into (rows, cols, 3), zeros where a generator set none."""
    if not cell_slope_vectors:
        return None
    slope_vectors = np.zeros((n_subterrains[0], n_subterrains[1], 3), dtype=np.float32)
    for (i, j), v in cell_slope_vectors.items():
        slope_vectors[i, j] = v
    return slope_vectors


def generate_height_field(
    terrain_generators: list[list[TerrainTermOptions]],
    subterrain_size: Vec2FType,
    n_subterrains: Vec2IType,
    horizontal_scale: float,
    vertical_scale: float,
    randomize: bool,
) -> np.ndarray:
    """Generate only the stitched height field (see :func:`generate_terrain_fields`)."""
    return generate_terrain_fields(
        terrain_generators=terrain_generators,
        subterrain_size=subterrain_size,
        n_subterrains=n_subterrains,
        horizontal_scale=horizontal_scale,
        vertical_scale=vertical_scale,
        randomize=randomize,
    ).height_field


class TerrainGenerator(ConfigurableMixin):
    """
    Terrain generator for Eden, wrapping Genesis' terrain utils.

    Note: Uses unparameterized ConfigurableMixin to avoid a circular import
    with eden.options.entities. _options_class_ is resolved lazily.
    """

    @classmethod
    def configure(cls, **kwargs):
        if cls._options_class_ is None:
            from eden.options.entities import TerrainTermOptions

            cls._options_class_ = TerrainTermOptions
        return super().configure(**kwargs)

    def __init__(self, options: TerrainTermOptions):
        ConfigurableMixin.__init__(self, options=options)

    def compute(self, subterrain: tu.SubTerrain) -> tu.SubTerrain:
        return subterrain


@TERRAIN_GENERATOR_REGISTRY.register()
class FlatTerrain(TerrainGenerator):
    """Flat terrain generator."""

    def compute(self, subterrain: tu.SubTerrain) -> tu.SubTerrain:
        subterrain.height_field_raw *= 0.0
        return subterrain
