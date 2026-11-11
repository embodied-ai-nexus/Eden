"""Height-field terrain export/import.

Eden terrains are generated in memory (:func:`generate_height_field`) and
normally live only inside the Genesis morph. This module defines a small
on-disk ``.npz`` format so a terrain's height field can be exported once and
re-used by offline tools (e.g. the TCRS terrain-adaptive motion retargeter)
without re-running the generators.

Format (version 1)
------------------
``height_field``
    ``float32 [H, W]`` grid of heights in **meters** (``vertical_scale`` is
    already applied). Axis 0 maps to world +x, axis 1 to world +y — the same
    convention as :func:`generate_height_field` and
    :class:`~eden.entities.terrain.Terrain`.
``horizontal_scale``
    ``float64`` cell size in meters: ``x = origin[0] + i * horizontal_scale``.
``origin``
    ``float64 [3]`` world position of grid corner ``[0, 0]``.
``_format_version``
    Integer format tag; loaders must reject unknown versions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from eden.options.entities import TerrainOptions

HEIGHT_FIELD_FORMAT_VERSION = 1


@dataclass
class HeightFieldData:
    """A height-field grid in meters plus the metadata needed to query it."""

    height_field_m: np.ndarray
    """``float32 [H, W]`` heights in meters; axis 0 → world +x, axis 1 → world +y."""

    horizontal_scale: float
    """Cell size in meters along both axes."""

    origin: tuple[float, float, float] = field(default=(0.0, 0.0, 0.0))
    """World position of grid corner ``[0, 0]``; ``origin[2]`` offsets all heights."""

    def __post_init__(self):
        self.height_field_m = np.asarray(self.height_field_m, dtype=np.float32)
        if self.height_field_m.ndim != 2:
            raise ValueError(f"height_field_m must be 2D [H, W]; got shape {self.height_field_m.shape}.")
        if not np.isfinite(self.height_field_m).all():
            n_bad = int((~np.isfinite(self.height_field_m)).sum())
            raise ValueError(
                f"height_field_m contains {n_bad} non-finite cell(s). Terrain generators fill uncovered "
                "cells with -inf; make sure the generator grid covers the full terrain before exporting."
            )
        if self.horizontal_scale <= 0:
            raise ValueError(f"horizontal_scale must be > 0; got {self.horizontal_scale}.")
        self.origin = tuple(float(v) for v in self.origin)
        if len(self.origin) != 3:
            raise ValueError(f"origin must have 3 components (x, y, z); got {self.origin}.")


def export_height_field(
    path: str,
    height_field_m: np.ndarray,
    horizontal_scale: float,
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> None:
    """Save a height field (in meters) to ``path`` in the versioned ``.npz`` format."""
    if not str(path).endswith(".npz"):
        # np.savez would silently append ".npz", breaking a later load with the
        # original path string.
        raise ValueError(f"Height-field export path must end with '.npz'; got '{path}'.")
    data = HeightFieldData(height_field_m=height_field_m, horizontal_scale=horizontal_scale, origin=origin)
    np.savez_compressed(
        path,
        height_field=data.height_field_m,
        horizontal_scale=np.float64(data.horizontal_scale),
        # float64: world coordinates can be large, and float32 loses ~1 m of
        # precision near 1e7 — the height grid itself stays float32.
        origin=np.asarray(data.origin, dtype=np.float64),
        _format_version=np.int64(HEIGHT_FIELD_FORMAT_VERSION),
    )


def load_height_field(path: str) -> HeightFieldData:
    """Load a height field previously written by :func:`export_height_field`."""
    with np.load(path) as raw:
        missing = {"height_field", "horizontal_scale", "origin", "_format_version"} - set(raw.files)
        if missing:
            raise ValueError(
                f"'{path}' is not an Eden height-field npz: missing key(s) {sorted(missing)}. "
                f"Found keys: {sorted(raw.files)}."
            )
        version = int(raw["_format_version"])
        if version != HEIGHT_FIELD_FORMAT_VERSION:
            raise ValueError(
                f"'{path}' has height-field format version {version}; this Eden version reads "
                f"version {HEIGHT_FIELD_FORMAT_VERSION}. Re-export the terrain."
            )
        return HeightFieldData(
            height_field_m=raw["height_field"],
            horizontal_scale=float(raw["horizontal_scale"]),
            origin=tuple(raw["origin"].tolist()),
        )


def terrain_options_from_height_field_file(path: str) -> "TerrainOptions":
    """Build a ``TerrainOptions`` that reconstructs an exported height field verbatim.

    The returned options place the terrain at the file's origin and reproduce
    the stored grid exactly (float raw field, ``vertical_scale=1.0``), so a
    scene built from them matches what the TCRS
    ``HeightFieldTerrain`` queried during adaptation.
    """
    from eden.options.entities import TerrainOptions
    from eden.utils.terrains.height_field_file_terrain import HeightFieldFileTerrain

    data = load_height_field(path)
    height, width = data.height_field_m.shape
    return TerrainOptions(
        horizontal_scale=data.horizontal_scale,
        vertical_scale=1.0,
        subterrain_size=(
            (height - 1) * data.horizontal_scale,
            (width - 1) * data.horizontal_scale,
        ),
        terrain_generators=[[HeightFieldFileTerrain.configure(file=path)]],
        default_root_pos=data.origin,
    )


def height_field_from_terrain_options(options: "TerrainOptions") -> HeightFieldData:
    """Materialize the height field a :class:`TerrainOptions` would build, in meters.

    Runs the same :func:`generate_height_field` call as
    :meth:`eden.entities.terrain.Terrain._create_morph_from_options`, applies
    ``vertical_scale``, and stamps ``options.default_root_pos`` as the origin.
    """
    import genesis as gs

    from eden.utils.terrains.base import generate_height_field

    if gs.EPS is None:
        # generate_height_field reads gs.EPS, which Genesis only sets at init;
        # without this guard the failure is an opaque TypeError deep inside.
        raise RuntimeError("height_field_from_terrain_options requires an initialized Genesis — call en.init() first.")
    if not options.terrain_generators or not options.terrain_generators[0]:
        raise ValueError(
            "height_field_from_terrain_options: options.terrain_generators must be a non-empty "
            f"rectangular grid of generators; got {options.terrain_generators!r}."
        )
    n_subterrains = (len(options.terrain_generators), len(options.terrain_generators[0]))
    height_field = generate_height_field(
        terrain_generators=options.terrain_generators,
        subterrain_size=options.subterrain_size,
        n_subterrains=n_subterrains,
        horizontal_scale=options.horizontal_scale,
        vertical_scale=options.vertical_scale,
        randomize=options.randomize,
    )
    return HeightFieldData(
        height_field_m=height_field.astype(np.float32) * options.vertical_scale,
        horizontal_scale=options.horizontal_scale,
        origin=tuple(options.default_root_pos),
    )
