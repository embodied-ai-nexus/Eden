"""Terrain height-query sources for the TCRS pipeline.

Upstream TCRS queries terrain through MuJoCo raycasts against a scene XML
(``RaycastTerrain``). Eden replaces that seam with a height-field-backed
implementation that can be built from any of Eden's terrain descriptions:

- :class:`~eden.options.entities.TerrainOptions` (procedural generator grid),
- a raw :class:`~eden.options.entities.EntityOptions` pointing at a 3D mesh
  file (rasterized to a height field by downward raycasts at construction),
- an exported height-field ``.npz`` (see :mod:`eden.utils.terrains.io`).

The protocol the pipeline actually consumes is small: ``height_at`` /
``height_batch``, plus an *optional*
``foot_side_penetration`` the planner probes with ``hasattr`` for side-wall
push-out. Upstream's ``steps`` list, ``local_heightmap`` and ``half_width_y``
are dead code and are deliberately not ported.

Height queries interpolate the grid bilinearly. This matches the physical
surface: Genesis triangulates the height field into a mesh, so the simulated
terrain *is* piecewise-linear between grid points.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np
from scipy.ndimage import map_coordinates

import eden as en
from eden.utils.terrains.io import HeightFieldData, height_field_from_terrain_options, load_height_field

if TYPE_CHECKING:
    from eden.options.entities import EntityOptions, TerrainOptions


@runtime_checkable
class TerrainSource(Protocol):
    """What the TCRS pipeline requires of a terrain.

    Implementations may additionally provide ``foot_side_penetration``; the
    pipeline probes for it with ``hasattr`` (same contract as upstream).
    """

    def height_at(self, x: float, y: float = 0.0) -> float: ...

    def height_batch(self, x: np.ndarray, y: np.ndarray | None = None) -> np.ndarray: ...


class HeightFieldTerrain:
    """Bilinear height queries over a regular grid, in world coordinates.

    Out-of-bounds queries return ``oob_height`` (upstream returns 0.0 outside
    the scanned scene — keep that default) and log a one-time warning with the
    observed out-of-bounds fraction so a mis-placed clip is visible.
    """

    #: Horizontal march step for side-clearance rays (m). Matches upstream's
    #: 5 mm x-scan resolution; height queries are cheap so fine is fine.
    SIDE_MARCH_STEP = 0.005

    def __init__(
        self,
        height_field_m: np.ndarray,
        horizontal_scale: float,
        origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
        oob_height: float = 0.0,
    ):
        data = HeightFieldData(height_field_m=height_field_m, horizontal_scale=horizontal_scale, origin=origin)
        self._grid = data.height_field_m.astype(np.float64) + data.origin[2]
        self._horizontal_scale = data.horizontal_scale
        self._origin_xy = np.array(data.origin[:2], dtype=np.float64)
        self._oob_height = float(oob_height)
        self._oob_warned = False

    # ------------------------------------------------------------------
    # Constructors, one per Eden terrain description
    # ------------------------------------------------------------------

    @classmethod
    def from_terrain_options(cls, options: "TerrainOptions", oob_height: float = 0.0) -> "HeightFieldTerrain":
        """Build from a procedural :class:`TerrainOptions` (runs the generators)."""
        data = height_field_from_terrain_options(options)
        return cls(
            height_field_m=data.height_field_m,
            horizontal_scale=data.horizontal_scale,
            origin=data.origin,
            oob_height=oob_height,
        )

    @classmethod
    def from_file(cls, path: str, oob_height: float = 0.0) -> "HeightFieldTerrain":
        """Build from an exported height-field ``.npz`` (:mod:`eden.utils.terrains.io`)."""
        data = load_height_field(path)
        return cls(
            height_field_m=data.height_field_m,
            horizontal_scale=data.horizontal_scale,
            origin=data.origin,
            oob_height=oob_height,
        )

    @classmethod
    def from_entity_options(
        cls,
        options: "EntityOptions",
        resolution: float = 0.02,
        miss_height: float = 0.0,
        oob_height: float = 0.0,
    ) -> "HeightFieldTerrain":
        """Rasterize a mesh-file entity to a height field by downward raycasts.

        Applies the entity's ``scale``, ``up``/``front`` alignment and
        ``default_root_pos``/``default_root_quat`` so the grid matches where
        the entity would sit in a scene. Cells the rays miss (holes, overhangs
        seen from below) get ``miss_height`` — never ``-inf``, which would
        poison every mean/min in the planner; a miss fraction is logged.
        """
        import trimesh
        from genesis.utils.geom import quat_to_R

        from eden.utils.geom import align_up_and_front

        if resolution <= 0:
            raise ValueError(f"resolution must be > 0; got {resolution}.")
        file = options.fetch_asset()
        mesh = trimesh.load(file, force="mesh")
        if not isinstance(mesh, trimesh.Trimesh) or mesh.faces.shape[0] == 0:
            raise ValueError(f"'{file}' did not load as a triangle mesh; cannot rasterize a height field from it.")

        # World transform: scale, then up/front alignment, then the user pose —
        # the same composition Entity._create_morph_from_options applies.
        transform = np.eye(4)
        transform[:3, :3] = align_up_and_front(options.up, options.front)
        scale = options.scale if options.scale is not None else 1.0
        mesh.apply_scale(scale)
        mesh.apply_transform(transform)
        pose = np.eye(4)
        pose[:3, :3] = quat_to_R(np.asarray(options.default_root_quat, dtype=np.float64))
        pose[:3, 3] = np.asarray(options.default_root_pos, dtype=np.float64)
        mesh.apply_transform(pose)

        (min_x, min_y, _), (max_x, max_y, max_z) = mesh.bounds
        # Integer cell counts (not np.arange with a float step, which can append a
        # spurious cell past the mesh bounds → an artificial missed border strip).
        nx = int(np.floor((max_x - min_x) / resolution + 1e-9)) + 1
        ny = int(np.floor((max_y - min_y) / resolution + 1e-9)) + 1
        xs = min_x + resolution * np.arange(nx)
        ys = min_y + resolution * np.arange(ny)
        xx, yy = np.meshgrid(xs, ys, indexing="ij")
        n_rays = xx.size
        if n_rays > 4_000_000:
            en.logger.warning(
                f"HeightFieldTerrain.from_entity_options: rasterizing '{file}' casts {n_rays:,} rays "
                f"({nx}x{ny} grid at resolution={resolution} m). This is slow without an accelerated "
                "trimesh ray backend (pyembree/rtree); increase `resolution` if construction stalls."
            )
        ray_origins = np.stack([xx.ravel(), yy.ravel(), np.full(xx.size, max_z + 1.0)], axis=1)
        ray_directions = np.tile([0.0, 0.0, -1.0], (ray_origins.shape[0], 1))

        # multiple_hits=False returns the nearest hit per ray (both trimesh
        # intersectors), which for a from-above downward ray IS the highest —
        # walkable — surface, so no per-ray max reduction is needed.
        locations, index_ray, _ = mesh.ray.intersects_location(ray_origins, ray_directions, multiple_hits=False)
        # Start from -inf so surfaces below miss_height are never clobbered by
        # the fill value; misses are identified by the untouched -inf and only
        # then filled with miss_height.
        height = np.full(xx.size, -np.inf, dtype=np.float64)
        if len(index_ray):
            height[index_ray] = locations[:, 2]
        miss_mask = ~np.isfinite(height)
        height[miss_mask] = float(miss_height)
        n_miss = int(miss_mask.sum())
        if n_miss:
            en.logger.warning(
                f"HeightFieldTerrain.from_entity_options: {n_miss}/{xx.size} raster cells "
                f"({n_miss / xx.size:.1%}) missed '{file}'; filled with miss_height={miss_height}."
            )
        return cls(
            height_field_m=height.reshape(xx.shape),
            horizontal_scale=resolution,
            origin=(float(min_x), float(min_y), 0.0),
            oob_height=oob_height,
        )

    # ------------------------------------------------------------------
    # TerrainSource protocol
    # ------------------------------------------------------------------

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """Terrain extent as ``(x_min, x_max, y_min, y_max)``."""
        h, w = self._grid.shape
        return (
            float(self._origin_xy[0]),
            float(self._origin_xy[0] + (h - 1) * self._horizontal_scale),
            float(self._origin_xy[1]),
            float(self._origin_xy[1] + (w - 1) * self._horizontal_scale),
        )

    def state_bytes(self) -> bytes:
        """Stable byte encoding of everything that determines a height query.

        Grid values, cell size, xy origin and the out-of-bounds fill — callers
        (e.g. the TCRS dataset modifier's result cache) hash these bytes instead
        of reaching into the private fields.
        """
        return b"".join(
            (
                np.ascontiguousarray(self._grid, dtype=np.float64).tobytes(),
                np.float64(self._horizontal_scale).tobytes(),
                np.asarray(self._origin_xy, dtype=np.float64).tobytes(),
                np.float64(self._oob_height).tobytes(),
            )
        )

    def height_at(self, x: float, y: float = 0.0) -> float:
        return float(self.height_batch(np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)))

    def height_batch(self, x: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        y = np.zeros_like(x) if y is None else np.asarray(y, dtype=np.float64)

        i = (x - self._origin_xy[0]) / self._horizontal_scale
        j = (y - self._origin_xy[1]) / self._horizontal_scale
        h, w = self._grid.shape
        in_bounds = (i >= 0) & (i <= h - 1) & (j >= 0) & (j <= w - 1)
        if not in_bounds.all() and not self._oob_warned:
            self._oob_warned = True
            frac = 1.0 - float(np.mean(in_bounds))
            en.logger.warning(
                f"HeightFieldTerrain: {frac:.1%} of height queries fell outside the terrain bounds "
                f"{self.bounds}; returning oob_height={self._oob_height} for those. "
                "Check the clip's terrain placement (terrain offset / origin)."
            )

        i = np.clip(i, 0.0, h - 1)
        j = np.clip(j, 0.0, w - 1)
        # Bilinear interpolation on the pre-clipped coords (probed equivalent to
        # the manual corner math to < 1e-12, incl. h==1/w==1 grids). ravel/reshape
        # keeps 0-d (height_at) queries working.
        z = map_coordinates(self._grid, [i.ravel(), j.ravel()], order=1, mode="nearest").reshape(i.shape)
        return np.where(in_bounds, z, self._oob_height)

    def print_info(self) -> None:
        """Log a terrain summary (upstream prints; Eden routes to the logger)."""
        x_min, x_max, y_min, y_max = self.bounds
        en.logger.info(
            f"HeightFieldTerrain: grid {self._grid.shape[0]}x{self._grid.shape[1]} @ "
            f"{self._horizontal_scale} m, x=[{x_min:.2f}, {x_max:.2f}], y=[{y_min:.2f}, {y_max:.2f}], "
            f"z=[{self._grid.min():.3f}, {self._grid.max():.3f}] m"
        )

    # ------------------------------------------------------------------
    # Optional capability (hasattr-gated by the pipeline, as upstream)
    # ------------------------------------------------------------------

    def side_clearance(self, x: float, y: float, z: float, direction: np.ndarray, max_dist: float = 2.0) -> float:
        """Distance from ``(x, y, z)`` along horizontal ``direction`` to the first terrain wall.

        A "wall" is where the terrain surface rises above ``z``. Marches the
        height field in :data:`SIDE_MARCH_STEP` increments (upstream shoots a
        horizontal ``mj_ray``); returns ``max_dist`` when nothing is hit.
        """
        dir_xy = np.array([float(direction[0]), float(direction[1])], dtype=np.float64)
        norm = np.linalg.norm(dir_xy)
        if norm < 1e-8:
            return max_dist
        dir_xy /= norm
        ts = np.arange(self.SIDE_MARCH_STEP, max_dist + self.SIDE_MARCH_STEP, self.SIDE_MARCH_STEP)
        heights = self.height_batch(x + ts * dir_xy[0], y + ts * dir_xy[1])
        hits = np.nonzero(heights > z)[0]
        if hits.size == 0:
            return max_dist
        return float(ts[hits[0]])

    def foot_side_penetration(
        self,
        ankle_x: float,
        ankle_y: float,
        ankle_z: float,
        fwd_x: float,
        fwd_y: float,
        foot_half_width: float = 0.05,
        foot_half_length: float = 0.12,
    ) -> tuple[float, float]:
        """Push-out correction if the foot rectangle penetrates terrain side walls.

        Port of upstream ``RaycastTerrain.foot_side_penetration``: probes 4
        horizontal directions at sole height and accumulates a push away from
        any wall closer than the foot half-extent. Returns ``(0, 0)`` if clear.
        """
        # Probe near sole level, but at least 5 mm above the LOCAL surface. Deviation
        # from upstream, whose absolute 0.005 m floor assumes the ground sits at world
        # z=0: identical when the local surface is at z≈0, but on sunken terrain the
        # absolute floor probed at wall height and reported phantom penetrations.
        check_z = max(ankle_z - 0.03, self.height_at(ankle_x, ankle_y) + 0.005)

        lat_x, lat_y = -fwd_y, fwd_x
        push_x, push_y = 0.0, 0.0
        checks = [
            (fwd_x, fwd_y, foot_half_length),
            (-fwd_x, -fwd_y, foot_half_length),
            (lat_x, lat_y, foot_half_width),
            (-lat_x, -lat_y, foot_half_width),
        ]
        for dx, dy, half_size in checks:
            # max_dist=half_size: any clearance >= half_size means "no push", so
            # marching further is wasted work (~17x fewer height samples).
            dist = self.side_clearance(ankle_x, ankle_y, check_z, np.array([dx, dy]), max_dist=half_size)
            if dist < half_size:
                penetration = half_size - dist
                push_x -= penetration * dx
                push_y -= penetration * dy
        return push_x, push_y
