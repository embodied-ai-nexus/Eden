"""Terrain entity wrapper for height-field and procedural terrain morphs."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
import genesis as gs

import eden as en
from eden.entities.rigid import RigidEntity

#: Squares along one axis of the ``checker.png`` asset — one texture repeat spans this many.
_CHECKER_SQUARES_PER_TILE = 2
#: Checker square size (m) rendered when ``uv_scale`` is left on auto.
_CHECKER_SQUARE_SIZE_M = 1.0
#: Ceiling on how many copies of a texture may be baked to emulate an anisotropic ``uv_scale``
#: (see :meth:`Terrain._split_uv_scale`). 16 copies of a 512 px asset is a 8192 px texture,
#: still inside the smallest ``GL_MAX_TEXTURE_SIZE`` in circulation.
_MAX_TEXTURE_TILES = 16


class Terrain(RigidEntity):
    """A special Entity that represents the terrain."""

    def __init__(self, env, options):
        super().__init__(env, options)
        if self._options.use_checker_surface:
            self.surface = self._create_checker_surface()

    def _create_morph_from_options(self):
        """Create terrain morph from TerrainOptions fields."""
        from eden.utils.terrains.base import generate_terrain_fields

        opts = self._options
        if tuple(opts.default_root_quat) != (1.0, 0.0, 0.0, 0.0):
            raise ValueError(
                f"TerrainOptions.default_root_quat must be the identity (1, 0, 0, 0); got "
                f"{tuple(opts.default_root_quat)}. get_height/bounds assume an axis-aligned height-field grid, "
                "so a rotated terrain would silently return wrong heights."
            )
        n_subterrains = self._n_subterrains()

        # Keep the full generation output: goal-following command terms read the goal/slope
        # side channels off the entity after build (world-framed in post_build).
        self._terrain_fields = generate_terrain_fields(
            terrain_generators=opts.terrain_generators,
            subterrain_size=opts.subterrain_size,
            n_subterrains=n_subterrains,
            horizontal_scale=opts.horizontal_scale,
            vertical_scale=opts.vertical_scale,
            randomize=opts.randomize,
        )
        height_field = self._terrain_fields.height_field

        uv_scale, texture_tiles = self._resolve_uv()
        self._warn_if_uv_truncated()
        if texture_tiles != (1, 1) and opts.surface is not None:
            raise ValueError(
                f"TerrainOptions.uv_scale={opts.uv_scale} needs the texture repeated "
                f"{texture_tiles[0]}:{texture_tiles[1]} in x:y, which Genesis' scalar morph uv_scale "
                "cannot express. Eden carries the difference by tiling the texture, and it only owns "
                "the texture when use_checker_surface is set."
            )

        self.morph = gs.morphs.Terrain(
            # Terrain geometry is baked from the height field at build time and the
            # entity is fixed-base, so the placement must go through the morph —
            # without it, default_root_pos would be silently ignored for terrains.
            pos=tuple(opts.default_root_pos),
            subterrain_size=opts.subterrain_size,
            n_subterrains=n_subterrains,
            horizontal_scale=opts.horizontal_scale,
            vertical_scale=opts.vertical_scale,
            uv_scale=uv_scale,
            randomize=opts.randomize,
            height_field=height_field,
            visualization=opts.visualization,
        )

    def _n_subterrains(self) -> tuple[int, int]:
        """Grid shape of the terrain, in subterrains (rows along x, cols along y)."""
        generators = self._options.terrain_generators
        return len(generators), len(generators[0])

    def _uv_repeats(self) -> tuple[float, float]:
        """How many times the surface texture repeats across the terrain, in x and y."""
        opts = self._options
        if opts.uv_repeats is not None:
            return opts.uv_repeats
        if not opts.use_checker_surface:
            return 1.0, 1.0
        # Auto: size the checker in metres, so its squares stay square on any terrain shape.
        rows, cols = self._n_subterrains()
        per_repeat_m = _CHECKER_SQUARES_PER_TILE * _CHECKER_SQUARE_SIZE_M
        return rows * opts.subterrain_size[0] / per_repeat_m, cols * opts.subterrain_size[1] / per_repeat_m

    def _resolve_uv(self) -> tuple[float, tuple[int, int]]:
        """The morph's scalar ``uv_scale``, plus the texture tiling that carries any anisotropy.

        Pure in the options — no logging, no stashed state — so the morph and the checker
        texture can each ask for it rather than one handing it to the other.
        :meth:`_warn_if_uv_truncated` reports the lossy cases, from the morph path alone.
        """
        return self._split_uv_scale(*self._uv_repeats())

    def _warn_if_uv_truncated(self) -> None:
        """Report a UV mapping that cannot carry the repeats that were asked for.

        Anisotropy is carried as whole copies of the texture, capped by
        :data:`_MAX_TEXTURE_TILES` so a long lane cannot grow the texture back to the size
        keeping the repetition in pixels produced. Past the cap — or on a ratio that is not a
        whole number of copies — the texture repeats at the wrong rate along one axis and the
        pattern renders stretched, which is the very thing per-axis repeats exist to avoid, so
        it must not pass silently. Called once, from the morph path, since that always runs.
        """
        want_x, want_y = self._uv_repeats()
        scale, (tile_x, tile_y) = self._split_uv_scale(want_x, want_y)
        got_x, got_y = scale * tile_x, scale * tile_y
        if math.isclose(got_x, want_x, rel_tol=0.02) and math.isclose(got_y, want_y, rel_tol=0.02):
            return
        en.logger.warning(
            f"terrain texture repeats {want_x:.1f} x {want_y:.1f} were requested but "
            f"{got_x:.1f} x {got_y:.1f} rendered — an aspect ratio past _MAX_TEXTURE_TILES "
            f"({_MAX_TEXTURE_TILES}), or one that is not a whole number of texture copies, cannot be "
            "expressed on Genesis' scalar morph uv_scale. The pattern will look stretched; even out "
            "the terrain's aspect ratio or set uv_scale to a ratio this can carry."
        )

    @staticmethod
    def _split_uv_scale(repeat_x: float, repeat_y: float) -> tuple[float, tuple[int, int]]:
        """Split desired per-axis repeats into what Genesis can express, plus a texture tiling.

        ``gs.morphs.Terrain.uv_scale`` is a single scalar applied to both axes (it multiplies a
        normalized position, see ``genesis/utils/terrain.py``), so it alone cannot repeat a
        texture more often across x than y. The remainder is carried by duplicating the texture
        itself: a texture holding ``t`` copies of the pattern along an axis repeats ``t`` times
        more often there. The scalar is pinned to the *smaller* repeat count so the tiling
        factors stay >= 1, and thus as small as possible.

        Tiling factors are whole copies, so a ratio that is not close to an integer is rounded
        and the pattern is stretched by the remainder (asking for 80 x 22 gives 88 x 22). A ratio
        beyond ``_MAX_TEXTURE_TILES`` is clamped rather than allowed to grow the texture.
        """
        scale = min(repeat_x, repeat_y)
        tile_x, tile_y = (min(max(round(r / scale), 1), _MAX_TEXTURE_TILES) for r in (repeat_x, repeat_y))
        return scale, (tile_x, tile_y)

    def _create_checker_surface(self):
        """Checkerboard surface: a single asset-sized texture, repeated via the UV mapping.

        The repetition deliberately does **not** live in the pixels. Baking it by tiling the
        512 px asset once per checker square grows the texture with terrain *area*: a 5 x 5 grid
        of 32 m subterrains reached 40960 px square, 5 GB in memory, past the
        ``GL_MAX_TEXTURE_SIZE`` of most GPUs — for a 1.9 KB, two-tone source image. Genesis'
        ``uv_scale`` and the renderer's default REPEAT wrapping do the same job for free, so all
        that is baked here is :meth:`_split_uv_scale`'s small anisotropy factor.
        """
        import os

        import numpy as np
        from genesis.utils.misc import get_assets_dir
        from PIL import Image

        checker_image = np.array(Image.open(os.path.join(get_assets_dir(), "textures/checker.png")))
        _, (tile_x, tile_y) = self._resolve_uv()
        if (tile_x, tile_y) != (1, 1):
            checker_image = np.tile(checker_image, (tile_x, tile_y, 1))
        return gs.surfaces.Default(
            diffuse_texture=gs.textures.ImageTexture(
                image_array=checker_image,
            )
        )

    def pre_build(self) -> None:
        super().pre_build()
        (terrain_geom,) = self._entity.geoms

        assert "height_field" in terrain_geom.metadata
        height_field = terrain_geom.metadata["height_field"]
        self._height_field = torch.as_tensor(height_field, device=gs.device, dtype=gs.tc_float)
        assert self._height_field.ndim == 2, f"Height field must be 2D, but got {self._height_field.shape}"
        self._height_field *= self._entity.morph.vertical_scale

        # NOTE: reshape from (width, height) to (height, width) for grid_sample calculation
        # NOTE: we only need one copy since all environments share the same terrain
        self._height_field = self._height_field.T
        # TODO: heterogeneous terrain support
        self._height_field = self._height_field[None, None, :, :].expand(self._env.num_envs, -1, -1, -1)

    def post_build(self) -> None:
        super().post_build()

        # relative=False: world-frame origin (matching the pre-1.1.1 Genesis default).
        self._origin = self._entity.get_pos(relative=False)[0]

        # get_height returns WORLD z: the morph pos z physically raises the surface,
        # so height queries must carry the same offset (the offline exporter's npz
        # already carries origin z the same way). Offset the underlying 2D grid and
        # re-expand so the per-env dimension stays a zero-copy view.
        offset_field = self._height_field[0, 0] + self._origin[2]
        self._height_field = offset_field[None, None, :, :].expand(self._env.num_envs, -1, -1, -1)

        # Calculate total terrain size from subterrain configuration
        subterrain_size = self._entity.morph.subterrain_size
        n_subterrains = self._entity.morph.n_subterrains

        total_x_size = subterrain_size[0] * n_subterrains[0]
        total_y_size = subterrain_size[1] * n_subterrains[1]
        self._size = (total_x_size, total_y_size)

        # Calculate bounds from origin and size
        x_min = self._origin[0]
        y_min = self._origin[1]
        x_max = x_min + total_x_size
        y_max = y_min + total_y_size
        self._bounds = (x_min, x_max, y_min, y_max)

        self._bounds_min_buffer = torch.tensor([x_min, y_min], device=gs.device, dtype=gs.tc_float)
        self._bounds_inv_range_buffer = torch.tensor(
            [1.0 / (x_max - x_min), 1.0 / (y_max - y_min)],
            device=gs.device,
            dtype=gs.tc_float,
        )

        self._build_subterrain_buffers()

    def _build_subterrain_buffers(self) -> None:
        """World-frame per-cell buffers: cell-center origins plus the goal/slope side channels."""
        import numpy as np

        fields = self._terrain_fields
        morph = self._entity.morph
        rows, cols = morph.n_subterrains
        sub_x, sub_y = morph.subterrain_size
        origin = self._origin.cpu().numpy()

        # (rows, cols, 3) cell centers; z sampled from the stitched height field.
        ci, cj = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
        center_x = origin[0] + (ci + 0.5) * sub_x
        center_y = origin[1] + (cj + 0.5) * sub_y
        px = np.clip(
            np.round((ci + 0.5) * sub_x / morph.horizontal_scale).astype(int), 0, fields.height_field.shape[0] - 1
        )
        py = np.clip(
            np.round((cj + 0.5) * sub_y / morph.horizontal_scale).astype(int), 0, fields.height_field.shape[1] - 1
        )
        center_z = origin[2] + fields.height_field[px, py] * morph.vertical_scale
        subterrain_origins = np.stack([center_x, center_y, center_z], axis=-1)
        self._subterrain_origins = torch.as_tensor(subterrain_origins, device=gs.device, dtype=gs.tc_float)

        if fields.goals is not None:
            goals = fields.goals.copy()
            goals[..., 0] += origin[0] + (np.arange(rows) * sub_x)[:, None, None]
            goals[..., 1] += origin[1] + (np.arange(cols) * sub_y)[None, :, None]
            goals[..., 2] += origin[2]
            self._goals = torch.as_tensor(goals, device=gs.device, dtype=gs.tc_float)
            self._goals_count = torch.as_tensor(fields.goals_count, device=gs.device, dtype=torch.long)
        else:
            self._goals = None
            self._goals_count = None

        if fields.slope_vectors is not None:
            self._slope_vectors = torch.as_tensor(fields.slope_vectors, device=gs.device, dtype=gs.tc_float)
        else:
            self._slope_vectors = None

    @property
    def subterrain_origins(self) -> torch.Tensor:
        """(rows, cols, 3) world-frame subterrain cell centers (z from the height field)."""
        return self._subterrain_origins

    @property
    def goals(self) -> torch.Tensor:
        """(rows, cols, max_goals, 3) world-frame course waypoints per subterrain cell.

        Cells with fewer goals are padded by repeating their last real goal (see
        :attr:`goals_count`).
        """
        if self._goals is None:
            raise RuntimeError(
                "The terrain grid produced no goals — use goal-producing (Parkour*) generators for every "
                "cell to consume goal-following terms."
            )
        return self._goals

    @property
    def goals_count(self) -> torch.Tensor:
        """(rows, cols) number of real (unpadded) goals per subterrain cell."""
        if self._goals_count is None:
            raise RuntimeError(
                "The terrain grid produced no goals — use goal-producing (Parkour*) generators for every "
                "cell to consume goal-following terms."
            )
        return self._goals_count

    @property
    def slope_vectors(self) -> torch.Tensor:
        """(rows, cols, 3) normalized per-cell slope directions (zeros where not applicable)."""
        if self._slope_vectors is None:
            raise RuntimeError("The terrain grid produced no slope vectors — only HalfSlopedTerrain sets them.")
        return self._slope_vectors

    @property
    def height_field_units(self):
        """The stitched (X, Y) height field in ``vertical_scale`` units (numpy, dim0 = x)."""
        return self._terrain_fields.height_field

    @property
    def horizontal_scale(self) -> float:
        return self._entity.morph.horizontal_scale

    @property
    def vertical_scale(self) -> float:
        return self._entity.morph.vertical_scale

    def export_height_field(self, path: str) -> None:
        """Export this terrain's height field (in meters) to ``path``.

        Writes the versioned ``.npz`` format defined in
        :mod:`eden.utils.terrains.io`, usable by offline tools (e.g. the TCRS
        terrain-adaptive motion retargeter) without rebuilding the scene.
        """
        from eden.utils.terrains.io import export_height_field

        if not hasattr(self, "_origin"):
            raise RuntimeError("Terrain.export_height_field requires a built scene — call env.build() first.")

        (terrain_geom,) = self._entity.geoms
        height_field_m = terrain_geom.metadata["height_field"] * self._entity.morph.vertical_scale
        export_height_field(
            path,
            height_field_m=height_field_m,
            horizontal_scale=self._entity.morph.horizontal_scale,
            origin=tuple(self._origin.tolist()),
        )

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """Return the terrain limits as an ``(x_min, x_max, y_min, y_max)`` tuple.

        Returns
        -------
        tuple[float, float, float, float]
            The terrain bounds in ``xxyy`` format (matching ``_bounds``).
        """
        return self._bounds

    def get_height(self, grid: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """
        Get the height of the terrain at the given xy coordinates.

        Parameters
        ----------
        grid : torch.Tensor, shape (B, h_out, w_out, 2)
            The grid of xy coordinates to sample the height from: world xy (m) if ``normalize`` is True, else
            already normalized to the terrain bounds as [-1, 1].
        normalize : bool, optional
            Whether to normalize world xy to [-1, 1] here (into a new tensor; ``grid`` is not modified).
            Defaults to True.

        Returns
        -------
        height : torch.Tensor, shape (B, h_out, w_out)
            The world-frame z of the terrain surface at the given xy coordinates.
        """
        if normalize:
            # Normalize coordinates to [-1, 1] range expected by grid_sample
            # norm_xy = 2 * (xy - mins) / (maxs - mins) - 1
            grid = 2 * (grid - self._bounds_min_buffer) * self._bounds_inv_range_buffer - 1.0

        # Border padding mode isn't supported on Mac GPU (mps)
        # https://github.com/pytorch/pytorch/issues/125098
        interpolated = F.grid_sample(
            self._height_field,  # B, 1, height, width
            grid,  # B, h_out, w_out, 2
            mode="bilinear",
            padding_mode="border" if gs.device.type != "mps" else "zeros",
            align_corners=True,
        )  # B, 1, h_out, w_out

        return interpolated.squeeze(1)  # B, h_out, w_out
