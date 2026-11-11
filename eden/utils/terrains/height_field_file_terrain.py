"""Terrain generator that loads an exported height-field ``.npz`` verbatim.

Closes the loop on :mod:`eden.utils.terrains.io`: a terrain exported via
:func:`~eden.utils.terrains.io.export_height_field` can be rebuilt as a scene
entity, so offline tools (e.g. the TCRS motion adapter) and the simulated
scene share exactly one surface. Use
:func:`~eden.utils.terrains.io.terrain_options_from_height_field_file` to get
matching ``TerrainOptions`` in one call.
"""

from __future__ import annotations

from genesis.ext.isaacgym import terrain_utils as tu

from eden.utils.terrains.base import TERRAIN_GENERATOR_REGISTRY, TerrainGenerator


@TERRAIN_GENERATOR_REGISTRY.register()
class HeightFieldFileTerrain(TerrainGenerator):
    """Fill the subterrain with the grid from an exported height-field ``.npz``.

    The enclosing ``TerrainOptions`` must match the file: the same
    ``horizontal_scale``, ``vertical_scale=1.0`` (the stored grid is already
    in meters), and ``subterrain_size = ((H-1)*hs, (W-1)*hs)``. The file's
    ``origin`` is not applied here — set ``default_root_pos`` to it (or build
    the options via ``terrain_options_from_height_field_file``).
    """

    file: str = ""

    def compute(self, subterrain: tu.SubTerrain) -> tu.SubTerrain:
        from eden.utils.terrains.io import load_height_field

        if not self.file:
            raise ValueError("HeightFieldFileTerrain: 'file' is required (path of an exported height-field .npz).")
        data = load_height_field(self.file)
        expected = (subterrain.width, subterrain.length)
        if data.height_field_m.shape != expected:
            raise ValueError(
                f"HeightFieldFileTerrain: '{self.file}' grid is {data.height_field_m.shape} but the "
                f"TerrainOptions imply {expected}; set subterrain_size=((H-1)*hs, (W-1)*hs) with the "
                "file's horizontal_scale."
            )
        if abs(subterrain.horizontal_scale - data.horizontal_scale) > 1e-9:
            raise ValueError(
                f"HeightFieldFileTerrain: horizontal_scale mismatch — file has {data.horizontal_scale}, "
                f"TerrainOptions has {subterrain.horizontal_scale}."
            )
        # The stored grid is meters; raw values are meters/vertical_scale.
        subterrain.height_field_raw[:] = data.height_field_m / subterrain.vertical_scale
        return subterrain
