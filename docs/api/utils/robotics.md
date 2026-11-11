# Robotics and Asset Utilities

Helpers for actuator math, meshes and point clouds, contact aggregation, terrain generation, procedural meshes and
asset download.

## Actuators

Reflected inertia, RPM conversions and the Raibert heuristic that the [actuator presets](../options/actuators.md)
use to derive armature and PD gains.

::: eden.utils.actuator
    options:
      show_root_heading: false

## Meshes and point clouds

::: eden.utils.mesh
    options:
      show_root_heading: false

::: eden.utils.pc
    options:
      show_root_heading: false

::: eden.utils.procedural_meshes
    options:
      show_root_heading: false

## Contacts

The tensor backend of `RigidEntity.get_grouped_contacts`.

::: eden.utils.contacts
    options:
      show_root_heading: false

## Terrains

Terrain generators register in `TERRAIN_GENERATOR_REGISTRY` and compose into a `TerrainOptions`
([entity options](../options/entities.md#terrain)). See the [scenes and terrains guide](../../guides/scenes-and-terrains.md).

::: eden.utils.terrains.base
    options:
      show_root_heading: false

::: eden.utils.terrains.heightfield_terrains
    options:
      show_root_heading: false

::: eden.utils.terrains.procedural_terrains
    options:
      show_root_heading: false

::: eden.utils.terrains.parkour_terrains
    options:
      show_root_heading: false

::: eden.utils.terrains.height_field_file_terrain
    options:
      show_root_heading: false

### Height-field export and import

::: eden.utils.terrains.io
    options:
      show_root_heading: false

## Plane grids

::: eden.utils.plane_grid
    options:
      show_root_heading: false

## Assets

Assets download on first use into `eden.utils.misc.get_assets_dir()`: `$EDEN_ASSETS_DIR` when it is set (it must
be absolute), otherwise `$XDG_CACHE_HOME/eden/assets`, by default `~/.cache/eden/assets`. Fetch an
entity's or a policy's file through its owner (`options.fetch_asset(file)`, `MyPolicy.fetch_asset(file)`) rather than
calling `get_asset_path` with a `local_dir` yourself. See the [assets page](../../getting-started/assets.md).

::: eden.utils.assets
    options:
      show_root_heading: false

## Kinematics-only robot envs

::: eden.utils.genesis_runtime
    options:
      show_root_heading: false

## See also

- [Entities guide](../../guides/entities.md)
- [Robots guide](../../guides/robots.md)
