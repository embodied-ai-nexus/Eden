# Entity Options

Entity options describe one thing in the scene: a mesh or URDF/MJCF asset (`EntityOptions`, `RobotOptions`), a
primitive shape, a terrain, a bank of per-env variants (`GroupedEntityOptions`) or a multi-entity rig
(`CompositeEntityOptions`). `SceneOptions` holds them as named children; each name becomes the key of
`env.entities`. The [entities guide](../../guides/entities.md) explains pose frames, DOF fields and contact
parameters; the [scenes guide](../../guides/scenes-and-terrains.md) covers scenes and terrain.

```python
from eden.options.entities import BoxOptions, SceneOptions

scene_options = SceneOptions(box=BoxOptions(size=(0.1, 0.1, 0.1), default_root_pos=(0.0, 0.0, 0.6)))
```

Quaternion fields (`default_root_quat`, ...) are **wxyz**.

## Scene

::: eden.options.entities.SceneOptions

## Base and asset entities

::: eden.options.entities.BaseEntityOptions

::: eden.options.entities.EntityOptions
    options:
      show_bases: true
      inherited_members: false

::: eden.options.entities.RobotOptions
    options:
      show_bases: true
      inherited_members: false

::: eden.options.entities.MetadataOptions

## Primitives

::: eden.options.entities.PrimitiveOptions
    options:
      show_bases: true
      inherited_members: false

::: eden.options.entities.PlaneOptions
    options:
      show_bases: true
      inherited_members: false

::: eden.options.entities.BoxOptions
    options:
      show_bases: true
      inherited_members: false

::: eden.options.entities.SphereOptions
    options:
      show_bases: true
      inherited_members: false

::: eden.options.entities.CylinderOptions
    options:
      show_bases: true
      inherited_members: false

::: eden.options.entities.RodOptions
    options:
      show_bases: true
      inherited_members: false

## Terrain

::: eden.options.entities.TerrainOptions
    options:
      show_bases: true
      inherited_members: false

::: eden.options.entities.TerrainTermOptions

The terrain generators a `TerrainOptions` composes are on the [robotics utilities page](../utils/robotics.md#terrains).

## Per-env variants

::: eden.options.entities.GroupedEntityOptions
    options:
      show_bases: true
      inherited_members: false

## Composite rigs

The base of the composite robot presets (an arm plus a hand, a humanoid plus two hands); see
[robot presets](robots.md#composite-robots).

::: eden.options.entities.CompositeEntityOptions

## See also

- [Entities guide](../../guides/entities.md)
- [Robots guide](../../guides/robots.md)
- [Entities API](../entities.md)
