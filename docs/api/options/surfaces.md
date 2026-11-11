# Surface Options

Surface options describe how an entity renders (colour, roughness, metalness, emission, textures). Genesis
surfaces cannot be created before `gs.init()`, so `SurfaceOptions` records the specification in the config and
creates the Genesis surface when the scene is built. Set one as an entity's `surface` field.

```python
from eden.options.surfaces import PlasticSurfaceOptions
```

::: eden.options.surfaces
    options:
      show_root_heading: false

## See also

- [Sensors and rendering guide](../../guides/sensors-and-rendering.md)
